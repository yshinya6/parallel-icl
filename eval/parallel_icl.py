import argparse
import copy
import json
import math
import os
import pdb
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import shortuuid
import torch
import torch.distributed
import torch.nn.functional as F
import transformers
from huggingface_hub import login
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from torch_kmeans import KMeans
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from decoding.parallel import _sample
from decoding.utils import validate_model_kwargs
from eval.metrics import eval_results
from utils.instructions import ICL_INSTRUCTIONS
from utils.make_input import get_make_input_func_icl
from utils.reduce_images import reduce_images

login(token=os.environ["HUGGINGFACE_TOKEN"])
transformers.generation.utils.GenerationMixin._validate_model_kwargs = validate_model_kwargs
transformers.generation.utils.GenerationMixin._sample = _sample


def chunk_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def maximize_diversity_chunking(features: torch.Tensor, n_chunks: int) -> torch.Tensor:
    device = features.device
    num_samples, _ = features.shape
    assert n_chunks <= num_samples
    # ----- Step 1: seed selection -----
    seed_indices = []

    # random selection for seeds
    first_seed_idx = torch.randint(0, num_samples, (1,), device=device).item()
    seed_indices.append(first_seed_idx)

    # select rest seeds
    chosen_seeds_features = features[seed_indices]

    for _ in range(n_chunks - 1):
        dist_to_seeds = torch.cdist(features, chosen_seeds_features, p=2)
        min_dist_to_seeds, _ = torch.min(dist_to_seeds, dim=1)
        next_seed_idx = torch.argmax(min_dist_to_seeds).item()
        seed_indices.append(next_seed_idx)
        chosen_seeds_features = features[seed_indices]

    # ----- Step 2: assign rest samples -----
    chunks = [[idx] for idx in seed_indices]

    # discard already assigned samples
    all_indices = set(range(num_samples))
    assigned_indices = set(seed_indices)
    unassigned_indices = list(all_indices - assigned_indices)

    # randomize orders
    shuffled_indices = torch.tensor(unassigned_indices, device=device)[
        torch.randperm(len(unassigned_indices), device=device)
    ]

    for point_idx in shuffled_indices:
        point_feature = features[point_idx : point_idx + 1]  # (1, feature_dim)

        max_avg_dist = -1.0
        best_chunk_idx = -1

        for j in range(n_chunks):
            chunk_indices = chunks[j]
            chunk_features = features[chunk_indices]

            distances = torch.cdist(point_feature, chunk_features, p=2)

            avg_dist = torch.mean(distances)

            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                best_chunk_idx = j

        chunks[best_chunk_idx].append(point_idx.item())

    assignments = torch.empty(num_samples, dtype=torch.long, device=device)
    for chunk_id, chunk_indices in enumerate(chunks):
        assignments[torch.tensor(chunk_indices, device=device)] = chunk_id

    return assignments


def chunk_context(
    context: torch.Tensor,
    chunk_num: int,
    features: torch.Tensor,
    target_similarity: torch.Tensor,
    strategy: str,
):
    match strategy:
        case "random":
            indices = torch.arange(0, len(context))
        case "similarity":
            _, indices = torch.sort(target_similarity, descending=True)
        case "kmeans":
            model = KMeans(n_clusters=chunk_num, verbose=False)
            labels = model.fit_predict(features.unsqueeze(0))
            _, indices = torch.sort(labels.squeeze(0))
        case "diversity":
            labels = maximize_diversity_chunking(features.float(), chunk_num)
            _, indices = torch.sort(labels.squeeze(0))
    new_context = [context[idx.item()] for idx in indices]
    return new_context, indices


def get_feature_and_similarity(encoder, processor, img_path: str, question: str, context: torch.Tensor, strategy: str):
    images = [Image.open(img_path).convert("RGB")]
    images = images + [Image.open(ctx["image"]).convert("RGB") for ctx in context]
    texts = [question]
    texts = texts + [" ".join([ctx["question"], ctx["answer"]]) for ctx in context]
    inputs = processor(text=texts, images=images, padding="max_length", truncation=True, return_tensors="pt").to(
        encoder.device
    )

    with torch.no_grad():
        outputs = encoder(**inputs)

    match strategy:
        case "visual":
            features = outputs.image_embeds
        case "textual":
            features = outputs.text_embeds
        case "multimodal":
            features = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=-1)
        case _:
            raise NotImplementedError()

    similarities = torch.matmul(features, features.T)[0]
    return features[1:], similarities[1:]


def get_automodel_class(model_path: str):
    model_name = model_path.split("/")[-1].lower()
    if model_name.startswith("phi"):
        return AutoModelForCausalLM
    else:
        return AutoModelForImageTextToText


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = model_path
    automodel_class = get_automodel_class(model_name)
    make_input = get_make_input_func_icl(model_name)
    model = automodel_class.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=args.trust_remote_code,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"

    encoder = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    enc_processor = AutoProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384",
        use_fast=True,
    )
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    responces = []
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.generation_config:
        generation_config = GenerationConfig.from_pretrained(model_name)
    else:
        generation_config = None

    counter = 0
    total_elapsed_time = 0.0
    total_prefill_time = 0.0
    total_token_per_sec = 0.0
    total_generated_tokens = 0.0
    total_embedding_time = 0.0
    total_chunking_time = 0.0

    for line in tqdm(questions, total=len(questions)):
        counter += 1
        idx = line["question_id"]
        category = line["category"]

        # Make input context
        instruction = ICL_INSTRUCTIONS[category]
        question = line["text"].replace("<image>", "")
        max_shot_key = list(line["context"].keys())[-1]
        num_shot = min(args.shot, len(line["context"][max_shot_key]))
        contexts = line["context"][max_shot_key][:num_shot]
        answer = line["answer"]
        image_path = os.path.join(args.image_folder, line["image"])
        for ctx in contexts:
            ctx["image"] = os.path.join(args.image_folder, ctx["image"][0])
            ctx["question"] = ctx["question"].replace("<image>", "")
        chunk_size = args.shot // args.chunk_num

        # Chunk input contexts according to strategies
        start = time.perf_counter()
        features, similarities = get_feature_and_similarity(
            encoder, enc_processor, image_path, question, contexts, args.feature_strategy
        )
        end = time.perf_counter()
        embedding_time = end - start
        start = time.perf_counter()
        if args.demo_pruning:
            reduced_shot = int(args.shot * args.reduction_rate)
            _, valid_indices = reduce_images(features[1:], features[0], "div_prune", reduced_shot)
            contexts = [contexts[index.item()] for index in valid_indices]
            features = torch.stack([features[index.item()] for index in valid_indices])
            similarities = torch.stack([similarities[index.item()] for index in valid_indices])
        contexts, chunk_indices = chunk_context(
            contexts, args.chunk_num, features, similarities, args.chunking_strategy
        )
        end = time.perf_counter()
        chunking_time = end - start
        chunked_similarities = chunk_list(similarities[chunk_indices], chunk_size)
        chunk_score = F.softmax(torch.stack(chunked_similarities).mean(dim=-1), dim=-1)

        model_kwargs = make_input(processor, instruction, contexts, image_path, question, chunk_size=chunk_size).to(
            model.device, dtype=torch.bfloat16
        )

        input_len = model_kwargs["input_ids"].shape[-1]
        with torch.inference_mode():
            start = time.perf_counter()
            output = model.generate(
                **model_kwargs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                cache_implementation=args.cache_implementation,
                generation_config=generation_config,
                return_dict_in_generate=True,
                context_scores=chunk_score,
                compile_strategy=args.compile_strategy,
            )
            end = time.perf_counter()

        outputs = processor.decode(output.sequences[0][input_len:], skip_special_tokens=True).strip()
        # [MODIFIED] measuring times
        elapsed_time = end - start
        token_per_sec = len(output.sequences[0][input_len:]) / elapsed_time
        total_elapsed_time += elapsed_time
        total_token_per_sec += token_per_sec
        total_prefill_time += output.prefill_time
        total_generated_tokens += len(output.sequences[0]) - input_len
        total_embedding_time += embedding_time
        total_chunking_time += chunking_time

        if args.verbose and (counter % args.print_freq == 0):
            print("-----")
            print(f"Question: {line['text']}")
            print(f"Answer: {outputs}")
            print(f"Avg. Decoding Time (sec): {total_elapsed_time / counter}")
            print(f"Avg. Generated Tokens: {total_generated_tokens / counter}")
            print(f"Avg. Token / Sec: {total_token_per_sec / counter}")
            print("-----")

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": question,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        responces.append(
            {"question_id": idx, "category": category, "question": question, "prediction": outputs, "answer": answer}
        )
        # ans_file.flush()
        del model_kwargs, output
    ans_file.close()

    print(f"Average Decoding Time (sec): {total_elapsed_time / counter}")
    print(f"Avg. Prefill Time (sec): {total_prefill_time / counter}")
    print(f"Avg. Embedding Time (sec): {total_embedding_time / counter}")
    print(f"Avg. Chunking Time (sec): {total_chunking_time / counter}")
    print(f"Avg. Generated Tokens: {total_generated_tokens / counter}")
    print(f"Average Token / Sec: {total_token_per_sec / counter}")

    # Summarize results
    print("### Task Results")
    scores = eval_results(args.evaluation, responces)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--assistant-model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./MIBench/")
    parser.add_argument("--question-file", type=str, default="./MIBench/multimodal_in_context_learning_val.jsonl")
    parser.add_argument("--answers-file", type=str, default="./test/answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cache_implementation", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--generation_config", action="store_true")
    parser.add_argument("--num_logits_to_keep", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=5)
    parser.add_argument("--shot", type=int, default=32)
    parser.add_argument("--chunk_num", type=int, default=2)
    parser.add_argument("--feature_strategy", type=str, default="multimodal")
    parser.add_argument("--chunking_strategy", type=str, default="kmeans")
    parser.add_argument("--compile_strategy", type=str, default="score_pow")
    parser.add_argument("--evaluation", type=str, default="mibench")
    parser.add_argument("--demo_pruning", action="store_true")
    parser.add_argument("--reduction_rate", type=float, default=0.5)
    args = parser.parse_args()
    # llava-onevision-qwen2-7b-si-hf
    eval_model(args)
