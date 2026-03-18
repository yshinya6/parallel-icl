import argparse
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
import transformers
from huggingface_hub import login
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from decoding.sample import _sample
from eval.metrics import eval_results
from utils.instructions import ICL_INSTRUCTIONS
from utils.make_input import get_make_input_func, get_make_input_func_icl

transformers.generation.utils.GenerationMixin._sample = _sample

login(token=os.environ["HUGGINGFACE_TOKEN"])


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
    make_input = get_make_input_func(model_name)
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
    total_token_per_sec = 0.0
    total_generated_tokens = 0.0
    total_decoding_time = 0.0
    total_decoding_time_per_iter = 0.0
    total_prefill_time = 0.0
    total_input_tokens = 0.0

    for line in tqdm(questions, total=len(questions)):
        counter += 1
        idx = line["question_id"]
        category = line["category"]

        question = line["text"].replace("<image>", "")
        answer = line["answer"]
        image_path = os.path.join(args.image_folder, line["image"])
        model_kwargs = make_input(processor, image_path, question).to(model.device, dtype=torch.bfloat16)

        input_len = model_kwargs["input_ids"].shape[-1]
        total_input_tokens += input_len
        with torch.inference_mode():
            start = time.perf_counter()
            output_dict = model.generate(
                **model_kwargs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                cache_implementation=args.cache_implementation,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            end = time.perf_counter()
        outputs = processor.decode(output_dict.sequences[0][input_len:], skip_special_tokens=True).strip()
        # [MODIFIED] measuring times
        elapsed_time = end - start
        token_per_sec = len(output_dict.sequences[0][input_len:]) / elapsed_time
        total_elapsed_time += elapsed_time
        total_token_per_sec += token_per_sec
        total_generated_tokens += len(output_dict.sequences[0]) - input_len
        total_decoding_time += output_dict.decoding_time
        total_decoding_time_per_iter += output_dict.decoding_time_per_iter
        total_prefill_time += output_dict.prefill_time

        if args.verbose and (counter % args.print_freq == 0):
            print("-----")
            print(f"Question: {question}")
            print(f"Answer: {outputs}")
            print(f"Avg. Decoding Time (sec): {total_elapsed_time / counter}")
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
        del model_kwargs, output_dict
    ans_file.close()

    print(f"Average Generate Time (sec): {total_elapsed_time / counter}")
    print(f"Average Prefill Time (sec): {total_prefill_time / counter}")
    print(f"Average Decoding Time / Iter (sec): {total_decoding_time_per_iter / counter}")
    print(f"Average Token / Token: {total_token_per_sec / counter}")
    print(f"Average Input Tokens: {total_input_tokens / counter}")
    print(f"Average Generated Tokens: {total_generated_tokens / counter}")

    # Summarize results
    print("### Task Results")
    scores = eval_results(args.evaluation, responces)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
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
    parser.add_argument("--evaluation", type=str, default="mibench")

    args = parser.parse_args()

    eval_model(args)
