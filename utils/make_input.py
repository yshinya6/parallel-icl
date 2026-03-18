import os
import pdb

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.image_utils import load_image


def get_make_input_func(model_path):
    model_name = model_path.split("/")[-1]
    make_input_func = None
    if model_name.startswith("gemma-3"):
        make_input_func = make_input_for_gemma3
    elif model_name.startswith("Llama-3.2"):
        make_input_func = make_input_for_llama3
    elif model_name.startswith("Qwen2"):
        make_input_func = make_input_for_qwen2h
    elif model_name.startswith("llama3-llava-next"):
        make_input_func = make_input_for_llama3
    elif model_name.startswith("llava"):
        make_input_func = make_input_for_llama3
    elif model_name.startswith("Phi-4"):
        make_input_func = make_input_for_phi4
    elif model_name.startswith("Idefics"):
        make_input_func = make_input_for_idefics3
    elif model_name.startswith("InternVL"):
        make_input_func = make_input_for_internvl
    else:
        raise NotImplementedError()
    return make_input_func


def get_make_input_func_icl(model_path):
    model_name = model_path.split("/")[-1]
    make_input_func = None
    if model_name.startswith("Qwen2"):
        make_input_func = make_input_for_qwen2h_icl
    elif model_name.startswith("llava"):
        make_input_func = make_input_for_llama3_icl
    elif "idefics" in model_name.lower():
        make_input_func = make_input_for_idefics3_icl
    elif "internvl" in model_name.lower():
        make_input_func = make_input_for_internvl_icl
    else:
        raise NotImplementedError()
    return make_input_func


def chunk_list(input_list, chunk_size):
    return [input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def make_input_for_gemma3(processor, image_path, text, text_only=False, tokenize=True):
    if text_only:
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]
    else:
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            },
        ]
    input_dict = processor.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=tokenize,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=True,  # for captureing local features
        pan_and_scan_min_ratio_to_activate=0.5,
        pan_and_scan_max_num_crops=32,
        pan_and_scan_min_crop_size=4,
    )
    return input_dict


def make_input_for_idefics3(processor, image_path, text, text_only=False, tokenize=True):
    if text_only:
        image = None
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]
    else:
        image = [load_image(image_path)]
        message = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
    prompt = processor.apply_chat_template(message, add_generation_prompt=True)
    input_dict = processor(
        text=prompt,
        images=image,
        tokenize=tokenize,
        return_dict=True,
        return_tensors="pt",
    )
    return input_dict


def make_input_for_idefics3_icl(
    processor,
    instruction,
    contexts,
    image_path,
    query,
    chunk_size=None,
    tokenize=True,
):
    assert len(contexts) > 0
    if chunk_size is None:
        chunk_size = len(contexts)
    context_chunks = chunk_list(contexts, chunk_size)
    message_chunks = []
    images = []
    for ctx_chunk in context_chunks:
        per_chunk_images = []
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        for context in ctx_chunk:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": context["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": context["answer"]}]},
            ]
            per_chunk_images.append(load_image(context["image"]))
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        per_chunk_images.append(load_image(image_path))
        message_chunks.append(messages)
        images.append(per_chunk_images)
    prompt = processor.apply_chat_template(message_chunks, add_generation_prompt=True)
    input_dict = processor(
        text=prompt,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return input_dict


def make_input_for_llama3(processor, image_path, text, text_only=False, tokenize=True):
    if text_only:
        image = None
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]
    else:
        image = Image.open(image_path).convert("RGB")
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
    input_text = processor.apply_chat_template(message, add_generation_prompt=True)
    input_dict = processor(image, input_text, tokenize=tokenize, add_special_token=False, return_tensors="pt")
    input_dict["pad_token_id"] = processor.tokenizer.eos_token_id
    return input_dict


def make_input_for_llama3_icl(
    processor,
    instruction,
    contexts,
    image_path,
    query,
    chunk_size=None,
    tokenize=True,
    max_pixels=1280,
):
    if chunk_size is None:
        chunk_size = len(contexts)
    context_chunks = chunk_list(contexts, chunk_size)
    message_chunks = []
    for ctx_chunk in context_chunks:
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        for context in ctx_chunk:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": context["image"]},
                        {"type": "text", "text": context["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": context["answer"]}]},
            ]
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        message_chunks.append(messages)
    input_dict = processor.apply_chat_template(
        message_chunks,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    )
    input_dict["pad_token_id"] = processor.tokenizer.eos_token_id
    return input_dict


def make_input_for_qwen2h(processor, image_path, text, text_only=False, tokenize=True):
    if text_only:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        image_input, video_input = None, None
    else:
        contents = [{"type": "image", "image": image_path, "min_pixels": 512 * 28 * 28, "max_pixels": 1280 * 28 * 28}]
        contents += [{"type": "text", "text": text}]
        message = [
            {
                "role": "user",
                "content": contents,
            }
        ]
        image_input, video_input = process_vision_info(message)
    input_text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input_dict = processor(text=[input_text], images=image_input, videos=video_input, padding=True, return_tensors="pt")
    return input_dict


def make_input_for_qwen2h_icl(
    processor,
    instruction,
    contexts,
    image_path,
    query,
    chunk_size=None,
    tokenize=True,
    max_pixels=1280,
):
    if chunk_size is None:
        chunk_size = len(contexts)
    context_chunks = chunk_list(contexts, chunk_size)
    message_chunks = []
    for ctx_chunk in context_chunks:
        messages = [{"role": "system", "content": instruction}]
        for context in ctx_chunk:
            img_path = context["image"]
            messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                            "min_pixels": 512 * 28 * 28,
                            "max_pixels": max_pixels * 28 * 28,
                        },
                        {"type": "text", "text": context["question"]},
                    ],
                },
                {"role": "assistant", "content": context["answer"]},
            ]
        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": 512 * 28 * 28,
                        "max_pixels": max_pixels * 28 * 28,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        message_chunks.append(messages)
    image_input, video_input = process_vision_info(message_chunks)
    input_texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in message_chunks
    ]
    input_dict = processor(text=input_texts, images=image_input, videos=video_input, padding=True, return_tensors="pt")
    return input_dict


def make_input_for_phi4(processor, image_path, text, text_only=False, tokenize=True):
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    image_prompt = "<|image_1|>"  # assume single image
    if text_only:
        prompt = f"{user_prompt}{text}{prompt_suffix}{assistant_prompt}"
        image = None
    else:
        prompt = f"{user_prompt}{image_prompt}{text}{prompt_suffix}{assistant_prompt}"
        image = Image.open(image_path).convert("RGB")
    input_dict = processor(text=prompt, images=image, return_tensors="pt")
    return input_dict


def make_input_for_internvl_icl(
    processor,
    instruction,
    contexts,
    image_path,
    query,
    chunk_size=None,
    tokenize=True,
    max_pixels=1280,
):
    if chunk_size is None:
        chunk_size = len(contexts)
    context_chunks = chunk_list(contexts, chunk_size)
    message_chunks = []
    for ctx_chunk in context_chunks:
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        for context in ctx_chunk:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": context["image"]},
                        {"type": "text", "text": context["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": context["answer"]}]},
            ]
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        message_chunks.append(messages)
    input_dict = processor.apply_chat_template(
        message_chunks,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    )
    input_dict["pad_token_id"] = processor.tokenizer.eos_token_id
    return input_dict


def make_input_for_internvl(processor, image_path, text, text_only=False, tokenize=True):
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": query},
            ],
        },
    ]
    input_dict = processor.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    )
    input_dict["pad_token_id"] = processor.tokenizer.eos_token_id
    return input_dict
