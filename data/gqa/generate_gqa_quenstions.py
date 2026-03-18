import argparse
import json
import os
import pdb
import random

# python generate_gqa_quenstions.py --meta_test ./testdev_balanced_questions.json --meta_train ./train_balanced_questions.json --dst test.json 
parser = argparse.ArgumentParser()
parser.add_argument("--meta_test", type=str)
parser.add_argument("--meta_train", type=str)
parser.add_argument("--dst", type=str)

args = parser.parse_args()
DEMO_SIZE = 128

all_questions = []

with open(args.meta_test, "r") as f:
    questions = json.load(f)

with open(args.meta_train, "r") as f:
    train_questions = list(json.load(f).values())

for qid, data in questions.items():
    question_id = qid
    imageId = f"{data['imageId']}.jpg"
    text = data["question"]
    answer = data["answer"]
    category = "gqa"

    # sampling demonstrations from training set
    demo_lists = random.sample(train_questions, DEMO_SIZE)
    demos = []
    for demo_item in demo_lists:
        demos.append(
            {
                "image": [f"{demo_item['imageId']}.jpg"],
                "question": demo_item["question"],
                "answer": data["answer"],
            }
        )
    context = {f"{DEMO_SIZE}-shot": demos}

    all_questions.append(
        {
            "question_id": question_id,
            "image": imageId,
            "text": text,
            "answer": answer,
            "category": category,
            "context": context,
        }
    )

with open(args.dst, "w", encoding="utf-8") as f:
    for l in all_questions:
        json.dump(l, f, ensure_ascii=False)
        f.write("\n")
