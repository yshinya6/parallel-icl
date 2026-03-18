import re
from collections import defaultdict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm

from utils.answer_processor import EvalAIAnswerProcessor


def eval_vqa(responces):
    scores = {"correct": 0, "all": 0}
    for res in responces:
        pred = res["prediction"].lower()
        if isinstance(res["answer"], list):
            answers = set([a.lower() for a in res["answer"]])
        else:
            answers = set([res["answer"].lower()])
        if pred in answers:
            scores["correct"] += 1
        scores["all"] += 1
    scores["accuracy"] = float(scores["correct"]) / float(scores["all"])
    print(f"Accuracy: {scores['accuracy']}")
    return scores


def eval_mibench(responces):
    scores = {res["category"]: {"correct": 0, "all": 0} for res in responces}
    for res in responces:
        pred = res["prediction"].lower()
        if isinstance(res["answer"], list):
            answers = set([a.lower() for a in res["answer"]])
        else:
            answers = set([res["answer"].lower()])
        if pred in answers:
            scores[res["category"]]["correct"] += 1
        scores[res["category"]]["all"] += 1
    for key in scores.keys():
        accuracy = float(scores[key]["correct"]) / float(scores[key]["all"])
        print(f"{key}: {accuracy}")
    return scores


def eval_caption(responces):
    gts = {}  # Ground Truths
    res = {}  # Results
    for item in responces:
        q_id = item["question_id"]
        res[q_id] = [item["prediction"]]
        gts[q_id] = item["answer"]
    scores = {}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    scores["cider"] = cider_score
    print(f"CIDEr Score: {cider_score:.4f}")
    bleu_scorer = Bleu()
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    scores["bleu"] = bleu_score
    print(f"BLEU Score: {bleu_score}")
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    print(f"Rouge Score: {rouge_score:.4f}")
    scores["rouge"] = rouge_score
    return scores


def eval_results(task: str, responces):
    if task == "mibench":
        return eval_mibench(responces)
    elif task == "coco_caption":
        return eval_caption(responces)
    elif task in ["gqa", "classification", "textvqa"]:
        return eval_vqa(responces)
    else:
        raise NotImplementedError()
