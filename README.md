# [Parallel In-context Learning for Large Vision Language Models (CVPR2026 Findings)](https://arxiv.org/abs/2603.16092)
![Pipeline of Parallel-ICL. We cluster demonstrations in a multi-modal feature space and utilize each cluster as chunks (context chunking). Then, we process chunk-wise contexts with LVLMs and weight their outputs (logits) based on query-chunk similarity, composing an ensemble for the final prediction as PoE (context compilation). This can be computed by the weighted sum of outputs at the logit level.](./top.png)

## Overview
Large vision-language models (LVLMs) employ multi-modal in-context learning (MM-ICL) to adapt to new tasks by leveraging demonstration examples. While increasing the number of demonstrations boosts performance, they incur significant inference latency due to the quadratic computational cost of Transformer attention with respect to the context length. To address this trade-off, we propose **Parallel In-Context Learning (Parallel-ICL)**, a plug-and-play inference algorithm. Parallel-ICL partitions the long demonstration context into multiple shorter, manageable chunks. It processes these chunks in parallel and integrates their predictions at the logit level, using a weighted Product-of-Experts (PoE) ensemble to approximate the full-context output. Guided by ensemble learning theory, we introduce principled strategies for Parallel-ICL: (i) clustering-based context chunking to maximize inter-chunk diversity and (ii) similarity-based context compilation to weight predictions by query relevance. This repostiory provide a sample implementation of Parallel-ICL based on HuggingFace Transformers.

## Requirements
### Middleware Requirements
- CUDA >= 12.3
### Python Requirements
- See `apptainer/config.def`

## Preparations
### Evaluation Dataset: GQA
- 1. Download input images and metadata from [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- 2. Place images in `data/gqa/images`
- 3. Place metadata (`{train,test}_balanced_questions.json`) in `data/gqa/`
- 4. Run `python generate_gqa_quenstions.py --meta_test ./testdev_balanced_questions.json --meta_train ./train_balanced_questions.json --dst test.json` in `data/gqa/`

## Example on Qwen2.5VL-7B
```bash
bash experiments/01_gqa/qwen/qwen_gqa_ccc+divprune_32shot-2chunk.sh
```

## Citation

```bibtex
@inproceedings{Yamaguchi_CVPR26F_Parallel-ICL,
  title={Parallel In-context Learning for Large Vision Language Models},
  author={Yamaguchi, and Chijiwa, Daiki and Sakao, Tamao and Hasegawa, Taku},
  booktitle={Findings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```