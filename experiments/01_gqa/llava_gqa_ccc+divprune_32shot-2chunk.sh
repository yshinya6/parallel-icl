TASK=gqa
MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=32
CHUNK=2

python eval/ccc_icl.py --shot $SHOT --chunk_num $CHUNK --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --demo_pruning