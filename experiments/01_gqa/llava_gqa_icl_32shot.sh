TASK=gqa
MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=32

python eval/icl.py --shot $SHOT --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK