TASK=gqa
MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=0

python eval/zeroshot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK