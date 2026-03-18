TASK=gqa
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=0

python eval/zeroshot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK