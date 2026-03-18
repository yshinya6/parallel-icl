TASK=gqa
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=32

python eval/icl.py --shot $SHOT --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK