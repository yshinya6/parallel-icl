TASK=gqa
MODEL=OpenGVLab/InternVL3_5-8B-hf
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=32

python eval/icl.py --shot $SHOT --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK