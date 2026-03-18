TASK=gqa
MODEL=OpenGVLab/InternVL3_5-8B-hf
IMGDIR=./data/gqa/images
QUESTION=./data/gqa/test.json
SHOT=32
CHUNK=2

python eval/ccc_icl.py --shot $SHOT --chunk_num $CHUNK --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --demo_pruning