#!/bin/bash


CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/pvqa/answers


# PVQA
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pvqa/pvqa_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/pvqa/images/test \
    --answers-file ./playground/data/eval/pvqa/answers/$CKPT-w-yn.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# Evaluate


# PVQA
python llava/eval/quilt_eval.py \
    --gt ./playground/data/eval/pvqa/pvqa_test_w_ans.json \
    --pred ./playground/data/eval/pvqa/answers/$CKPT-w-yn.jsonl

