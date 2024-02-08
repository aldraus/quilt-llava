#!/bin/bash

CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/quiltvqa/answers



# QUILT-VQA RED
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/quiltvqa/quiltvqa_red_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/quiltvqa/red_circle \
    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt-red-circle.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# QUILT-VQA No RED
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$CKPT \
    --question-file ./playground/data/eval/quiltvqa/quiltvqa_nored_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/quiltvqa/images \
    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt-nored-circle.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# Evaluate


# QUILT-VQA  RED
python llava/eval/quilt_eval.py \
    --quilt True \
    --gt ./playground/data/eval/quiltvqa/quiltvqa_red_test_w_ans.json \
    --pred ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt-red-circle.jsonl

# QUILT-VQA No RED
python llava/eval/quilt_eval.py \
    --quilt True \
    --gt ./playground/data/eval/quiltvqa/quiltvqa_red_test_w_ans.json \
    --pred ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt-nored-circle.jsonl
