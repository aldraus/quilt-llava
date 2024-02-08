#!/bin/bash


CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/pmcvqa/answers

# PMCVQA
python -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
    --image-folder ./playground/data/eval/pmcvqa/pmcvqa_images \
    --answers-file ./playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# Evaluate
# PMCVQA
python llava/eval/pmc_eval.py \
    --question-file ./playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
    --result-file ./playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl \
    --output-file ./playground/data/eval/pmcvqa/answers/$CKPT_output.jsonl \
    --output-result ./playground/data/eval/pmcvqa/answers/$CKPT_result.json
