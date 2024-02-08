#!/bin/bash




CKPT="wisdomik/Quilt-Llava-v1.5-7b" # or "./checkpoints/...your model"

mkdir -p ./playground/data/eval/quiltvqa/answers
mkdir -p ./playground/data/eval/pmcvqa/answers
mkdir -p ./playground/data/eval/pvqa/answers

# QUILT-VQA
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/quiltvqa/quiltvqa_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/quiltvqa/images \
    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

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
    --model-path $CKPT \
    --question-file ./playground/data/eval/quiltvqa/quiltvqa_nored_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/quiltvqa/images \
    --answers-file ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt-nored-circle.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# PMCVQA
python -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
    --image-folder ./playground/data/eval/pmcvqa/pmcvqa_images \
    --answers-file ./playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# PVQA
python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pvqa/pvqa_test_wo_ans.jsonl \
    --image-folder ./playground/data/eval/pvqa/images/test \
    --answers-file ./playground/data/eval/pvqa/answers/$CKPT-w-yn.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# Evaluate

# QUILT-VQA
python llava/eval/quilt_eval.py \
    --quilt True \
    --gt ./playground/data/eval/quiltvqa/quiltvqa_test_w_ans.json \
    --pred ./playground/data/eval/quiltvqa/answers/$CKPT-w-yn-quilt.jsonl

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

# PMCVQA
python llava/eval/pmc_eval.py \
    --question-file ./playground/data/eval/pmcvqa/pmcvqa_test_wo_ans.json \
    --result-file ./playground/data/eval/pmcvqa/answers/$CKPT-w-yn-pmc.jsonl \
    --output-file ./playground/data/eval/pmcvqa/answers/$CKPT_output.jsonl \
    --output-result ./playground/data/eval/pmcvqa/answers/$CKPT_result.json

# PVQA
python llava/eval/quilt_eval.py \
    --gt ./playground/data/eval/pvqa/pvqa_test_w_ans.json \
    --pred ./playground/data/eval/pvqa/answers/$CKPT-w-yn.jsonl

