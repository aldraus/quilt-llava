# Evaluation
We evaluate models on a diverse set of 4 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs, maintaining consistency with other LLaVA evaluation setups.


## Evaluate on Custom Datasets

You can evaluate Quilt-LLaVA on your custom datasets by converting your dataset to LLaVA's jsonl format, and evaluate using [`model_vqa.py`](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py).

Below we provide a general guideline for evaluating datasets with some common formats.

1. Short-answer (e.g. Quilt-VQA, Quilt-VQA-RED, PVQA).

```
<question>
Answer the question using a single word (e.g yes/no) or phrase.
```

2. Option-only for multiple-choice (e.g. PMC-VQA).

```
<question>
A. <option_1>
B. <option_2>
C. <option_3>
D. <option_4>
Answer with the option's letter from the given choices directly.
```

3. Natural QA (e.g. Quilt-VQA, Quilt-VQA-RED, PVQA).

No postprocessing is needed.

## Scripts

**Important notice**: Upon the request from the community, as ~15% images of the original CC-3M dataset are no longer accessible, we upload [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the CC-3M license. This may be taken down at any time when requested by the original CC-3M dataset owner or owners of the referenced images.


Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `./playground/data/eval`. This also provides a general structure for all datasets.

### QUILT-VQA

1. Download [`quilt_vqa image zip`](https://huggingface.co/datasets/wisdomik/Quilt_VQA/blob/main/quilt_vqa.zip), unzip and put it under `./playground/data/eval/quiltvqa/images`.
2. Sinlge-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/quilt_vqa.sh
```

### QUILT-VQA-RED

1. Download [`red_circle image zip`](https://huggingface.co/datasets/wisdomik/QuiltVQA_RED/blob/main/red_circle.zip), unzip and put it under `./playground/data/eval/quiltvqa/red_circle`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/quilt_red_vqa.sh
```

### PMC-VQA

1. Download [`pmcvqa_pathology_subset.csv`](https://huggingface.co/datasets/wisdomik/QuiltVQA_All/resolve/main/pmcvqa_pathology_subset.csv) put it in ./playground/data/eval/pmcvqa/ and then run 

```python
python llava/scripts/convert_pmcvqa_for_eval.py \ 
    --src ./playground/data/eval/pmcvqa/pmcvqa_pathology_subset.csv
    --dst ./playground/data/eval/pmcvqa/
```

next extract [`pmcvqa_images.zip`](https://huggingface.co/datasets/wisdomik/QuiltVQA_RED/resolve/main/pmcvqa_images.zip) to `pmcvqa_images`. Put them under `./playground/data/eval/pmcvqa/pmcvqa_images`.


2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pmcvqa.sh
```

### PVQA

https://drive.google.com/file/d/12WMDWqagP5SXleO_NGB83PlzIsf4zvBc/view?usp=drive_link

1. Download [`pvqa.zip`](https://drive.google.com/file/d/12WMDWqagP5SXleO_NGB83PlzIsf4zvBc/view?usp=drive_link), extract it and put the test set [`pvqa/qas/test/test_qa.pkl`](pvqa/qas/test/test_qa.pkl) into ./playground/data/eval/pvqa/ then run 

```python
python llava/scripts/convert_pvqa_for_eval.py \ 
    --src ./playground/data/eval/pvqa/test_qa.pkl
    --dst ./playground/data/eval/pvqa/
```


2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pvqa.sh
```
