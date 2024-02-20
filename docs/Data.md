## Data


| Instruction-Tuning data | Size |
| --- | ---: |
| [quilt_instruct_107K](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/quilt_instruct_107k.json) | 189 MiB |
| [quilt_instruct_ablation_40k](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/quilt_instruct_ablation_40k.json) | 37.4 MB |
| [quilt_instruct_complex_abductive](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/quilt_instruct_complex_abductive.json) | 43.2 MB |
| [quilt_instruct_conv_desc](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/quilt_instruct_conv_desc.json) | 32 MB |



| Evaluation files | Size |
| --- | ---: |
| [Quilt-VQA](https://huggingface.co/datasets/wisdomik/Quilt_VQA) | 	305 MiB |
| [Quilt-VQA Red Circle](https://huggingface.co/datasets/wisdomik/QuiltVQA_RED) | 95.8 MiB |

| Raw Mouse Cursor Data | Filename | Size |
| --- | --- |  ---: |
| [Cursors](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/cursor.parquet) |  cursor.parquet | 333 MiB |
| [Diagnosis and Supporting Facts](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/diagnosis_and_clues.parquet) | 1 MiB |

| Image URLS | Filename | Size |
| --- | --- | ---: |
| [Images (please request time-limited access through this form)](https://forms.gle/TKohQ7zLwYfFn8qRA) | quilt_instruct.zip | 25 GiB |

### Pretraining Dataset
The pretraining dataset used in this release is a subset of [QUILT-1M](https://quilt1m.github.io/) dataset, filtered to remove PubMed and Twitter subsets.  Please see [here](hhttps://huggingface.co/datasets/wisdomik/Quilt-LLaVA-Pretrain) for a detailed description of the dataset structure and how to download the images.


| Data | File | Size |
| --- |  --- | ---: |
| [QUILT-1M](https://quilt1m.github.io/) 723K subset | [quilt_pretrain.json](https://huggingface.co/datasets/wisdomik/Quilt-LLaVA-Pretrain/blob/main/quilt_pretrain.json) | 262 MB |

It must not be used for any other purposes. The use of these images must comply with the QUILT-1M license.

### GPT-4 Prompts
We provide our prompts and few-shot samples for GPT-4 queries, to better facilitate research in this domain.  Please check out the [`evaluation code`](https://github.com/aldraus/quilt-llava/blob/main/llava/eval/quilt_gpt_eval.py) with more details in the paper.
