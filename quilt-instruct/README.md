# Generating Quilt-Instruct from scratch

If you want to generate Quilt-Instruct yourself, please first download the following files

### Data Download
| Cursor and Diagnosis | Size |
| --- | ---: |
| [Cursors](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/cursor.parquet) | 	333 MiB |
| [Diagnosis and Supporting Facts](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/diagnosis_and_clues.parquet) | 1 MiB |

The cursor file includes raw cursor positions (in the "cursor" column) and clustered cursors (found in the "clustered_whole_sentences" column). It also contains additional metadata, such as whether the caption includes a diagnosis, and if there's a bounding box, among other information.

If you need to cluster the raw cursors, you can run quilt-llava/quilt-instruct/utils/cluster_singlerow.py with the default settings. This script will take the raw cursors, conduct spatio-temporal clustering on them, and then map these clusters to captions based on how close they are in time.

To generate Quilt-Instruct, you should use the "clustered_whole_sentences" column. To run it for independent prompts of conversations and detailed descriptions please run:

```Shell
TIKTOKEN_CACHE_DIR="" python pipeline_prompts.py --prompt_type conversation

TIKTOKEN_CACHE_DIR="" python pipeline_prompts.py --prompt_type detailed_description
```

To run it for complex reasoning and iterative abductive reasoning you need Diagnosis and Supporting Facts extracted from the whole videos, which can be found in diagnosis_and_clues.parquet file. Please first download it then run


```Shell
TIKTOKEN_CACHE_DIR="" python pipeline_prompts.py --prompt_type iterative_abduction

TIKTOKEN_CACHE_DIR="" python pipeline_prompts.py --prompt_type complex_reasoning
```
