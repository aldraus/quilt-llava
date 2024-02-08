## Install

If you are using Windows, do *NOT* proceed, see instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/aldraus/quilt-llava.git
cd quilt-llava
```

2. Install Package
```Shell
conda create -n qllava python=3.10 -y
conda activate qllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train


Quilt-LLaVA training consists of two stages: (1) feature alignment stage: use our 723K filtered image-text pairs from [QUILT-1M](https://quilt1m.github.io/) to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 107K GPT-generated multimodal instruction-following data from [QUILT-Instruct](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K) to teach the model to follow multimodal instructions.

Quilt-LLaVA is trained on 4 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.


### Hyperparameters
We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quilt-LLaVA-v1.5-7B | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quilt-LLaVA-v1.5-7B | 128 | 2e-5 | 1 | 2048 | 0 |

### Download Vicuna checkpoints (automatically)

Our base model Vicuna v1.5, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. No action is needed.

### Pretrain (feature alignment)

Please download the 723K subset/filtered image-text pairs from [QUILT-1M](https://quilt1m.github.io/) dataset with reformatted to QA styling we use in the paper [here](https://huggingface.co/datasets/wisdomik/Quilt-LLaVA-Pretrain).

Pretrain takes around 10 hours for LLaVA-v1.5-7B on 4x A100 (80G).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/aldraus/quilt-llava/blob/main/scripts/v1_5/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower wisdomik/QuiltNet-B-32`: CLIP ViT-B/32 224px.

### Visual Instruction Tuning

1. Prepare data

Please download the annotation of our instruction tuning data [quilt_instruct_107k.json](https://huggingface.co/datasets/wisdomik/QUILT-LLaVA-Instruct-107K/blob/main/quilt_instruct_107k.json), and download the images from Quilt-1M dataset:
- (Rescaled) On [Zenodo](https://zenodo.org/record/8239942) you can access the dataset with all images resized to 512x512 px (36 Gb)
- (Full) To access the dataset with full-sized images via Google Drive, please request time-limited access through this form [Google](https://forms.gle/TKohQ7zLwYfFn8qRA) (110 Gb)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── Quilt-LLaVA-Pretrain
│   └── quilt_1m/
            └── xxxxxxx.jpg
                ...
            └── yyyyyyy.jpg
    ├── quilt_pretrain.json
```

2. Start training!

You may download our pretrained projectors in [Quilt-Llava-v1.5-7b](https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b/blob/main/mm_projector.bin). It is not recommended to use legacy projectors, as they may be trained with a different version of the codebase, and if any option is off, the model will not function/train as we expected.

Visual instruction tuning takes around 15 hours for LLaVA-v1.5-7B on 4x A100 (80G).

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/aldraus/quilt-llava/blob/main/scripts/v1_5/finetune.sh).

If you are do not have enough GPU memory:

- Use LoRA: [`finetune_lora.sh`](https://github.com/aldraus/quilt-llava/blob/main/scripts/v1_5/finetune_lora.sh). Make sure `per_device_train_batch_size*gradient_accumulation_steps` is the same as the provided script for best reproducibility.
- Replace `zero3.json` with `zero3_offload.json` which offloads some parameters to CPU RAM. This slows down the training speed.

If you are interested in finetuning LLaVA model to your own task/data, please check out [`Finetune_Custom_Data.md`](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)。

New options to note:

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length False`: this should only be changed to True when your instruction tuning dataset contains both language data and multimodal (e.g. Quilt-LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

## Evaluation
We evaluate models on a diverse set of 4 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.


```Shell
python model_vqa.py \
    --model-path wisdomik/Quilt-Llava-v1.5-7b \
    --question-file ./playground/data/quilt_gpt/quilt_gpt_questions.jsonl \
    --image-folder ./playground/data/eval/quiltvqa/images \
    --answers-file /path/to/answer-file-our.jsonl
```

2. Evaluate the generated responses.  In our case, [`answer-file-ref.jsonl`](./playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl) is the response generated by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" 

python llava/eval/quilt_gpt_eval.py \
    --question ./playground/data/quilt_gpt/quilt_gpt_questions.jsonl \
    --context ./playground/data/quilt_gpt/quilt_gpt_captions.jsonl \
    --answer-list \
    /path/to/answer-file-ref.jsonl \
    /path/to/answer-file-our.jsonl \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python llava/eval/quilt_gpt_summarize.py \
    --dir /path/to/review/
```
