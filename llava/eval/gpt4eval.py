import os
import json
import math
import base64
import argparse
import shortuuid
from tqdm import tqdm


from llava.constants import DEFAULT_IMAGE_TOKEN 
from llava.conversation import conv_templates
from openai import OpenAI
import openai, backoff


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


client = OpenAI(
      organization='org-adqxNtt2dnVrZPf72p4l34WS',
)


def eval_model(args):
    if args.mvqa:
        questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        if args.mvqa:
            idx = line["id"]
            question = line['conversations'][0]
            qs = question['value'].replace('<image>', '').strip()
            cur_prompt = qs
            cur_prompt = '<image>' + '\n' + cur_prompt
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        else:
            idx = line["question_id"]
            qs = line["text"]
            cur_prompt = qs
        
        image_file = line["image"]

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_path = os.path.join(args.image_folder, image_file)
        # Getting the base64 string
        base64_image = encode_image(image_path)

        response = completions_with_backoff(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                                 "text": f"{prompt}"},
                        {"type": "image_url",
                                "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=args.temperature,
        )

        outputs = response.choices[0]        
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": "gpt4V",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mvqa", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
