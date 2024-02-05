import argparse
import json
import os

import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, model='gpt4', max_tokens=1024):
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=model,#'gpt-4-0314',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


pt = """We would like to request your feedback on the performance of two AI assistants in response to the user question
displayed above. The user asks the question on observing an image. For your reference, the visual content in 
the image is represented with caption, describing the same image, which is embedded with bounding box coordinates
of each object in the scene, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1.
These values correspond to the top left x, top left y, bottom right x, and bottom right y.
\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses.
Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall
performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1
and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a
comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the
responses were presented does not affect your judgment."""

rule = {'role': 'Assistant',
        'prompt':pt,
       }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-o', '--output')
    parser.add_argument('-m', '--eval-engine', default='gpt4')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    openai.api_type = "azure"
    openai.api_base = "https://gpt-4-proper.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    cap = open(os.path.expanduser(args.context))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []
    review_file = open(f'{args.output}', 'a')

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js, cap_js in zip(f_q, f_ans1, f_ans2, cap):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)
        cap_str = json.loads(cap_js)['text']

        category = json.loads(ques_js)['category']

        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, model=args.eval_engine, max_tokens=args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
