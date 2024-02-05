import argparse
import json
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D"])
    parser.add_argument('--anchor-file', type=str)
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    problems = json.load(open(args.question_file))
    predictions = [json.loads(line) for line in open(args.result_file)]
    if args.anchor_file:
        anchors = [json.loads(line) for line in open(args.anchor_file)]
        anchor_ids = [pred['question_id'] for pred in anchors]

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}
                        
    gt_ids = [item['id'] for item in problems]
    pred_ids = [pred['question_id'] for pred in predictions]
    # import pdb; pdb.set_trace()
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"
    if args.anchor_file:
        assert anchor_ids == pred_ids, "please make sure pred and anchors are exactly matched"


    for pred, gt, anchor in zip(predictions, problems, anchors if args.anchor_file else predictions):    
        pred_text = pred['text'][:len(anchor['text'])]

        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and (pred_text[1:2] == ":" or pred_text[1:2] == "."):
            answer = pred_text[0]
        else:
            pattern = re.compile(r'([A-Z]).')
            res = pattern.findall(pred_text)
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                pattern = re.compile(r'option ([A-Z]).')
                res = pattern.findall(pred_text)
                if len(res) == 1:
                    answer = res[0]  # 'A', 'B', ...
                else:
                    pattern = re.compile(r' ([A-Z]):')
                    res = pattern.findall(pred_text)
                    if len(res) == 1:
                        answer = res[0]  # 'A', 'B', ...
                    else:
                        pattern = re.compile(r"'([A-Z])'")
                        res = pattern.findall(pred_text)
                        if len(res) == 1:
                            answer = res[0]  # 'A', 'B', ...
                        else:
                            pattern = re.compile(r'is ([A-Z]).')
                            res = pattern.findall(pred_text)
                            if len(res) == 1:
                                answer = res[0]  # 'A', 'B', ...
                            else:
                                answer = "FAILED"    
        
        analysis = {
            'question_id': pred['question_id'],
            'parsed_ans': answer,
            'ground_truth': gt['conversations'][1]['value'],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][pred['question_id']] = answer
        sqa_results['outputs'][pred['question_id']] = pred_text

        if answer == gt['conversations'][1]['value']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
