import argparse
import os
import json
import pprint
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate
from quilt_utils import *

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--quilt', type=bool, default=False, help='whether to evaluate on quilt outputs')
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--pred_file_parent_path', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--anchor', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to anchor prediction file, unused except for eval of lengthy preds', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred, quilt=False, anchor=None):    
    closed_scores2 = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)

    for gt_item, pred_item, anchor_item in zip(gt, pred, anchor if anchor else pred):
        gt_value = gt_item['answer'].lower()
        pred_value = pred_item['text'].lower()
        anchor_value = anchor_item['text'].lower()


        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)
        anchor_value = normalize_word(anchor_value)

        pred_value = pred_value[:len(anchor_value)]

        if gt_item['answer_type'] == 'OPEN' or gt_item['answer_type'] == 'other':
            # for open-ended question
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(pred_item['question_id'])

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(pred_item['question_id'])

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)

        elif gt_item['answer_type'] == 'CLOSED':
            # for close-ended question (Yes/No)
            closed_scores2['q_id'].append(pred_item['question_id'])

            if quilt:
                gt_value = gt_item['yes_no_answer'].lower()

            assert gt_value in ['yes', 'no'], f"assert gt_value in : {pred_item['question_id'], gt_value}"
            answer = gt_value
            # Only keep the first sentence
            #if pred_value.find('.') != -1:
            #    pred_value = pred_value.split('.')[0]

            pred_value = pred_value.replace(',', '')
            words = pred_value.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                pred_answer = 'no'
            else:
                pred_answer = 'yes'
            
            if pred_answer == answer:
                closed_scores2['hit'].append(1)
            else:
                closed_scores2['hit'].append(0)
                
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])
    closed_score2 = sum(closed_scores2['hit']) / len(closed_scores2['hit']) if len(closed_scores2['hit']) != 0 else 0.0

    return tabulate(
        [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['yes/no accuracy', closed_score2*100]
        ], 
        headers=['Metric', 'Performance']
    )


if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    if args.anchor:
        anchor = load_jsonl(args.anchor)
        anchor_ids = [item['question_id'] for item in anchor]

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(gt, pred, quilt=args.quilt, anchor=anchor if args.anchor else None)
    pprint.pprint(results)
