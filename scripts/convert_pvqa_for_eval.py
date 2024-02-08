import os
import json
import argparse
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./pvqa/pvqa/qas/test/test_qa.pkl", type=str, required=True)
    parser.add_argument("--dst", default="./playground/data/eval/pvqa/", type=str, required=True)
    return parser.parse_args()


def reformat_pvqa(test_path: str, output_dir:str):
    pvqa_json = []
    test_qa_w_id = []
    
    with open(test_path, 'rb') as file:
        test_qa = pickle.load(file)

    for test_id, test_qa_entry in enumerate(test_qa):

        if test_qa_entry['answer'].lower() in ['yes', 'no']:
            test_qa_entry.update({'answer_type':'CLOSED'})
            text = test_qa_entry['question'] + " Please choose from the following two options: [yes, no]"    
        else:
            test_qa_entry.update({'answer_type':'OPEN'})
            text = test_qa_entry['question']

        pvqa_json.append({
            "image": f"{test_qa_entry['image']}.jpg",
            "text": text,
            "category": "conv",
            "question_id": test_id, #int(test_qa_entry['image'][test_qa_entry['image'].find('_') + 1 :]), 
            "answer_type": test_qa_entry['answer_type'],
        })
        test_qa_entry.update({'id':test_id})
        test_qa_w_id.append(test_qa_entry)
    
    output_filename = os.path.join(output_dir, "pvqa_test_wo_ans.jsonl")
    with open(output_filename, 'w') as json_file:
        for item in pvqa_json:
            json.dump(item, json_file)
            json_file.write('\n')
    
    output_filename = os.path.join(output_dir, "pvqa_test_w_ans.json")
    with open(output_filename, 'w') as json_file:
        json.dump(test_qa_w_id, json_file)

if __name__ == "__main__":
    args = get_args()
    reformat_pvqa(args.src, args.dst)                  
