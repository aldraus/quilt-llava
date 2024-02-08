import os
import json
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./pmcvqa_pathology_subset.csv", type=str, required=True)
    parser.add_argument("--dst", default="./playground/data/eval/pmcvqa/", type=str, required=True)
    return parser.parse_args()


def reformat_pmcvqa(test_path: str, output_dir:str, only_return=False):
    df = pd.read_csv(test_path)
    histo_pmc = df[df['in_quilt'] == False]
    histo_pmc.reset_index(drop=True, inplace=True)
    pmcvqa_json = []
    
    for ind, row in histo_pmc.iterrows():
        pmcvqa_json.append({
            "id": ind,
            "image": f"{row['Figure_path']}",
            "conversations": [
                  {
                    "from": "human",
                    "value": f"<image>\n{row['Question']}\n{row['Choice A']}\n{row['Choice B']}\n{row['Choice C']}\n{row['Choice D']}".replace('\n ', '\n')
                  },
                  {
                    "from": "gpt",
                    "value": f"{row['Answer']}"
                  }],
            "context": row['Caption'],
            "choices": [row['Choice A'], row['Choice B'], row['Choice C'], row['Choice D']]
        })
        
    if not only_return:
        output_filename = os.path.join(output_dir, "pmcvqa_test_wo_ans.jsonl")
        with open(output_filename, 'w') as json_file:
            for item in pmcvqa_json:
                json.dump(item, json_file)
                json_file.write('\n')

        output_filename = os.path.join(output_dir, "pmcvqa_test_wo_ans.json")
        with open(output_filename, 'w') as json_file:
            json.dump(pmcvqa_json, json_file)
    return pmcvqa_json

if __name__ == "__main__":
    args = get_args()
    reformat_pmcvqa(args.src, args.dst)       
