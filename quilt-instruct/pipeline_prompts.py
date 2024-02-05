import pandas as pd
import os
import backoff
from utils import prompts
import openai
import logging
import argparse
import json
from litellm import completion

openai.api_type = "azure"
os.environ["TIKTOKEN_CACHE_DIR"] =""
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://gpt-4-proper.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2023-07-01-preview" 
openai.api_base = "https://gpt-4-proper.openai.azure.com/"
openai.api_key = os.environ.get("OPENAI_API_KEY")



@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def io_gpt(system_instr, payload, gpt_model="gpt-4", temp=0, max_tokens=512):
    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": payload},
    ]
    response = completions_with_backoff(deployment_id = os.getenv("OPENAI_API_BASE"), engine=gpt_model, temperature=temp, messages=messages, max_tokens=max_tokens)
    content = response['choices'][0]['message']['content']
    return content, response


    
def main():

    data = pd.read_parquet('roi_data_similarmerged_facecorrected_tracesfixed_textpadded_asrcorrected_final_clustersadded_medicalcontentdetected_rerun_uniqueidentifieradded_DETAILED_DESCRIPTION.parquet')  
 
    
    directory = os.path.join('/projects/brain1/saygin/youtube_project_laion/clip-retrieval/notebook/localized_narratives/quilt_instruction_tuning_dataset/', args.prompt_type) + '/'
    os.makedirs(directory, exist_ok=True)
    prompt_type = 'sys_' + args.prompt_type
    prompt = getattr(prompts, prompt_type, None)
    total_rows = len(data)
    for index, (idx, row) in enumerate(data.iterrows(), start=1):
        unique_row_id = row['unique_row_id']
        text = row['clustered_whole_sentences']

        filename = f'{unique_row_id}_{args.prompt_type}.txt'
        filepath = os.path.join(directory, filename)
        json_filename = f'{unique_row_id}_{args.prompt_type}.json'
        json_filepath = os.path.join(directory, json_filename)

        if os.path.exists(filepath) or os.path.exists(json_filepath):
            logger.info(f"Files for unique_row_id {unique_row_id} already exist. Skipping.")
            continue  

        try:
            content, response = io_gpt(prompt, text, gpt_model="gpt4", max_tokens=512)
            
            with open(filepath, 'w') as file:
                file.write(content)
            
            with open(json_filepath, 'w') as file:
                json.dump(response, file, indent=4)
            
            if index % 50 == 0:
                logger.info(f"Processed {index}/{total_rows} rows.")
                
        except Exception as e:
            logger.error(f"Failure for unique_row_id: {unique_row_id}. Error: {str(e)}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--prompt_type', type=str, required=True, help='The type of prompt to use.')
    args = parser.parse_args()
    log_file_name = f"gpt_failures_{args.prompt_type}.log"  # dynamic log file name based on prompt_type
    logging.basicConfig(filename=log_file_name, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    main()
    
    
    
    
    


