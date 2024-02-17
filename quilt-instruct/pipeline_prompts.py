import pandas as pd
import os
import backoff
from utils import prompts
import openai
import logging
import argparse
import json
import random
from litellm import completion


openai.api_type = "azure"
os.environ["TIKTOKEN_CACHE_DIR"] =""
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://gpt-4-proper.openai.azure.com/openai/deployments/gpt4/chat/completions?api-version=2023-07-01-preview" 
openai.api_base = "https://gpt-4-proper.openai.azure.com/"
openai.api_key = "Enter your key"



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

def setup_logger(name, log_file, level=logging.INFO):
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)  
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def setup_directory_and_logger(output_file_path, prompt_type):
    directory = os.path.join(output_file_path, prompt_type) + '/'
    os.makedirs(directory, exist_ok=True)
    log_file_name = f"gpt_failures_{prompt_type}.log"
    logger = setup_logger('application_logger', log_file_name)
    return directory, logger

def read_data(prompt_type):
    data = pd.read_parquet('cursor_small.parquet')
    if prompt_type in ['complex_reasoning', 'iterative_abduction']:
        data = data[data['prompt_type'] == 'reasoning'].reset_index()
    else:
        data = data[data['prompt_type'] != 'reasoning'].reset_index()
    return data

def write_content_to_file(filepath, content, json_response=None):
    try:
        with open(filepath, 'w') as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")
        
    if json_response is not None:
        json_filepath = filepath.replace('.txt', '.json')
        try:
            with open(json_filepath, 'w') as file:
                json.dump(json_response, file, indent=4)
        except Exception as e:
            print(f"Error writing to JSON file {json_filepath}: {e}")


def process_row_general(row, directory, logger, prompt,prompt_type):
    unique_row_id = row['unique_row_id']
    text = row['clustered_whole_sentences']
    filename = f'{unique_row_id}_{prompt_type}.txt'
    filepath = os.path.join(directory, filename)
    content, response = io_gpt(prompt, text, gpt_model="gpt4", max_tokens=512)
    write_content_to_file(filepath, content)
    logger.info(f"Processed {prompt_type} for {unique_row_id}")
       

def process_row_complex_reasoning(row, directory, logger, diags, prompt):

    unique_row_id = row['unique_row_id']
    text = row['extended_text_sentences_asrcorrected']['utterance']
    vid_id = row['video_id']
    diagnosis_and_clues = diags[diags['video_id'] == vid_id]['content'].values[0]    
    diagnosis_and_clues = diagnosis_and_clues.replace("clues", "Clues from Whole Slide")
    text = ', "Single Patch": ' +'[' + text +']'
    text = diagnosis_and_clues[:-1] + text + '}'
    content, response = io_gpt(prompt, text, gpt_model="gpt4", max_tokens=512)
    filename = f'{unique_row_id}_complex_reasoning.txt'
    filepath = os.path.join(directory, filename)
    
    write_content_to_file(filepath, content)
    logger.info(f"Processed complex reasoning for {unique_row_id}")


def process_row_iterative_abduction(row, directory, logger, diags, prompt_type):
    unique_row_id = row['unique_row_id']
    text = row['extended_text_sentences_asrcorrected']['utterance']
    vid_id = row['video_id']
    diagnosis_and_clues = diags[diags['video_id'] == vid_id]['content'].values[0]
    diagnosis_and_clues = diagnosis_and_clues[:-1] + "Student's Image:" +'[' + text +']' + '}'
    conversation_history = []
    
    prompt_student = getattr(prompts, 'prompt_student', None)
    prompt_assistant = getattr(prompts, 'prompt_assistant', None)
    prompt_assistant += diagnosis_and_clues
    filename = f'{unique_row_id}_{prompt_type}.txt'
    filepath = os.path.join(directory, filename)
    number_of_conversations = [4, 6] #
    weights = [0.85, 0.15] 
    dialogue_length = random.choices(number_of_conversations, weights)[0]
    for i in range(dialogue_length):
        if i == 0:
            payload = text
        else:
            payload = "".join([f"{message}\n" for message in conversation_history])  # Extracting GPT's last statement

        try:
            content, response = io_gpt(
                    prompt_student if i % 2 == 0 else prompt_assistant,  
                    payload,  
                    gpt_model="gpt4",
                    temp=0,  
                    max_tokens=512
                )
            conversation_history.append(content)  
                
            if "CORRECT!!!" in content or "End of Guidance" in content:
                break
                
        except Exception as e:
                logger.error(f"Failure for unique_row_id: {unique_row_id}. Error: {str(e)}")
                
    conversation_str = '\n'.join(conversation_history)
    write_content_to_file(filepath, conversation_str)
    logger.info(f"Processed {prompt_type} for {unique_row_id}")
    
def pipeline_processing(output_file_path, prompt_type):
    directory, logger = setup_directory_and_logger(output_file_path, prompt_type)
    data = read_data(prompt_type)
    prompt = getattr(prompts, 'sys_' + prompt_type, None)
    diags = None
    if prompt_type in ['complex_reasoning', 'iterative_abduction']:
        diags = pd.read_parquet('diagnosis_and_clues.parquet')
    
    if prompt_type in ['conversation', 'detailed_description']:
        process_row_function = lambda row: process_row_general(row, directory, logger, prompt, prompt_type)
    elif prompt_type == 'complex_reasoning':
        process_row_function = lambda row: process_row_complex_reasoning(row, directory, logger, diags, prompt)
    elif prompt_type == 'iterative_abduction':
        process_row_function = lambda row: process_row_iterative_abduction(row, directory, logger, diags, prompt_type)
    else:
        logger.error(f"Unsupported prompt_type: {prompt_type}")
        return  

    for index, row in data.iterrows():
        process_row_function(row)
        
        if index % 50 == 0:
            logger.info(f"Processed {index}/{len(data)} rows.")


            
def main():
    parser = argparse.ArgumentParser(description='Unified script for processing prompts based on the type.')
    parser.add_argument('--prompt_type', type=str, required=True, help='The type of prompt to use.')
    parser.add_argument('--output_file_path', type=str, default=os.getcwd(), help='The base path for output files. Defaults to the current directory.')
    args = parser.parse_args()

    pipeline_processing(args.output_file_path, args.prompt_type)

if __name__ == "__main__":
    main()
    


