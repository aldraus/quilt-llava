import pandas as pd
import ast
import backoff
import numpy as np
import os
import openai
from tqdm import tqdm
import logging
from helper import get_full_questions, generate_user_msg


os.environ["OPENAI_API_KEY"] = "" 
openai.api_type = "azure"
openai.api_base = "https://gpt-4-proper.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

sys_msg = """You are a senior pathologist. You are given a text where a pathologist is describing a histopathology image.
Generate well-defined question/answer pairs from these sentenses.
Consider the following requirements to generate questions and answers:
- Only generate questions about information being seen in the image.
- Do not generate questions from the context if the question is not explicitly asked.
- Do not generate question/answers based on the information that can not be seen in the image being described like patient's age, gender, medical history, other studies/images outside of the current case.
- If the text does not explicitly provide the answer to a question, do not generate a question/answer pair.
- Do not answer the questions from information outside of the given text. 
- Do not use phrases like "mentioned", "suggests" or “text” in the answers. Instead, refer to the information as being seen "in the image".
- Do not reveal answers in the questions.

Following are three examples:

example 1: they will they will be infiltrative pattern, invasive pattern of the tumor, solid looking tumor with you can see here again, papillary structures are very clearly seen even in the here. This is the picture of serous cyst adenocarcinoma, why we are saying it is carcinoma? Because the lining epithelial cells they have these they have the characteristic feature of atypia, atypia we all know hyperchromatism, pleomorphism, atypical mitotic activity, high MC ratio all these features are seen in these cells along with infiltration. These cells they have infiltrated the underlying stroma, when they have infiltrated the underlying stroma we call this is a malignant tumor, this is serous cyst adenocarcinoma. And we all know that the serous cyst papillary carcinoma we these adenocarcinoma of the ovary this frequently is associated with presence of psammoma bodies and what is somoma body it is a calcified body, rounded calcified bodies are the psammoma bodies and these are the typical features or typical findings that are commonly seen in the papillary serous cyst adenocarcinoma of the ovary. 

QA: {"Why is this image considered a carcinoma?": "The picture is considered a carcinoma because the lining epithelial cells have the characteristic feature of atypia, which includes hyperchromatism, pleomorphism, atypical mitotic activity, high MC ratio, and these features are seen in these cells along with infiltration. These cells have infiltrated the underlying stroma, and when they have infiltrated the underlying stroma, it is called a malignant tumor, specifically serous cyst adenocarcinoma."}

example 2: Okay, now just take a look at the surface over here. I think you know what kind of organ we’re dealing with, don’t you? I think you know this is colon again, don’t you? And you can see a muscularis mucosa here. I could blow it up a little bit if you want, and I will actually. I think you could say, yeah, that looks like a colon. You can see it’s pretty well limited to the muscularis mucosa. But there’s a lot of inflammation. There’s a lot of lymphocytes here in the submucosa. So it’s not a normal colon, but at least you can recognize it as a colon. But let’s take a look at the whole picture now. And let’s take a look at this area here. It’s a colon, and there’s inflammation. But look, what do we have in here as well? Is it pretty clear that these are infiltrating glands? Yes, somebody already said infiltrating glands.

QA: {"What kind of organ is visible in the image?": "The organ visible in the image is the colon.", 
"Are there any signs of infiltrating glands in the image?": "Yes, there are signs of infiltrating glands in the image."}

example 3: And here's the mucosal aspect again. What is the organ and what is going on in this crazy looking mucosa? That's why we have high power on our microscope. You can see that there is an inflammatory process going on, primarily involving the mucosa and submucosa. The muscular layers, mostly nice bands of smooth muscle, do not look like they are too disrupted or infiltrated by inflammatory cells.    

QA: {"What is happening in the mucosa visible in the image?": "There is an inflammatory process going on, primarily involving the mucosa and submucosa. The muscular layers, mostly nice bands of smooth muscle, do not look like they are too disrupted or infiltrated by inflammatory cells."}
"""



@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def prompt_gpt(sys_msg, user_msg, gpt_model="gpt4", temp=0, max_tokens=256):
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]

    response = completions_with_backoff(deployment_id = openai.api_base,
                                         engine=gpt_model, 
                                         temperature=temp,
                                         messages=messages, 
                                         max_tokens=max_tokens)
    
    content = response['choices'][0]['message']['content']
    return content, response


def main(eval_df, sys_msg, gpt_model="gpt4", temp=0, max_tokens=500, logger=None, save_path=None):

    responses = pd.DataFrame()
    start_idx = 0
    end_idx = len(eval_df)
    count = 0
    for idx, row in tqdm(eval_df[start_idx:end_idx].iterrows(), total=len(eval_df[start_idx:end_idx])):

        chunk_text = row["q_padded_text_sentences_asrcorrected"]['utterance']
        q_list = get_full_questions(chunk_text)
        user_msg = generate_user_msg(chunk_text, q_list)

        try:
            content, response = prompt_gpt(sys_msg, user_msg, gpt_model, temp, max_tokens)

            new_row = {"video_id": row["video_id"],
                        "chunk_id": row["chunk_id"],
                        "image_path": row["image_path"],
                        "chunk_text_time": row["q_padded_text_time"],
                        "chunk_text_sentences": row["q_padded_text_sentences"],
                        "chunk_text_sentences_asrcorrected": row["q_padded_text_sentences_asrcorrected"],
                        "QA": content}

            responses = pd.concat([responses, pd.DataFrame([new_row])], ignore_index=True)

            count += 1

        except Exception as e:
            logger.error(f"Failure for row_idx: {idx}. Error: {str(e)}")


        if count % 50 == 0:
            print("Saving responses until count:", count)
            responses.to_json(save_path)
    
    responses.to_json(save_path)


if __name__ == "__main__":
    
    log_file_name = "./data/errors/gpt_failures_eval_QA.log" 
    logging.basicConfig(filename=log_file_name, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    eval_df = pd.read_json('./data/eval_set_data.json')

    save_path = "./data/eval_qa.json"


    main(eval_df, sys_msg, gpt_model="gpt4", temp=0, max_tokens=500, logger=logger, save_path=save_path)