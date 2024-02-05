from PIL import Image
import itertools
import jsonlines
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import copy
import ast
import numpy as np

import bisect
import torch
import open_clip
import glob
import os, re
import json
import traceback
import openai
openai.api_type = "azure"

openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "ENTER YOUR BASE" 
os.environ["OPENAI_API_KEY"] = "ENTER YOUR OPENAI KEY" 
os.environ["TIKTOKEN_CACHE_DIR"] =""

import os 
import backoff
import openai
from litellm import completion



@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
def completions_with_backoff(**kwargs):
    return completion(**kwargs)  


def io_gpt(system_instr, payload, gpt_model="gpt-3.5-turbo", temp=0, max_tokens=256):
    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": payload},
    ]
    response = completions_with_backoff(model=gpt_model, temp=temp, messages=messages, max_tokens=max_tokens)
    content = response['choices'][0]['message']['content']
    return content, response


def replace_words_in_text(text, replace_dict):
    for key, value in replace_dict.items():
        text = re.sub(re.escape(key), value, text, flags=re.IGNORECASE)
    return text


def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)#, return_tensors="pt")
    return image



def calculate_overlap(interval1, interval2):
    """
    Calculate the overlap between two time intervals.
    """
    # Each interval is a tuple (start, end)
    start1, end1 = interval1
    start2, end2 = interval2
    return max(0, min(end1, end2) - max(start1, start2))

def extract_text_from_srt(entire_caption, start_time, end_time, pad_right=0, pad_left=0):
    srt_chunks = [word_entry['end_time'] for word_entry in entire_caption]
    srt_chunks.insert(0, 0.0) # insert 0 time at the start of the video, since we using endtimes
    #print(len(srt_chunks))
    
    srt_start = bisect.bisect(srt_chunks, start_time)
    srt_end = bisect.bisect(srt_chunks, end_time)

    srt_start -= pad_left - 1  # pad one extra cause of the way bisect works ... padding 1 | correct | bisect
    if srt_start < 0:
        srt_start = 0
    srt_end += pad_right

    text_list = []
    for part in range(srt_start, srt_end + 1):  # added one here cause of the way range works
        try:
            text_list.append(entire_caption[part]['utterance'])  # issue was here with list index out of range
        except Exception as e:
            traceback.print_tb(e.__traceback__)
    text = ' '.join(text_list)


    return text


def get_videos_histo_chunk(video_id_path):
    rdata_df = pd.DataFrame()
    list_of_fjf = glob.glob(os.path.join(video_id_path, '*.json'))

    for index__, data_file in enumerate(list_of_fjf):
        with open(data_file, 'r') as f:
            data = json.load(f)

        imchunks = data['img_text_roi_chunks']["chunks"] #roi_chunks to get the roi chunks
        video_id_ = data['video_id']
        video_duration = data['duration']
        pair_chunk_time = data['pair_chunk_time']
        height = data['height']
        width = data['width']
        fps = data['fps']
        histo_chunk_times = data['histo_chunk_times']
        entire_caption = data['video_word_timed_caption']

        if not imchunks:
            print(data)
            continue

        for chunk_index in range(len(imchunks)):
            chunk = imchunks[chunk_index]

            noisy_text_list = chunk['noisy_timed_captions']
            prior_confirmed_maps = chunk['asr_correction_dict']

            start_time = chunk['start_time']
            end_time = chunk['end_time']

            noisy_text = ''.join([utterance['utterance'] for utterance in noisy_text_list])
            clean_text = replace_words_in_text(noisy_text, prior_confirmed_maps)


            for itr in chunk['image_text_roi_map']:
                if itr['image']['path'] and itr['image']['text']:
                    all_text = itr['image']['text']
                    med_num = len(all_text)
                    for roi in itr['roi']:
                        all_text.extend(roi['text'])
                    text = str(all_text)

                    all_num = len(all_text)
                    magn__ = itr['image']['magn']
                    time__ = itr['image']['time']
                    concept_id = itr['image']['umls']
                    roi_list = itr['roi']
                    data = {'video_id': video_id_,
                            'chunk_id': chunk_index,
                            'chunk_start_time': start_time,
                            'chunk_end_time': end_time,
                            'image_path': os.path.basename(itr['image']['path']),
                            'medical_text': all_text[:med_num],
                            'roi_text': all_text[med_num:],
                            'combined_caption': text, 
                            #'med_num': med_num,
                            #'all_num': all_num,
                            'noisy_text': noisy_text,
                            'corrected_text': clean_text,
                            #'map': prior_confirmed_maps,
                            'med_umls_ids': concept_id,
                            'magnification': magn__,
                            #'img_time': time__,
                            #'roi': roi_list,
                            "video_duration": video_duration,
                            "pair_chunk_time": pair_chunk_time,
                            "height": height,
                            "width": width,
                            "fps": fps,
                            "histo_chunk_times": histo_chunk_times,
                            "entire_video_word_caption": entire_caption,
                           }
                    rdata_df = pd.concat([rdata_df, pd.DataFrame([data])], ignore_index=True)
    return rdata_df
