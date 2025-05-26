import os
import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer_logits
from videollama2.mm_utils import tokenizer_multimodal_token
import argparse
import json
import torch
import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict    
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import scipy.io
from transformers.trainer_pt_utils import (nested_detach)
from tqdm import tqdm

def preprocess_plain(source, tokenizer, modal_token: str = None):
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    assert len(source) == 2
    assert modal_token in source[0]['value']
    message = [
        {'role': 'user', 'content': source[0]['value']},
        {'role': 'assistant', 'content': source[1]['value']}
    ]
    conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
    return dict(input_ids=input_ids) #, labels=targets


def collator(inputs,device):
    batch = dict(
        input_ids=inputs["input_ids"].to(device),
        attention_mask= inputs['attention_mask'],
    )
    batch['images'] = []
    batch['images'].append((inputs['video'], 'video'))
    return batch

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    model.get_tokenizer(tokenizer)
    conversation = [
        {
            "from": "human",
            "value": f"<video>\nPlease evaluate the audio quality, audio-visual content consistency and overall audio-visual quality of the given content one by one. Provide three words to characterize each quality dimension."
        },
        {
            "from": "gpt",
            "value": f"Audio quality: /n, audio-visual consistency: /n, overall audio-visual quality: /n."
        }
    ]

    modal_token = "<video>"
    data_dict = preprocess_plain(conversation, tokenizer, modal_token=modal_token)
    data_dict["input_ids"] = data_dict["input_ids"][0].unsqueeze(0)
    data_dict['attention_mask'] = data_dict["input_ids"][0].ne(tokenizer.pad_token_id).unsqueeze(0)
    preprocess = processor["video"]
    

    audio_video_path = args.video_path
    audio_video_tensor = preprocess(audio_video_path, va=True)
    audio_video_tensor['video'] = audio_video_tensor['video'].half().to(device)
    audio_video_tensor['audio'] = audio_video_tensor['audio'].half().to(device)
    data_dict['video'] = audio_video_tensor
    inputs = collator(data_dict,device)
    with torch.no_grad():
        outputs = model(**inputs,return_dict=False)
    print('Audio quality: {:.4f}, audio-visual consistency: {:.4f}, overall audio-visual quality: {:.4f}'
          .format(outputs[0,0].cpu().item(),outputs[0,1].cpu().item(),outputs[0,2].cpu().item()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, help='') 
    parser.add_argument('--video-path', required=True, help='')
    args = parser.parse_args()

    inference(args)
