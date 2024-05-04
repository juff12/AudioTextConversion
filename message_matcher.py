from utils import MessageMatcher
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--sent_model', type=str, default='multi-qa-MiniLM-L6-cos-v1', help='name of the sentence transformer semantic model')
    parser.add_argument('--sim_cutoff', type=float, default=0.7, help='similarity cutoff for message matching')
    parser.add_argument('--history_limit', type=int, default=40, help='maximum number of previous messages to consider for matching')
    parser.add_argument('--threshold', type=float, default=0.55, help='threshold for message matching')
    parser.add_argument('--forward_chunks', type=int, default=0, help='number of forward chunks to consider for matching')
    parser.add_argument('--backward_chunks', type=int, default=10, help='number of backward chunks to consider for matching')
    parser.add_argument('--delay', type=int, default=10, help='delay in seconds for message matching')
    parser.add_argument('--multi_match', type=bool, default=False, help='allow multiple matches for a single message')
    return parser.parse_args()

def main():
    opt = args()
    
    # set the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # select the model to use
    model = SentenceTransformer(opt.sent_model).to(device)
    # set the main directory to process
    dir = opt.dir
    
    # create the message matcher
    matcher = MessageMatcher(model, sim_cutoff=opt.sim_cutoff, history_limit=opt.history_limit,
                             threshold=opt.threshold, forward_chunks=opt.forward_chunks,
                             backward_chunks=opt.backward_chunks, delay=opt.delay, multi_match=opt.multi_match)

    # get the sub directories
    sub_dirs = os.listdir(dir)

    for sub in tqdm(sub_dirs):
        chat = pd.read_csv(os.path.join(dir,f"{sub}/{sub}.csv"))
        # convert the time column to numeric
        chat[['time']] = chat[['time']].apply(pd.to_numeric)       

        # open the audio text file
        with open(os.path.join(dir,f"{sub}/audio_text_{sub}.json")) as file:
            audio_text = json.load(file)
        
        # match the messages
        pairs = matcher.match(audio_text, chat)
        # save the pairs to a json file
        file_path = os.path.join(dir, f"{sub}/pairs_{sub}_{opt.threshold}_formatted.json")
        matcher.save_json(pairs, file_path)

if __name__ == "__main__":
    main()