import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import TextCleaner
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    return parser.parse_args()

def clean_matched_pairs(cleaner, file_path, time_seconds=3600):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # get the speakers before a certain time
    speakers = get_speakers_before_time(data, time_seconds)
    
    # store all the data with the speakers in the segment of time in a new list
    new_data = [item for item in data if any([True if speaker in speakers else False for speaker in item['speaker_id']])]
    
    ##########################
    # Consider adding partial cleaning here, currently the speed is slow so it isnt worth it
    ##########################

    with open(file_path.replace('matched', 'clean_matched'), 'w') as f:
        json.dump(new_data, f, indent=4)

# get the speakers befor a certain time
def get_speakers_before_time(data, time_seconds=3600):
    # get the speakers before a certain time
    i = 0
    speakers = set()
    while data[i]['timestamp'][0] < time_seconds:
        # add the speakers to the lsit
        for speaker in data[i]['speaker_id']:
            if speaker not in speakers:
                speakers.append(speaker)
        i += 1
    return speakers

def main():
    opt = args()
    parent_dir = opt.dir
    # get the sub directories
    sub_dirs = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    # get the unprocessed matched json files
    files = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir) if f.endswith('.json') and 'matched' in f and 'clean' not in f]    
    
    # create the text cleaner
    cleaner = TextCleaner()

    # clean the json files
    for file in files:
        clean_matched_pairs(cleaner, file)

    # get the cleaned json files directory
    files = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir) if f.endswith('.json') and 'clean_matched' in f]
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
        text = ' '.join([item['text'] for item in data])
        # clean the text
        cleaned_text = cleaner.clean_text(text)
        # save the cleaned text as a txt file
        with open(f.replace('clean_matched', 'clean_text').replace('.json', '.txt'), 'w') as file:
            file.write(cleaned_text)

if __name__ == "__main__":
    main()