import json
import os
import argparse
from utils import TextCleaner
from utils.functions import clean_matched_pairs
from tqdm import tqdm

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--time_seconds', type=int, default=3600, help='time in seconds to consider for getting the speakers')
    return parser.parse_args()
    
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
    for file in tqdm(files):
        clean_matched_pairs(cleaner, file, opt.time_seconds)

    # get the cleaned json files directory
    files = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir) if f.endswith('.json') and 'clean_matched' in f]
    for f in tqdm(files):
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