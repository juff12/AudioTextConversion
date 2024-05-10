import json
import os
import argparse
from pathlib import Path
from utils.functions import prep_data

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--output_dir', type=str, default='data/datasets/test', help='output directory of the final data file')
    parser.add_argument('--remove_punc', type=bool, default=False, help='remove punctuation from the text')
    parser.add_argument('--restore_punc', type=bool, default=False, help='restore punctuation from the text')
    parser.add_argument('--lower', type=bool, default=False, help='lowercase the text')
    parser.add_argument('--type_data', type=str, default='txt', help='the type of file the preprocessed data is in')
    parser.add_argument('--max_len', type=int, default=3000, help='maximum length of the characters in a string')
    parser.add_argument('--time_cutoff', type=int, default=172800, help='time in seconds that was considered for clustering [default 2 days]')
    parser.add_argument('--is_streamer', type=bool, default=False, help='if the data is from a streamer')
    return parser.parse_args()

def main():
    opt = args()

    # create the directory if it does not exist
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # file extension
    ext = '.' + opt.type_data

    # get all the files in the directory
    subs = [os.path.join(opt.dir, sub) for sub in os.listdir(opt.dir)]
    files = [os.path.join(sub, f) for sub in subs for f in os.listdir(sub) if f'clusters_{opt.time_cutoff}' in f and f.endswith(ext)]
    
    final_data = []
    for f in files:
        with open(f, 'r') as file:
            if opt.type_data == 'json':
                data = json.load(file)
            elif opt.type_data == 'txt':
                data = file.read()
        # split the data by new line
        if opt.type_data == 'txt':
            data = data.split('\n')
            # create json style object
            data = [item for item in data if item != '']
        elif opt.type_data == 'json':
            data = [item['text'] for item in data if item['text'] != '']
        # add data to the final data object
        final_data = final_data + data
    
    final_data = prep_data(final_data, opt.is_streamer, opt.remove_punc,
                           opt.restore_punc, opt.lower, opt.max_len)

    # save the final data
    with open(os.path.join(opt.output_dir, 'final_data.json'), 'w') as file:
        json.dump(final_data, file, indent=4)

if __name__=="__main__":
    main()