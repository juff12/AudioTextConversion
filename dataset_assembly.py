import json
import os
import argparse
from pathlib import Path

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--output_dir', type=str, default='data/datasets/test', help='output directory of the final data file')
    return parser.parse_args()

def main():
    opt = args()

    # create the directory if it does not exist
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # get all the files in the directory
    subs = [os.path.join(opt.dir, sub) for sub in os.listdir(opt.dir)]
    files = [os.path.join(sub, f) for sub in subs for f in os.listdir(sub) if 'clusters' in f and f.endswith('.txt')]
    
    final_data = []
    for f in files:
        with open(f, 'r') as file:
            data = file.read()
        # split the data by new line
        data = data.split('\n')
        for item in data:
            # make sure its less than 2048 characters and not empty
            if len(item.strip()) >=2048 or item.strip() == '':
                continue
            final_data.append({"text": item.strip()})
    # save the final data
    with open(os.path.join(opt.output_dir, 'final_data.json'), 'w') as file:
        json.dump(final_data, file, indent=4)

if __name__=="__main__":
    main()