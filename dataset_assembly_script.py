import json
import os
import argparse
from pathlib import Path
import re

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--output_dir', type=str, default='data/datasets/test/', help='output location for the dataset')
    parser.add_argument('--outfile', type=str, default='final_data.json', help='output file name')
    parser.add_argument('--lower', type=bool, default=False, help='lowercase the text')
    parser.add_argument('--file_type', type=str, default='txt', help='the type of file the preprocessed data is in')
    parser.add_argument('--data_type', type=str, default='clusters', help='type of dataset to be processed [default][clusters/chat_response/pairs]')
    parser.add_argument('--time_cutoff', type=int, default=172800, help='time in seconds that was considered for clustering [default 2 days]')
    
    # eos, bos, inst tokens
    parser.add_argument('--eos', type=str, default='</s>', help='end of sentence token')
    parser.add_argument('--bos', type=str, default='<s>', help='beginning of sentence token')
    parser.add_argument('--inst_start', type=str, default='', help='instruction token start')
    parser.add_argument('--inst_end', type=str, default='', help='instruction token end')
    
    return parser.parse_args()

def main():
    opt = args()

    # create the directory if it does not exist
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    # file extension
    ext = '.' + opt.file_type

    filename = '-------------' # placeholder for the filename
    # get all the files in the directory
    subs = [os.path.join(opt.dir, sub) for sub in os.listdir(opt.dir)]
    if opt.data_type == 'clusters':
        filename = f'{opt.data_type}_{opt.time_cutoff}'
    elif 'chat_response' in opt.data_type or 'pairs' in opt.data_type:
        filename = f'{opt.data_type}'


    files = [os.path.join(sub, f) for sub in subs for f in os.listdir(sub) if filename in f and f.endswith(ext)]
    
    final_data = []
    for f in files:
        text = '' # placeholder for the text to be added
        # process each type of input data
        # load the data and prep for clusters
        if opt.data_type == 'clusters':
            with open(f, 'r') as file:
                if opt.file_type == 'json':
                    data = json.load(file)
                    data = [item['text'] for item in data if item['text'] != '']
                elif opt.file_type == 'txt': # open the txt file
                    data = file.read()
                    data = data.split('\n')
                    # create json style object
                    data = [item for item in data if item != '']
            # add bos and eos tokens
            for i, item in enumerate(data):
                text = f'{opt.bos} {item} {opt.eos}'
                # remove multiple spaces
                text = re.sub(r'\s+', ' ', text)
                data[i] = text
        # load and prep the data for chat response
        elif 'chat_response' in opt.data_type or 'pairs' in opt.data_type:
            with open(f, 'r') as file:
                data = json.load(file)
            # add the BOS and EOS tokens
            for i, item in enumerate(data):
                response = item['response'].strip()
                message = item['message'].strip()
                
                
                # missing response or message, add only the available one
                if response == '' or message == '':
                    text = f'{opt.bos} {message} {response} {opt.eos}'
                else:
                    text = f'{opt.bos}{opt.inst_start} {message} {opt.inst_end} {response} {opt.eos}'
                # remove multiple spaces
                text = re.sub(r'\s+', ' ', text)
                data[i] = text
        # add the text to the final data
        final_data += data

    # format the final data
    final_data = [{'text': item} for item in final_data if item != '']


    # save the final data
    with open(os.path.join(opt.output_dir, opt.outfile), 'w') as file:
        json.dump(final_data, file, indent=4)

if __name__=="__main__":
    main()