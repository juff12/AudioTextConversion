import os
import re
import argparse
import csv
import pandas as pd

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/', help='directory containing the files to be renamed')
    parser.add_argument('--rename_yt', type=bool, default=False, help='rename youtube files')
    parser.add_argument('--rename_twc', type=bool, default=False, help='rename twitch chat files')
    return parser.parse_args()

def rename_yt(dir):
    endings = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']
    # get the audio files in the directory and store them in a list paired with their ending
    files = [(file, ending) for file in os.listdir(dir) for ending in endings if file.endswith(ending)]
    for file, ext in files:
        # get the video id stored in brackets at the end of the string
        video_id = re.findall(r'\[([^]]+)\]'+ext, file)
        # if there is no ID, skip the file (already processed)
        if video_id == None or len(video_id) == 0:
            return None
        # rename the file with the video ID
        new_name = video_id[0] + ext
        os.rename(os.path.join(dir,file), os.path.join(dir, new_name))

def rename_twc(dir):
    # get the csv files in the directory
    files = [file for file in os.listdir(dir) if file.endswith('.csv')]
    # reformat the csv files to have the correct columns and save them
    reformat_data(files, dir)

def reformat_data(files, dir):
    # reformat the csv files
    for f in files:
        new_data = []
        try:
            with open(os.path.join(dir,f"twitch-chat-{f}.csv"), 'r', encoding='utf8') as file:
                reader = csv.reader(file, delimiter=',', dialect=csv.excel_tab)
                for i, row in enumerate(reader):
                    # skip the first row (titles)
                    if i == 0:
                        continue
                    # reformat the row to have the correct columns
                    new_row = [row[0], row[1], ' '.join(row[3:])]
                    new_data.append(new_row)
            # save the new data to a csv file
            df = pd.DataFrame(new_data, columns=['time', 'username', 'message'])
            df.to_csv(os.path.join(dir,f"{f}.csv"), index=False)
        except: # if the file ends with CSV but does not contain the correct format, skips it
            continue
def main():
    opt = args()
    dir = opt.dir
    if opt.rename_yt:
        rename_yt(dir)
    if opt.rename_twc:
        rename_twc(dir)

if __name__ == "__main__":
    main()