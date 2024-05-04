import os
import en_core_web_lg
from tqdm import tqdm
from utils import TopicClustering
import json
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--thresh_1', type=float, default=0.3, help='threshold 1 for clustering')
    parser.add_argument('--thresh_2', type=float, default=0.6, help='threshold 2 for clustering')
    return parser.parse_args()

def main():
    opt = args()

    # set the main directory to process
    dir = opt.dir
    # get the files in the directory
    files = [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir, file))]

    # Load the Spacy model
    nlp = en_core_web_lg.load()

    clusterer = TopicClustering(nlp)

    for id in tqdm(files):
        try:
            with open(os.path.join(dir,f"{id}/audio_text_{id}.json")) as file:
                audio_text = json.load(file)
            # get the clusters
            clusters_lens, final_texts = clusterer.cluster(audio_text['text'], thresh_1=opt.thresh_1, thresh_2=opt.thresh_2)

            # save to json
            clusterer.save_json(clusters_lens, final_texts, os.path.join(dir, f"{id}/clusters_{id}.json"))
        except Exception as e:
            print(e)
            continue
if __name__ == "__main__":
    main()