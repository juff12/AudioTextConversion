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
    parser.add_argument('--from_text', type=bool, default=False, help='use txt file instead of json')
    parser.add_argument('--save_type', type=str, default='json', help='type of file to save as [json/txt]')
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
            if opt.from_text:
                with open(os.path.join(dir,f"{id}/clean_text_{id}.txt")) as file:
                    audio_text = file.read()
            else:
                with open(os.path.join(dir,f"{id}/matched_{id}.json")) as file:
                    audio_text = json.load(file)
                audio_text = ' '.join([item['text'] for item in audio_text])
            # get the clusters
            cluster_topics, final_texts = clusterer.cluster(audio_text, thresh_1=opt.thresh_1, thresh_2=opt.thresh_2)
            
            # save the clusters based on type of save given
            if opt.save_type == 'txt':
                clusterer.save_txt(final_texts, os.path.join(dir, f"{id}/clusters_{id}.txt"))
            elif opt.save_type == 'json':
                # save to json
                clusterer.save_json(cluster_topics, final_texts, os.path.join(dir, f"{id}/clusters_{id}.json"))
        except Exception as e:
            print(e)
            continue
if __name__ == "__main__":
    main()