import numpy as np
import spacy
import json
import os
import en_core_web_sm
from presets.opt_clustering import opt
from tqdm import tqdm

def process(text, nlp):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i-1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text

def segments_text(text, nlp):
    # Initialize the clusters lengths list and final texts list
    clusters_lens = []
    final_texts = []

    # Process the chunk
    threshold = opt.threshhold
    sents, vecs = process(text, nlp)

    # Cluster the sentences
    clusters = cluster_text(sents, vecs, threshold)

    for cluster in clusters:
        cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
        cluster_len = len(cluster_txt)
        
        # Check if the cluster is too short
        if cluster_len < 60:
            continue
        
        # Check if the cluster is too long
        elif cluster_len > 3000:
            threshold = 0.6
            sents_div, vecs_div = process(cluster_txt, nlp)
            reclusters = cluster_text(sents_div, vecs_div, threshold)
            
            for subcluster in reclusters:
                div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                div_len = len(div_txt)
                
                if div_len < 60 or div_len > 3000:
                    continue
                
                clusters_lens.append(div_len)
                final_texts.append(div_txt)
                
        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)
    
    return clusters_lens, final_texts

def main():
    # get the files in the directory
    files = []
    for f in os.listdir(opt.dir):
        # skip the completed files
        if opt.skip_complete:
            if os.path.exists(os.path.join(opt.dir, "{f}/audio_text_{f}_clusters.json".format(f=f))):
                continue
        files.append(f)
    # get the files in the directory

    # Load the Spacy model
    nlp = en_core_web_sm.load()

    for id in tqdm(files):
        audio_text = open(os.path.join(opt.dir,"{f}/audio_text_{f}.json".format(f=id)))
        audio_text = json.load(audio_text)
        clusters_lens, final_texts = segments_text(audio_text['text'], nlp)

        # save to json
        with open(os.path.join(opt.dir, '{f}/audio_text_{f}_clusters.json'.format(f=id)), 'w') as f:
            data = [{'cluster': cluster_len, 'text': text} for cluster_len, text in zip(clusters_lens, final_texts)]
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()