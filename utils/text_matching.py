import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv
from presets.opt_text_matching import opt
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score_history(model, history, speech):
    # query to match a chat message to
    query_embedding = model.encode(speech)
    # chat messages
    message_history = [message for message in history]
    passage_embeddings = model.encode(message_history)
    scores = util.dot_score(query_embedding, passage_embeddings)

    return scores.numpy()

def update_history(chat, current_time, limit=40):
    # max number of message to appear in the chat
    history = chat[chat['time'] < current_time]
    if len(history) > limit:
        history = history[len(history) - limit:]
    return history['message'].values.tolist()

def reformat_data(files):
    # reformat the csv files
    for f in files:
        new_data = []
        with open(os.path.join(opt.dir,"{f}/twitch-chat-{f}.csv".format(f=f)), 'r', encoding='utf8') as file:
            reader = csv.reader(file, delimiter=',', dialect=csv.excel_tab)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                new_row = [row[0], row[1], ' '.join(row[3:])]
                row = new_row
                new_data.append(row)
        df = pd.DataFrame(new_data, columns=['time', 'username', 'message'])
        df.to_csv(os.path.join(opt.dir,"{f}/{f}.csv".format(f=f)), index=False)

def match_chat_response(model, audio_text, chat, threshold=0.5, forward_chunks=5, backward_chunks=10, history_limit=100, delay=10):
    pairs = []
    current_time = delay
    i = 0
    # first add each chunk to the response array until a chunk is similar to a chat message
    while i < len(audio_text['chunks']):
        chunk = audio_text['chunks'][i]
        # get the current time for the speaker
        if len(chunk['timestamp']) > 0:
            current_time = chunk['timestamp'][0] + delay
        # update the current chat history    
        history = update_history(chat, current_time, limit=history_limit)
        if len(history) == 0:
            i += 1
            continue
        # score current chunk against history to find the most similar chat message
        scores = score_history(model, history, chunk['text'])
        scores = scores.flatten()
        # if a score above a threshold is found, save the pair
        if (scores.shape[0] > 0) and (np.max(scores, axis=0) > threshold):
            # ending index of the chunk sequence
            end = i + backward_chunks
            if end > len(audio_text['chunks']):
                end = len(audio_text['chunks'])
            # starting index of the chunk sequence
            start = i - forward_chunks
            if start < 0: # set the start
                start = 0
            # get the response window
            response = audio_text['chunks'][start:end]
            if opt.multi_match:
                # add all scores above threshold to the pairs
                for idx, score in enumerate(scores):
                    if score > threshold:
                        max_idx = idx
                        message = history[max_idx]
                        pairs.append({'message': message, 'response': response})
            else:
                # add all scores above threshold to the pairs
                max_idx = int(np.argmax(scores, axis=0))
                message = history[max_idx]
                pairs.append({'message': message, 'response': response})

        i += 1
    return pairs

def save_json(pairs, filename):
    # restructure the data
    with open(filename, 'w') as f:
        json.dump(pairs, f, indent=4)


def score_messages(model, message, response):
    query_embedding = model.encode(message)
    response = response
    passage_embeddings = model.encode(response)
    scores = util.dot_score(query_embedding, passage_embeddings)
    return scores.numpy()[0]

# if the next chat message in the log relates to the current response, end the conversation
def format_conversation(model, pairs, sim_cutoff):
    formatted_pairs = []
    for i, pair in enumerate(pairs):
        if i == len(pairs)-1:
            formatted_pairs.append({'message': pair['message'], 'response': (' '.join([x['text'] for x in pair['response']]).replace('  ', ' '))})
            break
        next_message = pairs[i+1]['message']
        response_list = None
        for j, resp in enumerate(pair['response']):
            if score_messages(model, next_message, resp['text']) >= sim_cutoff:
                response_list = [x['text'] for x in pair['response'][:j]]
                # reset the next message
                break
        # if the conversation is not ended, add the full response
        if response_list is None:
            response_list = [x['text'] for x in pair['response']]
        formatted_pairs.append({'message': pair['message'], 'response': (' '.join(response_list).replace('  ', ' '))})
    return formatted_pairs

def remove_duplicates(chat_pairs):
    # remove the duplicates
    seen = []
    pairs = []
    for pair in chat_pairs:
        if pair['message'] in seen:
            continue
        pairs.append(pair)
        seen.append(pair['message'])
    return pairs

def main():
    # hyper params
    thresh = opt.threshhold
    forward_chunks = opt.forward_chunks
    backward_chunks = opt.backward_chunks
    history_limit = opt.history_limit
    delay = opt.delay

    # get the files in the directory
    files = []
    for f in os.listdir(opt.dir):
        # skip the completed files
        if opt.skip_complete:
            if os.path.exists(os.path.join(opt.dir, "{f}/pairs_{f}_{thresh}_formatted.json".format(f=f, thresh=thresh))):
                continue
        files.append(f)

    # reformat the csv files
    reformat_data(files)
    
    model = SentenceTransformer(opt.sem_model).to(device)

    for id in tqdm(files):
        chat = pd.read_csv(os.path.join(opt.dir,"{f}/{f}.csv".format(f=id)))
        # convert time stamps to integers
        chat[['time']] = chat[['time']].apply(pd.to_numeric)
        audio_text = open(os.path.join(opt.dir,"{f}/audio_text_{f}.json".format(f=id)))
        audio_text = json.load(audio_text)
        pairs = match_chat_response(model, audio_text, chat, threshold=thresh, 
                                    forward_chunks=forward_chunks, backward_chunks=backward_chunks,
                                    history_limit=history_limit, delay=delay)
        # save to json raw file
        save_json(pairs, os.path.join(opt.dir, "{f}/pairs_{f}_{thresh}_raw.json".format(f=id, thresh=thresh)))
        # save to json formatted file
        pairs_no_dup = remove_duplicates(pairs)
        formatted_pairs = format_conversation(model, pairs_no_dup, sim_cutoff=0.7)
        save_json(formatted_pairs, os.path.join(opt.dir, "{f}/pairs_{f}_{thresh}_formatted.json".format(f=id, thresh=thresh)))
if __name__ == "__main__":
    main()