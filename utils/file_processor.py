import os
import shutil
from pathlib import Path
import re
import csv
import pandas as pd
import json
from nltk.tokenize import sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import emoji
import numpy as np
from sentence_transformers import util

class FilePreProcessing():
    def __init__(self, directory, is_yt=False, has_twc=False):
        self.dir = directory
        self.yt = is_yt
        self.twc = has_twc
        self.audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']

    def prepare_files(self):
        # rename the youtube files
        if self.yt:
            self.rename_yt_files()
        # process the twc files if the user wants to
        if self.twc:
            self.reformat_twc_files()

        # function to check that the file is an audio file ending
        check_audio_end = lambda x, endings: any([x.endswith(ending) for ending in endings])

        # create a list of the files to create subdirectories for
        audio_files = [x for x in os.listdir(self.dir) if check_audio_end(x, self.audio_extensions)]
        # get the csv files for twc
        csv_files = [x for x in os.listdir(self.dir) if x.endswith('.csv')]

        # create a list of subdirectories from the audio file names
        sub_dirs = [self.create_subdir_names(x, self.audio_extensions) for x in os.listdir(self.dir) if check_audio_end(x, self.audio_extensions)]
        sub_dirs = [x for x in sub_dirs if x is not None] # remove None values
        
        # files to move
        files = audio_files + csv_files

        # create subdirectories
        self.create_subdirs(sub_dirs)

        # move the audio and csv files to the correct subdirectories
        self.move_files(sub_dirs, files)

    def reformat_twc_files(self, twc_files):
        # reformat the csv files
        for f in twc_files:
            new_data = []
            # open the file
            with open(os.path.join(self.dir,f"twitch-chat-{f}.csv"), 'r', encoding='utf8') as file:
                reader = csv.reader(file, delimiter=',', dialect=csv.excel_tab)
                # loop through each row
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    # new row, removes excess columns
                    new_row = [row[0], row[1], ' '.join(row[3:])]
                    row = new_row
                    new_data.append(row)
            # save the new data to a csv file
            df = pd.DataFrame(new_data, columns=['time', 'username', 'message'])
            df.to_csv(os.path.join(self.dir,f"{f}.csv"), index=False)

    def create_subdir_names(self, filename):
        for ending in self.audio_extensions:
            if filename.endswith(ending):
                return filename.replace(ending, '')
        return None # not a not file

    def new_yt_name(self, file):
        # find the ending file extension
        for ext in self.audio_extensions:
            if file.endswith(ext):
                # rename the file
                result = re.findall(r'\[([^]]+)\]'+ext, file)
                if result == None or len(result) == 0:
                    return None
                # return a string of the new audio file name
                return result[0] + ext
        return None
    
    def rename_yt_files(self):
        # renames each yt audio file in the folder
        for file in os.list(self.dir):
            # rename the files
            new_name = self.new_yt_name(file)
            if new_name is None:
                continue
            try:
                os.rename(os.path.join(self.dir, file), os.path.join(self.dir, new_name))
            except:
                pass
    
    def move_files(self, subs, files):
        for sub in subs:
            # get the files that cotain the sub string
            file_list = [file for file in files if sub in file]
            # loop through each file and move it to the correct subdirectory
            for file in file_list:
                curr_loc = os.path.join(self.dir, file)
                new_loc = os.path.join(self.dir, sub, file)
                shutil.move(curr_loc, new_loc)

    def create_subdirs(self, subs):
        for sub in subs:
            dir = os.path.join(self.dir, sub)
            Path(dir).mkdir(parents=True, exist_ok=True)
    
    def get_audio_files(self, directory, audio_file_endings):
        audio_files = []
        for file in os.listdir(directory):
            for ending in audio_file_endings:
                if file.endswith(ending):
                    audio_files.append(os.path.join(directory, file))
        if len(audio_files) == 0:
            return None
        return audio_files[0] # return the first audio file

class TextCleaner():
    def __init__(self,):
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        self.clean_json_text = None
        self.clean_txt_text = None

    def remove_repeat_sents(self, text):
        sentences = sent_tokenize(text)
        current = ''
        for i, sentence in enumerate(sentences):
            if current == '':
                current = sentence
                continue
            if sentence == current:
                sentences[i] = ''
                continue
            elif sentence != current:
                current = sentence
        text = ' '.join([sentence for sentence in sentences if sentence != ''])
        return text

    def remove_repeat(self, text):
        tokens = self.tokenizer.tokenize(text)  # Tokenize the text
        # Check if the tokens array is empty
        if len(tokens) == 0:
            return text
        cleaned_tokens = [tokens[0]]  # Initialize list with first token

        # Iterate through the tokens, skipping repetitions
        for i in range(1, len(tokens)):
            # Check if the current token is the same as the previous one
            if tokens[i] != tokens[i - 1]:
                cleaned_tokens.append(tokens[i])  # If not the same, add to cleaned list
        cleaned_text = self.detokenizer.detokenize(cleaned_tokens)  # Detokenize the cleaned tokens
        return cleaned_text  # Return the cleaned tokens array
    
    def remove_emojis(self, text):
        # Remove emojis from text
        cleaned_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
        return cleaned_text

    def remove_unicode(self, text):
        # Remove Unicode characters using regex   
        return re.sub(r'[\u0080-\uffff]', '', text)

    def fix_spacing(self, text: str):
        # remove double spaces
        text = text.replace('  ', ' ')
        # remove spaces before punctuation
        text = text.replace(' .', '.').replace(' ,', ',')
        text = text.replace(' ?', '?').replace(' !', '!')
        return text

    def clean_text(self, text):
        # remove emojis
        text = self.remove_emojis(text)
        # remove unicode characters
        text = self.remove_unicode(text)
        # remove repeat sentences
        text = self.remove_repeat_sents(text)
        # remove repeated words and phrases
        text = self.remove_repeat(text)
        # fix spacing
        text = self.fix_spacing(text)
        return text

    def clean_json(self, json_file):
        # open the json file
        with open(json_file, 'r') as file:
            audio_text = json.load(file)
        
        # get the text from the json file
        for i in range(len(audio_text)):
            # get the text
            text = audio_text[i]['text']
            # clean text
            text = self.clean_text(text)
            # save cleaned text
            audio_text[i]['text'] = text
        self.clean_json_text = audio_text
        return audio_text

    def clean_txt(self, txt_file):
        with open(txt_file, 'r') as file:
            text = file.read()
        # clean text
        self.clean_txt_text = self.clean_text(text)
        return self.clean_txt_text
    
    def save_json(self, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                json.dump(self.clean_json_text, file, indent=4)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving json file')
            print("Error: ", e)

    def save_txt(self, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                file.write(self.clean_txt_text)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving txt file')
            print("Error: ", e)

class TopicClustering():
    def __init__(self, nlp):
        self.nlp = nlp
        self.cleaner = TextCleaner()

    def process(self, text):
        doc = self.nlp(text)
        sents = list(doc.sents)
        vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

        return sents, vecs
    
    def cluster_text(self, sents, vecs, threshold):
        # initialize the clusters
        clusters = [[0]]
        for i in range(1, len(sents)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                clusters.append([])
            clusters[-1].append(i)
        
        return clusters

    def cluster(self, text, thresh_1=0.3, thresh_2=0.6):
        # Initialize the clusters lengths list and final texts list
        clusters_lens = []
        final_texts = []

        # Process the chunk
        sents, vecs = self.process(text)

        # Cluster the sentences
        clusters = self.cluster_text(sents, vecs, thresh_1)

        for cluster in clusters:
            cluster_txt = self.cleaner.clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)
            
            # Check if the cluster is too short
            if cluster_len < 60:
                continue
            
            # Check if the cluster is too long
            elif cluster_len > 3000:
                sents_div, vecs_div = self.process(cluster_txt)
                reclusters = self.cluster_text(sents_div, vecs_div, thresh_2)
                
                for subcluster in reclusters:
                    div_txt = self.cleaner.clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)
                    
                    if div_len < 60 or div_len > 3000:
                        continue
                    
                    clusters_lens.append(div_len)
                    final_texts.append(div_txt)
                    
            else:
                clusters_lens.append(cluster_len)
                final_texts.append(cluster_txt)
        return clusters_lens, final_texts
    
    def save_json(self, clusters_lens, final_texts, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                # zip the clusters and texts for itteration
                zipped_obj = zip(clusters_lens, final_texts)
                data = [{'cluster': cluster, 'text': text} for cluster, text in zipped_obj]
                json.dump(data, file, indent=4)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving json file')
            print("Error: ", e)
    
class MessageMatcher():
    def __init__(self, model, sim_cutoff=0.7, history_limit=40, threshold=0.5,
                 forward_chunks=0, backward_chunks=10, delay=10, multi_match=False):
        self.model = model
        self.sim_cutoff = sim_cutoff
        self.history_limit = history_limit
        self.cleaner = TextCleaner()
        self.threshold = threshold
        self.forward_chunks = forward_chunks
        self.backward_chunks = backward_chunks
        self.delay = delay
        self.multi_match = multi_match

    def score_history(self, history, speech):
        # query to match a chat message to
        query_embedding = self.model.encode(speech)
        # chat messages
        message_history = [message for message in history]
        passage_embeddings = self.model.encode(message_history)
        scores = util.dot_score(query_embedding, passage_embeddings)

        return scores.numpy()
    
    def update_history(self, chat, current_time):
        # max number of message to appear in the chat
        history = chat[chat['time'] < current_time]
        if len(history) > self.history_limit:
            history = history[len(history) - self.history_limit:]
        return history['message'].values.tolist()

    def score_messages(self, message, response):
        query_embedding = self.model.encode(message)
        response = response
        passage_embeddings = self.model.encode(response)
        scores = util.dot_score(query_embedding, passage_embeddings)
        return scores.numpy()[0]
    
    # if the next chat message in the log relates to the current response, end the conversation
    def format_conversation(self, pairs):
        formatted_pairs = []
        for i, pair in enumerate(pairs):
            if i == len(pairs)-1:
                formatted_pairs.append({'message': pair['message'], 'response': (' '.join([x['text'] for x in pair['response']]).replace('  ', ' '))})
                break
            next_message = pairs[i+1]['message']
            response_list = None
            for j, resp in enumerate(pair['response']):
                if self.score_messages(next_message, resp['text']) >= self.sim_cutoff:
                    response_list = [x['text'] for x in pair['response'][:j]]
                    # reset the next message
                    break
            # if the conversation is not ended, add the full response
            if response_list is None:
                response_list = [x['text'] for x in pair['response']]
            formatted_pairs.append({'message': pair['message'], 'response': (' '.join(response_list).replace('  ', ' '))})
        return formatted_pairs
    
    def match_chat_response(self, audio_text, chat):
        pairs = []
        current_time = self.delay
        i = 0
        # first add each chunk to the response array until a chunk is similar to a chat message
        while i < len(audio_text['chunks']):
            chunk = audio_text['chunks'][i]
            # get the current time for the speaker
            if len(chunk['timestamp']) > 0:
                current_time = chunk['timestamp'][0] + self.delay
            # update the current chat history    
            history = self.update_history(chat, current_time)
            if len(history) == 0:
                i += 1
                continue
            # score current chunk against history to find the most similar chat message
            scores = self.score_history(history, chunk['text'])
            scores = scores.flatten()
            # if a score above a threshold is found, save the pair
            if (scores.shape[0] > 0) and (np.max(scores, axis=0) > self.threshold):
                # ending index of the chunk sequence
                end = i + self.backward_chunks
                if end > len(audio_text['chunks']):
                    end = len(audio_text['chunks'])
                # starting index of the chunk sequence
                start = i - self.forward_chunks
                if start < 0: # set the start
                    start = 0
                # get the response window
                response = audio_text['chunks'][start:end]
                if self.multi_match:
                    # add all scores above threshold to the pairs
                    for idx, score in enumerate(scores):
                        if score > self.threshold:
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
    
    def clean_pairs(self, pairs):
        # clean the text in the pairs
        for i in range(len(pairs)):
            pairs[i]['message'] = self.cleaner.clean_text(pairs[i]['message'])
            pairs[i]['response'] = self.cleaner.clean_text(pairs[i]['response'])
        return pairs

    def match(self, audio_text, chat):
        pairs = self.match_chat_response(audio_text, chat)
        pairs = self.format_conversation(pairs)
        pairs = self.clean_pairs(pairs)
        return pairs
    
    def save_json(self, pairs, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                json.dump(pairs, file, indent=4)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving json file')
            print("Error: ", e)