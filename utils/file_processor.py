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
from sentence_transformers import SentenceTransformer, util
import en_core_web_lg
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from abc import ABC, abstractmethod



class FilePreProcessing():
    def __init__(self, directory, is_yt=False, has_twc=False):
        self.dir = directory
        self.yt = is_yt
        self.twc = has_twc
        self.audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']
        # lambda function for checking if a file is an audio file
        self.check_audio_end = lambda x, endings: any([x.endswith(ending) for ending in endings])
    
    def prepare_files(self):
        # rename the youtube files
        if self.yt:
            self.rename_yt_files()
        
        # create a list of the files to create subdirectories for
        audio_files = [x for x in os.listdir(self.dir) 
                       if self.check_audio_end(x, self.audio_extensions)]
        # get the csv files for twc
        csv_files = [x for x in os.listdir(self.dir) if x.endswith('.csv')]

        # create a list of subdirectories from the audio file names
        sub_dirs = [self.create_subdir_names(x) for x in os.listdir(self.dir) 
                    if self.check_audio_end(x, self.audio_extensions)]
        sub_dirs = [x for x in sub_dirs if x is not None] # remove None values
        
        # files to move
        files = audio_files + csv_files

        # create subdirectories
        self.create_subdirs(sub_dirs)

        # remove invalid files
        self.remove_bad_files(sub_dirs)

        # move the audio and csv files to the correct subdirectories
        self.move_files(sub_dirs, files)

        # process the twc files if the user wants to
        if self.twc:
            self.reformat_twc_files()

    def reformat_twc_files(self):
        # reformat the csv files
        for sub in os.listdir(self.dir):
            path = os.path.join(self.dir,sub)
            new_data = []
            # open the file
            with open(os.path.join(path,f"twitch-chat-{sub}.csv"), 'r', encoding='utf8') as file:
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
            df.to_csv(os.path.join(path,f"{sub}.csv"), index=False)

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

    def remove_bad_files(self, subs):
        # remove the invalid files
        for file in os.listdir(self.dir):
            # skip subdirectories
            # checks if the files is an audio file or csv file and that it contains a sub directory as part of the name
            if (file in subs or ((self.check_audio_end(file, self.audio_extensions) or file.endswith('.csv')) 
                and any([True if sub in file else False for sub in subs]))):
                continue
            else:
                # remove invalid files
                os.remove(os.path.join(self.dir, file))

    def move_files(self, subs, files):
        for sub in subs:
            # get the files that cotain the sub string
            file_list = [file for file in files if sub in file]
            # loop through each file and move it to the correct subdirectory
            for file in file_list:
                # function to check that the file is an audio file ending
                if self.check_audio_end(file, self.audio_extensions) or file.endswith('.csv'):
                    curr_loc = os.path.join(self.dir, file)
                    new_loc = os.path.join(self.dir, sub, file)
                    shutil.move(curr_loc, new_loc)
                else:
                    # remove invalid files
                    os.remove(os.path.join(self.dir, file))

    def create_subdirs(self, subs):
        for sub in subs:
            dir = os.path.join(self.dir, sub)
            Path(dir).mkdir(parents=True, exist_ok=True)
    
    def get_audio_files(self):
        audio_files = []
        for file in os.listdir(self.dir):
            for ending in self.audio_extensions:
                if file.endswith(ending):
                    audio_files.append(os.path.join(self.dir, file))
        if len(audio_files) == 0:
            return None
        return audio_files[0] # return the first audio file

class TextCleaner():
    def __init__(self,):
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.nlp = en_core_web_lg.load()
        self.clean_json_text = None
        self.clean_txt_text = None
        self.punc = ['.', ',', '?', '!']

    def remove_repeat_punct(self, text):
        # remove repeated punctuation
        tokens = self.tokenizer.tokenize(text)
        # Check if the tokens array is empty
        if len(tokens) == 0:
            return text
        # Iterate through the tokens, skipping repetitions of punctuation
        for i in range(1, len(tokens)):
            if tokens[i] in self.punc and tokens[i-1] in self.punc:
                if tokens[i] == '.' and tokens[i-1] == '.': # dont remove ellipsis
                    continue
                tokens[i] = ''
        # reconstruct the text
        cleaned_text = self.detokenizer.detokenize(tokens)
        return cleaned_text
    
    def remove_repeat(self, text):
        tokens = self.tokenizer.tokenize(text)  # Tokenize the text
        # Check if the tokens array is empty
        if len(tokens) == 0:
            return text
        cleaned_tokens = [tokens[0]]  # Initialize list with first token
        prev_token = tokens[0]  # Initialize previous token with first token        
        # Iterate through the tokens, skipping repetitions
        for i in range(1, len(tokens)):
            # Check if the current token is the same as the previous one
            if tokens[i] in self.punc:
                cleaned_tokens.append(tokens[i])
            elif tokens[i] != prev_token:
                cleaned_tokens.append(tokens[i])  # If not the same, add to cleaned list
                prev_token = tokens[i]
        cleaned_text = self.detokenizer.detokenize(cleaned_tokens)  # Detokenize the cleaned tokens
        cleaned_text = self.remove_repeat_punct(cleaned_text) # remove repeated punctuation
        return cleaned_text  # Return the cleaned tokens array
    
    def score_sentences(self, prev_sent, curr_sent):
        # get the embedding of the sentence
        prev_sent = self.semantic_model.encode(prev_sent, convert_to_tensor=True)
        curr_sent = self.semantic_model.encode(curr_sent, convert_to_tensor=True)
        # calculate the cosine similarity between the message and the spam and ham embeddings
        score = util.cos_sim(prev_sent, curr_sent)
        return score.cpu().numpy().flatten()[0]

    def remove_repeat_sents(self, text):
        # remove similar repeated sentences
        sentences = sent_tokenize(text)
        previous = '' # set the first sentence
        for i, sentence in enumerate(sentences):
            if previous == '':
                previous = sentence
                continue
            if sentence == previous or self.score_sentences(previous, sentence) > 0.79:
                sentences[i] = ''
                continue
            previous = sentence
        text = ' '.join([sentence for sentence in sentences if sentence != ''])
        return text

    def remove_emojis(self, text):
        # Remove emojis from text
        cleaned_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
        return cleaned_text

    def remove_unicode(self, text):
        # Remove Unicode characters using regex   
        return re.sub(r'[\u0080-\uffff]', '', text)

    def fix_spacing(self, text: str):
        # remove spaces before punctuation
        text = text.replace(' .', '.').replace(' ,', ',')
        text = text.replace(' ?', '?').replace(' !', '!')

        # add spaces to ellipsis 
        text = text.replace('...', ' ... ')

        # remove double periods
        text = text.replace('..', '. ')

        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text

    def restore_punctuation(self, text):
        # retore the punctuation, source code modified for smaller chunks
        # text = self.punc_model.restore_punctuation(text)
        punc_text = self.nlp(text)
        text = ' '.join([token.text for token in punc_text.sents])
        return text

    # remove n-gram repeats
    def remove_ngrams_sent(self, sent):
        split_sent = sent.split(' ')
        # maximum length of a repeat can only be 1/2 the total words in sentence
        max_ngram = len(split_sent) // 2
        winsz = max_ngram
        # iterate through each window size
        while winsz > 0:
            # iterate through each starting position
            i = 0
            while i + winsz < len(split_sent) and i + 2 * winsz <= len(split_sent):
                # start of the first window
                end_w1 = i + winsz
                # start of the second window
                end_w2 = end_w1 + winsz
                # check if the subsets are equivalent
                subset_1 = split_sent[i:end_w1]
                subset_2 = split_sent[end_w1:end_w2]
                # convert the subsets to lowercase
                subset_1 = [word.lower() for word in subset_1]
                subset_2 = [word.lower() for word in subset_2]
                #print(f"subset_1: {subset_1}, subset_2: {subset_2}")
                # if the subsets are equivalent, remove the second subset
                if subset_1 == subset_2:
                    # remove the second subset
                    for _ in range(winsz):
                        split_sent.pop(end_w1)
                else:
                    i += 1
            # reduce the window size
            winsz -= 1
            # update the max ngram
            max_ngram = len(split_sent) // 2
            # if the max_ngram is smaller, select that
            winsz = min(winsz, max_ngram)
        return ' '.join(split_sent)

    def remove_ngrams(self, text):
        # remove n-grams from the text
        sentences = sent_tokenize(text)
        for i in range(len(sentences)):
            # get the punctuation
            punc = ''
            if len(sentences[i]) == 0:
                continue
            elif sentences[i][-1] in self.punc:
                punc = sentences[i][-1]
            else:
                punc = '.'
                
            # remove commas and ellipsis
            temp_sent = sentences[i].replace(',', '').replace('...', ' ')
            # remove the ngrams from the sentence
            sentences[i] = self.remove_ngrams_sent(temp_sent)
            # add the punctuation back
            sentences[i] = sentences[i] + punc
        text = ' '.join(sentences)
        return text

    def clean_text(self, text):
        # remove emojis
        text = self.remove_emojis(text)
        # remove unicode characters
        text = self.remove_unicode(text)
        # remove repeats, again
        text = self.remove_repeat(text)
        # remove n-grams
        text = self.remove_ngrams(text)
        # restore the punctuation
        text = self.restore_punctuation(text)
        # remove repeat sentences
        text = self.remove_repeat_sents(text)
        # remove repeats again
        text = self.remove_repeat(text)
        # fix spacing
        text = self.fix_spacing(text)
        return text.strip()

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
        self.stop_words = set(stopwords.words('english'))
        self.punc = set(string.punctuation)

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

    def cluster(self, text, thresh_1=0.3, thresh_2=0.6, min_len=60, max_len=3000):
        # Initialize the clusters lengths list and final texts list
        cluster_topics = []
        final_texts = []

        # Process the chunk
        sents, vecs = self.process(text)

        # Cluster the sentences
        clusters = self.cluster_text(sents, vecs, thresh_1)

        for cluster in clusters:
            cluster_txt = self.cleaner.clean_text(' '.join([sents[i].text for i in cluster]))
            cluster_len = len(cluster_txt)
            
            # Check if the cluster is too short
            if cluster_len < min_len:
                continue
            
            # Check if the cluster is too long
            elif cluster_len > max_len:
                sents_div, vecs_div = self.process(cluster_txt)
                reclusters = self.cluster_text(sents_div, vecs_div, thresh_2)
                
                for subcluster in reclusters:
                    div_txt = self.cleaner.clean_text(' '.join([sents_div[i].text for i in subcluster]))
                    div_len = len(div_txt)
                    
                    if div_len < 60 or div_len > 3000:
                        continue
                    
                    # TODO: get the topic of the cluster
                    topic = self.cluster_topics(sent_tokenize(div_txt))
                    cluster_topics.append(topic)
                    final_texts.append(div_txt)
                    
            else:
                # TODO: get the topic of the cluster
                topic = self.cluster_topics(sent_tokenize(cluster_txt))
                cluster_topics.append(topic)
                final_texts.append(cluster_txt)
        return cluster_topics, final_texts

    def cluster_topics(self, cluster_txt):
        # TODO: get the topic of the cluster
        return 0

    def save_json(self, cluster_topics, final_texts, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                # zip the clusters and texts for itteration
                zipped_obj = zip(cluster_topics, final_texts)
                data = [{'topic': topic, 'text': text} for topic, text in zipped_obj]
                json.dump(data, file, indent=4)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving json file')
            print("Error: ", e)
    
    def save_txt(self, final_texts, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                # write each line on a newline
                for text in final_texts:
                    text = text + '\n'
                    file.write(text)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving text file')
            print("Error: ", e)


class AbstractMatcher(ABC):
    def __init__(self, model, message_sim=0.6, speech_sim=0.6, delay=10):
        super().__init__()
        self.cleaner = TextCleaner()
        self.model = model
        self.message_sim = message_sim
        self.speech_sim = speech_sim
        self.delay = delay

    @abstractmethod
    def _group_audio(self, audio_text):
        new_audio = []
        # merge the similar speakers
        i = 1
        last_speakers = audio_text[0]['speaker_id']
        new_entry = {'speaker_id': last_speakers, 'text': audio_text[0]['text'], 'timestamp': audio_text[0]['timestamp']}
        while i < len(audio_text):
            # check if the nex speech segment has the same speakers
            if all([True if speaker in last_speakers else False for speaker in audio_text[i]['speaker_id']]):
                # update the text
                new_entry['text'] += ' ' + audio_text[i]['text']
                # update the timestamp
                new_entry['timestamp'][1] = audio_text[i]['timestamp'][1]
            else:
                new_audio.append(new_entry)
                last_speakers = audio_text[i]['speaker_id']
                new_entry = {'speaker_id': last_speakers, 'text': audio_text[i]['text'], 'timestamp': audio_text[i]['timestamp']}
            i += 1
        # add the last entry
        new_audio.append(new_entry)

        # clean the text
        for i, item in enumerate(new_audio):
            new_audio[i]['text'] = self.cleaner.clean_text(item['text'])
        
        return new_audio
    
    @abstractmethod
    def _refine_chat(self, chat):
        # format the chat data
        chat[['time']] = chat[['time']].apply(pd.to_numeric)
        chat[['message']] = chat[['message']].astype(str)

        chat = chat[~chat['message'].str.contains('@')]
        # remove messages with less than 3 words
        chat = chat[chat['message'].str.split().apply(len) > 2]
        return chat
    
    @abstractmethod
    def score_messages(self, messages, speech):
        message_embeddings = self.model.encode(messages, convert_to_tensor=True)
        speech_embedding = self.model.encode(speech, convert_to_tensor=True)
        scores = util.cos_sim(message_embeddings, speech_embedding)
        return scores.cpu().numpy()

    @abstractmethod
    def score_speech(self, audio_group, new_speech):
        group_embeddings = self.model.encode(audio_group, convert_to_tensor=True)
        new_speech_embedding = self.model.encode(new_speech, convert_to_tensor=True)
        scores = util.cos_sim(new_speech_embedding, group_embeddings)
        scores = scores.cpu().numpy()
        if np.mean(scores) < self.speech_sim:
            return False # the speech is not related to the conversation
        return True # the speech is related to the converstation

    @abstractmethod
    def save_json(self, pairs, filename):
        try:
            # save the data
            with open(filename, 'w') as file:
                json.dump(pairs, file, indent=4)
        except Exception as e: # file couldnt be opened or no data to save
            print('Error saving json file')
            print("Error: ", e)

class MessageMatcher(AbstractMatcher):
    def __init__(self, model, message_sim=0.7, history_limit=40, speech_sim=0.5,
                 forward_chunks=0, backward_chunks=10, delay=10, multi_match=False):
        super().__init__(model, message_sim, speech_sim, delay)
        self.history_limit = history_limit
        self.forward_chunks = forward_chunks
        self.backward_chunks = backward_chunks
        self.multi_match = multi_match
    
    def _refine_chat(self, chat):
        return super()._refine_chat(chat)

    def score_speech(self, audio_group, new_speech):
        return super().score_speech(audio_group, new_speech)

    def score_messages(self, messages, speech):
        return super().score_messages(messages, speech)
    def save_json(self, pairs, filename):
        return super().save_json(pairs, filename)

    def _group_audio(self, audio_text):
        return super()._group_audio(audio_text)
    
    def check_matches(self, text, message):
        i = 0
        segments = []
        n = len(message) # length of the observation window
        while i < len(text):
            # break the text into segments
            if i + n > len(text):
                segments.append(text[i:])
                break
            else:
                segments.append(text[i:i + n])
            i += 1
        
        # score all the segments
        scores = self.score_messages(segments, message).flatten()

        if np.max(scores) > self.message_sim:
            idx = np.argmax(scores)
            return text[idx:]
        return None # no matches
    
    def fast_match_chat_response(self, audio_text, chat):
        pairs = []
        for k, item in tqdm(enumerate(audio_text), total=len(audio_text), desc='Pairing', ncols=100):
            # check the inveral is valid
            interval = item['timestamp']
            if interval[1] is None:
                interval[1] = interval[0] + self.delay
            
            start = interval[0]
            end = interval[1] + self.delay

            chat_window = chat['message'][(chat['time'] >= start) & (chat['time'] <= end)].values
            
            # no matches to be made            
            if item['text'] == '' or len(chat_window) == 0:
                continue
            
            # the streamers speech
            sents = sent_tokenize(item['text'])
            
            # break the speech into senetences
            scores = self.score_messages(chat_window, sents)

            assert scores.shape[0] == len(chat_window) # confirm the scores are the same length as the chat messages
            assert scores.shape[1] == len(sents) # confirm the scores are the same length as the chat messages
            # find where the streamer reads out a message from the chat
            for i, message in enumerate(chat_window):
                for j, sent in enumerate(sents):
                    # the message is related to the sentence
                    if scores[i][j] > self.message_sim:

                        response = ' '.join(sents[j:])
                        # check if the response goes onto the next item
                        if k+1 < len(audio_text):
                            response += ' ' + audio_text[k+1]['text']
                        
                        pairs.append({'context': ' '.join(sents),
                                      'message': message, 'response': response,
                                      'score': str(scores[i][j])})
        return pairs

    def deep_chat_message_match(self, audio_text, chat):
        pairs = []
        for item in tqdm(audio_text, total=len(audio_text), desc='Pairing', ncols=100):
            # check the inveral is valid
            interval = item['timestamp']
            if interval[1] is None:
                interval[1] = interval[0] + self.delay
            
            start = interval[0]
            end = interval[1] + self.delay

            chat_window = chat['message'][(chat['time'] >= start) & (chat['time'] <= end)].values
            
            # the streamers speech
            speech = item['text']
            # no matches to be made            
            if speech == '' or len(chat_window) == 0:
                continue
            
            # check if the speech is related to the chat
            for message in chat_window:
                response = self.check_matches(speech, message)
                # if a match was found, add it
                if response is not None:
                    pairs.append({'message': message, 'response': response})
                    speech = response # move the window forward
        return pairs


    def match(self, audio_text, chat):
        audio_text = self._group_audio(audio_text)
        chat = self._refine_chat(chat)
        pairs = self.fast_match_chat_response(audio_text, chat)
        return pairs

class ResponseMatcher(AbstractMatcher):
    def __init__(self, model, message_sim=0.6, speech_sim=0.6, message_delay=10):
        super().__init__(model, message_sim, speech_sim, message_delay)

    def _refine_chat(self, chat):
        return super()._refine_chat(chat)

    def score_speech(self, audio_group, new_speech):
        return super().score_speech(audio_group, new_speech)

    def score_messages(self, messages, speech):
        return super().score_messages(messages, speech)
    def save_json(self, pairs, filename):
        return super().save_json(pairs, filename)

    def _group_audio(self, audio_text):
        return super()._group_audio(audio_text)

    def match_chat_response(self, audio_text, chat):
        pairs = []
        # iterate through each audio chunk and match to chat messages
        for item in tqdm(audio_text, total=len(audio_text), desc='Pairing', ncols=100):
            interval = item['timestamp']
            if interval[1] is None:
                interval[1] = interval[0] + self.delay
            speech = item['text']
            # iterate through the chat message in the time interval
            interval_chat = chat['message'][(chat['time'] >= interval[0] + self.delay) & (chat['time'] <= interval[1] + self.delay)].values
            # skip empty speech and intervals with no messages
            if speech == '' or len(interval_chat) == 0:
                continue
            
            # score the messages
            scores = self.score_messages(interval_chat, speech)
    
            assert len(scores) == len(interval_chat) # confirm the scores are the same length as the chat messages
            for i, score in enumerate(scores):
                # make sure the message is related, but is not the streamer reading the message
                if score >= self.message_sim and score <= 0.75:
                    response = self.cleaner.clean_text(str(interval_chat[i]))
                    pairs.append({ 'message': speech, 'response': response, 'score': str(score)})
        return pairs