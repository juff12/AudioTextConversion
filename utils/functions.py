import os
import json
import re
from deepmultilingualpunctuation import PunctuationModel
from nltk.tokenize import sent_tokenize
from transformers import logging
from tqdm import tqdm

logging.set_verbosity(logging.ERROR)

# returns the audio file from the directory
def get_audio_files(directory, audio_file_endings):
    audio_files = []
    for file in os.listdir(directory):
        for ending in audio_file_endings:
            if file.endswith(ending):
                audio_files.append(os.path.join(directory, file))
    if len(audio_files) == 0:
        return None
    return audio_files[0] # return the first audio file

def clean_matched_speakers(cleaner, file_path, time_seconds=3600):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # get the speakers before a certain time
    try:
        speakers = get_speakers_before_time(data, time_seconds)
    except Exception as e:
        print(f"Error: {e}")
        print(f"File: {file_path}")
        return
    # store all the data with the speakers in the segment of time in a new list
    new_data = [item for item in data if any([True if speaker in speakers else False for speaker in item['speaker_id']])]
    
    ##########################
    # Consider adding partial cleaning here, currently the speed is slow so it isnt worth it
    ##########################

    with open(file_path.replace('matched', f'clean_matched'), 'w') as f:
        json.dump(new_data, f, indent=4)

# get the speakers befor a certain time
def get_speakers_before_time(data, time_seconds=3600):
    # get the speakers before a certain time
    i = 0
    speakers = set()
    while i < len(data) and data[i]['timestamp'][0] < time_seconds:
        # add the speakers to the lsit
        for speaker in data[i]['speaker_id']:
            speakers.add(speaker)
        i += 1
    return speakers

def remove_punctuation(data):
    # remove punctuation from the data
    for i, text in tqdm(enumerate(data), total=len(data), ncols=100, desc= 'Removing Punctuation'):
        # fix double '..'
        text = text.replace('.. ', '. ')
        # temporarily remove ellipsis
        text = text.replace('...', '--')
        # remove the other punctuation
        text = text.replace(', ',' ').replace('. ', ' ').replace('! ', ' ').replace('? ', ' ')
        
        # remove the last character if it is a punctuation
        if len(text) > 0 and text[-1] in ['.', '!', '?', ',']:
            text = text[:-1]
        
        # add back the ellipsis
        text = text.replace('--', '... ')
        # replace any multi spaces with single
        text = re.sub(r'\s+', ' ',text)
        # replace the data with the new text
        data[i] = text
    return data

def prep_data(data, is_streamer, remove_punc, restore_punc, lower, max_len):
    model = PunctuationModel()
    # remove the punctuation
    if remove_punc:
        data = remove_punctuation(data)
    else: # just fix common punctuation errors
        for i, text in tqdm(enumerate(data), total=len(data), ncols=100, desc= 'Restoring Punctuation'):
            text = text.replace('.. ', '. ')
            text = text.replace('...', '... ')
            ######################################
            # Change this to run on dataset for better efficiency
            ######################################
            if restore_punc and text != '':
                try:
                    text = model.restore_punctuation(text)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Text: {text}")
            data[i] = text
    if is_streamer:
        data = [remove_specific_sequences(item) for item in data if item != '']
    # convert to lower
    if lower:
        data = [item.lower() for item in data]
    # format the data for the model, remove empty sequences, and sequences that are too long
    data = [{'text': item.strip()} for item in data if len(item.strip()) <= max_len and item.strip() != '']
    return data

def remove_specific_sequences(text):
    sents = sent_tokenize(text)
    # remove thank you for the donation messages
    for i, sent in enumerate(sents):
        if len(re.findall(r'thank you, .*? for', sent, re.IGNORECASE)) > 0:
            sents[i] = ''
        elif len(re.findall(r'thank you for .*?', sent, re.IGNORECASE)) > 0:
            sents[i] = ''
        elif len(re.findall(r'thank you very much', sent, re.IGNORECASE)) > 0:
            sents[i] = ''
    return ' '.join([sent for sent in sents if sent != ''])