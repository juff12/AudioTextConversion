import os
import json
import re

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

    with open(file_path.replace('matched', 'clean_matched'), 'w') as f:
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

def _remove_punc_helper(text):
    # fix double '..'
    text = text.replace('.. ', '. ')
    # temporarily remove ellipsis
    text = text.replace('...', '--')
    # remove the other punctuation
    text = text.replace(', ',' ').replace('. ', ' ').replace('! ', ' ').replace('? ', ' ')
    # add back the ellipsis
    text = text.replace('--', '... ')
    # replace any multi spaces with single
    text = re.sub(r'\s+', ' ',text)
    return text

def remove_punctuation(data, type='txt'):
    # remove punctuation from the data for json
    if type == 'json':
        for item in data:
            # remove the punctuation
            item['text'] = _remove_punc_helper(item['text'])
    elif type == 'txt': # remove punctuation from the data for txt
        for i, item in enumerate(data):
            # remove the punctuation
            data[i] = _remove_punc_helper(item)
    return data