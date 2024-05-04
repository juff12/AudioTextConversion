import numpy as np
import spacy
import json
import os
import en_core_web_sm
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import emoji
from openai import OpenAI
import re
from pathlib import Path
import argparse


def clean_text(text):
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

def remove_repeat(text):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    tokens = tokenizer.tokenize(text)  # Tokenize the text
    cleaned_tokens = [tokens[0]]  # Initialize list with first token

    # Iterate through the tokens, skipping repetitions
    for i in range(1, len(tokens)):
        # Check if the current token is the same as the previous one
        if tokens[i] != tokens[i - 1]:
            cleaned_tokens.append(tokens[i])  # If not the same, add to cleaned list
    cleaned_text = detokenizer.detokenize(cleaned_tokens)  # Detokenize the cleaned tokens
    return cleaned_text  # Return the cleaned tokens array

def remove_emojis(text):
    # Remove emojis from text
    cleaned_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
    return cleaned_text

def remove_unicode(text):
    # Remove Unicode characters using regex
    return re.sub(r'[\u0080-\uffff]', '', text)

def token_chunking(text):
    sents = sent_tokenize(text)
    tokenized_chunks = []
    i = 0
    chunk = []
    chunk_len = 0
    while i < len(sents):
        words = sents[i].split(' ')
        if chunk_len + len(words) < 1000:
            chunk.append(sents[i])
            chunk_len += len(words)
            i += 1
        else:
            text_chunk = ' '.join(chunk)
            text_chunk = text_chunk.replace(' .', '.').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
            tokenized_chunks.append(text_chunk)
            chunk = []
            chunk_len = 0
    return tokenized_chunks

def format_block(chunk):
    text_block = ' '.join(chunk)
    text_block = text_block.replace(' .', '.').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
    return text_block


def character_chunking(text, size=512):
    sents = sent_tokenize(text)
    i = 0
    chunk_len = 0
    chunk = []
    blocks = []
    while i < len(sents):
        if chunk_len == 0 and len(sents[i]) > size:
            text_block  = ''
            for j in range(size):
                text_block += sents[i][j]
            chunk = []
            chunk_len = 0
            blocks.append(text_block)
            i += 1
            continue        
        if chunk_len + len(sents[i]) <= size:
            chunk_len += len(sents[i])
            chunk.append(sents[i])
            i += 1
            if i >= len(sents):
                text_block = format_block(chunk)
                blocks.append(text_block)
                chunk = []
                chunk_len = 0
        else:
            text_block = format_block(chunk)
            blocks.append(text_block)
            chunk = []
            chunk_len = 0
    return blocks

def run_text_cleaning(streamer):
    files = os.listdir('data/streamers/{s}'.format(s=streamer))
    nlp = en_core_web_sm.load()
    output = []
    for id in tqdm(files):
        with open('data/streamers/{s}/{f}/audio_text_{f}.json'.format(f=id, s=streamer), encoding='utf-8') as file:
            audio_text = json.load(file)
        audio_text = audio_text['text']
        # clean text
        audio_text = clean_text(audio_text)
        # remove repeated words and phrases
        audio_text = remove_repeat(audio_text)


        output.append(audio_text)
    out_text = ' '.join(output)
    out_text = out_text.replace(' .', '.').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?').replace(' !', '!')
    out_text = remove_emojis(out_text) # remove emojis
    out_text = remove_unicode(out_text) # remove unicode characters

    # save to text file
    Path('data/datasets/{s}'.format(s=streamer)).mkdir(parents=True, exist_ok=True)
    with open('data/datasets/{s}/{s}.txt'.format(s=streamer), 'w') as f:
        f.write(out_text)

def run_character_chunking(streamer):
    data = ''
    with open('data/datasets/{s}/{s}.txt'.format(s=streamer), 'r') as f:
        data = f.read()
    chunked = character_chunking(data)
    dataset = []
    for chunk in chunked:
        dataset.append({'text': chunk})
    Path('data/datasets/{s}'.format(s=streamer)).mkdir(parents=True, exist_ok=True)
    with open('data/datasets/{s}/{s}.json'.format(s=streamer), 'w') as f:
        json.dump(dataset, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streamer', type=str, default='', help='streamer name')
    opt = argparse.parse_args()
    streamer = opt.streamer
    run_text_cleaning(streamer)
    #run_openai(streamer)
    run_character_chunking(streamer)
    
if __name__ == "__main__":
    main()