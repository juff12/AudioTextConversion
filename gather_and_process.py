import subprocess
import argparse
from utils import FilePreProcessing, ASRDiarization, TopicClustering, MessageMatcher, TextCleaner
from pyannote.audio import Pipeline
import en_core_web_lg
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def args():
    parser = argparse.ArgumentParser()
    # arguments for gathering data
    parser.add_argument('--channel_url', type=str, default='', help='The url of the youtube channel to download the videos from')
    parser.add_argument('--min_dur', type=str, default=1800, help='the minimum length of video to download')
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='the parent directory to save to')
    
    # arguments for asr and diarization
    parser.add_argument('--api_key_loc', type=str, default='api_keys/hugging_face_token.txt', help='location of the api key to access the hugging face models')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-large-v3', help='name of the asr model')
    parser.add_argument('--diarization_model', type=str, default='pyannote/speaker-diarization-3.0', help='name of the diarization model')
    parser.add_argument('--load_files', type=bool, default=False, help='load the files instead of processing the audio')
    parser.add_argument('--save_asr', type=bool, default=True, help='save the asr files as separate json files')
    parser.add_argument('--save_diarization', type=bool, default=True, help='save the diarization files as separate rttm files')
    
    # arguments for matching the text
    parser.add_argument('--twc', type=bool, default=False, help='use twc data for matching') # mark true if twc is available and want matching
    parser.add_argument('--sent_model', type=str, default='multi-qa-MiniLM-L6-cos-v1', help='name of the sentence transformer semantic model')
    parser.add_argument('--sim_cutoff', type=float, default=0.7, help='similarity cutoff for message matching')
    parser.add_argument('--history_limit', type=int, default=40, help='maximum number of previous messages to consider for matching')
    parser.add_argument('--match_thresh', type=float, default=0.55, help='threshold for message matching')
    parser.add_argument('--forward_chunks', type=int, default=0, help='number of forward chunks to consider for matching')
    parser.add_argument('--backward_chunks', type=int, default=10, help='number of backward chunks to consider for matching')
    parser.add_argument('--delay', type=int, default=10, help='delay in seconds for message matching')
    parser.add_argument('--multi_match', type=bool, default=False, help='allow multiple matches for a single message')

    # arguments for cleaning
    parser.add_argument('--time_seconds', type=int, default=3600, help='time in seconds to consider for getting the speakers')

    # arguments for topic clustering
    parser.add_argument('--cluster_thresh_1', type=float, default=0.3, help='threshold 1 for clustering')
    parser.add_argument('--cluster_thresh_2', type=float, default=0.6, help='threshold 2 for clustering')
    parser.add_argument('--from_text', type=bool, default=True, help='use txt file instead of json')
    parser.add_argument('--save_type', type=str, default='txt', help='type of file to save as [json/txt]')
    
    return parser.parse_args()


def gather_data(channel_url, min_length, dir):
    # name of the conda environment
    conda_env = 'youtube-env'

    # command to run 
    command = f'yt-dlp --match-filter="duration>{min_length}" -ciw -o "{dir}/%(id)s.%(ext)s" -x --audio-format wav --audio-quality 0 --restrict-filenames {channel_url}'

    activate_cmd = f'conda activate {conda_env} && {command}'

    # Run the command to gather data
    subprocess.run(activate_cmd, shell=True)


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

def run_asrdiarization(opt):
    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # select the model to use
    model_id = opt.asr_model

    # create the asr pipeline
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"}
    )

    # get access token
    with open(opt.api_key_loc, 'r') as file:
        token = file.read()
    
    # create pipeline for speaker diarization
    diarization_pipeline = Pipeline.from_pretrained(
        opt.diarization_model,
        use_auth_token=token
    ).to(device)


    # set the main directory to process
    dir = opt.dir
    
    # create the message matcher
    matcher = ASRDiarization(asr_pipeline=asr_pipeline, diarization_pipeline=diarization_pipeline)

    # audio file endings
    endings = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']

    # get the sub directories
    sub_dirs = os.listdir(dir)

    for sub in tqdm(sub_dirs):
        try:
            # audio directory
            audio_dir = os.path.join(dir, sub)
            audio_file = get_audio_files(audio_dir, endings)
            
            if opt.load_files:
                audio_file = os.path.join(audio_dir, f'audio_text_{sub}.json')
                diarization_file = os.path.join(audio_dir, f'diarization_{sub}.rttm')
                # from files matching the diarization and asr
                matched_text = matcher.process_from_files(audio_file, diarization_file)
            else:
                # match the diarization and asr
                matched_text = matcher.process_audio(audio_file, save_dir=audio_dir,
                                                    save_id=sub, save_asr=opt.save_asr,
                                                    save_diarization=opt.save_diarization)
            # save the pairs to a json file
            file_path = os.path.join(audio_dir, f"matched_{sub}.json")
            matcher.save_json(matched_text, file_path)
        except Exception as e:
            print(e)
            continue

def run_topic_clustering(opt):
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
            cluster_topics, final_texts = clusterer.cluster(audio_text,
                                                            thresh_1=opt.cluster_thresh_1,
                                                            thresh_2=opt.cluster_thresh_2)
            
            # save the clusters based on type of save given
            if opt.save_type == 'txt':
                clusterer.save_txt(final_texts, os.path.join(dir, f"{id}/clusters_{id}.txt"))
            elif opt.save_type == 'json':
                # save to json
                clusterer.save_json(cluster_topics, final_texts, os.path.join(dir, f"{id}/clusters_{id}.json"))
        except Exception as e:
            print(e)
            continue

def run_message_matching(opt):
    # set the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # select the model to use
    model = SentenceTransformer(opt.sent_model).to(device)
    # set the main directory to process
    dir = opt.dir
    
    # create the message matcher
    matcher = MessageMatcher(model, sim_cutoff=opt.sim_cutoff, history_limit=opt.history_limit,
                             threshold=opt.match_thresh, forward_chunks=opt.forward_chunks,
                             backward_chunks=opt.backward_chunks, delay=opt.delay, multi_match=opt.multi_match)

    # get the sub directories
    sub_dirs = os.listdir(dir)

    for sub in tqdm(sub_dirs):
        chat = pd.read_csv(os.path.join(dir,f"{sub}/{sub}.csv"))
        # convert the time column to numeric
        chat[['time']] = chat[['time']].apply(pd.to_numeric)       

        # open the audio text file
        with open(os.path.join(dir,f"{sub}/audio_text_{sub}.json")) as file:
            audio_text = json.load(file)
        
        # match the messages
        pairs = matcher.match(audio_text, chat)
        # save the pairs to a json file
        file_path = os.path.join(dir, f"{sub}/pairs_{sub}_{opt.match_thresh}_formatted.json")
        matcher.save_json(pairs, file_path)

def clean_matched_pairs(cleaner, file_path, time_seconds=3600):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # get the speakers before a certain time
    speakers = get_speakers_before_time(data, time_seconds)
    
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
    while data[i]['timestamp'][0] < time_seconds:
        # add the speakers to the lsit
        for speaker in data[i]['speaker_id']:
            if speaker not in speakers:
                speakers.add(speaker)
        i += 1
    return speakers

def run_cleaning(opt):
    parent_dir = opt.dir
    # get the sub directories
    sub_dirs = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    # get the unprocessed matched json files
    files = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir) if f.endswith('.json') and 'matched' in f and 'clean' not in f]    

    # create the text cleaner
    cleaner = TextCleaner()

    # clean the json files
    for file in tqdm(files):
        clean_matched_pairs(cleaner, file, opt.time_seconds)

    # get the cleaned json files directory
    files = [os.path.join(sub_dir, f) for sub_dir in sub_dirs for f in os.listdir(sub_dir) if f.endswith('.json') and 'clean_matched' in f]
    for f in tqdm(files):
        with open(f, 'r') as file:
            data = json.load(file)
        text = ' '.join([item['text'] for item in data])
        # clean the text
        cleaned_text = cleaner.clean_text(text)
        # save the cleaned text as a txt file
        with open(f.replace('clean_matched', 'clean_text').replace('.json', '.txt'), 'w') as file:
            file.write(cleaned_text)


def main():
    opt = args()
    
    # get the data from youtube
    #gather_data(opt.channel_url, opt.min_dur, opt.dir)

    # initialize the file preprocessor
    preprocessor = FilePreProcessing(opt.dir, is_yt=False, has_twc=False)

    # create subdirectories for the data and movde the videos
    preprocessor.prepare_files()

    # process the audio files
    #run_asrdiarization(opt)

    # clean the files
    #run_cleaning(opt)

    # run the topic clustering
    #run_topic_clustering(opt)

    # match
    if opt.twc:
        run_message_matching(opt)

if __name__=="__main__":
    main()