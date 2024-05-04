from pyannote.audio import Pipeline
import argparse
import os
from tqdm import tqdm
import torch
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--reprocess_old', type=bool, default=False, help='reprocess old files')
    return parser.parse_args()

def get_audio_files(directory, audio_file_endings):
    audio_files = []
    for file in os.listdir(directory):
        for ending in audio_file_endings:
            if file.endswith(ending):
                audio_files.append(os.path.join(directory, file))
    if len(audio_files) == 0:
        return None
    return audio_files[0] # return the first audio file

def main():
    opt = args()

    # get access token
    with open('api_keys/hugging_face_token.txt', 'r') as file:
        token = file.read()
    
    # create pipeline for speaker diarization
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=token
    ).to(torch.device('cuda:0'))

    directory = opt.dir

    sub_dirs = os.listdir(directory)
    # audio file endings
    audio_file_endings = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']
    for dir in tqdm(sub_dirs):
        # skip already processed files
        output_file = os.path.join(directory, f"{dir}/diarization_{dir}.rttm")
        if opt.reprocess_old is False and os.path.exists(output_file):
            print(f"Skipping {dir}")
            continue
        try:
            # audio file to process
            audio_file = get_audio_files(os.path.join(directory, dir), audio_file_endings)
            waveform, sample_rate = torchaudio.load(audio_file)
            # create a pipeline for audio to text and diarization
            with ProgressHook() as hook:        
                diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
            print(type(diarization))
            
            with open(output_file, 'w') as file:
                diarization.write_rttm(file)
        except:
            print('Error')
            continue # if an error occurs while processing, continue processing

if __name__=='__main__':
    main()