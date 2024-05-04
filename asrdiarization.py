from utils import ASRDiarization
from pyannote.audio import Pipeline
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--api_key_loc', type=str, default='api_keys/hugging_face_token.txt', help='location of the api key to access the hugging face models')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-large-v3', help='name of the asr model')
    parser.add_argument('--diarization_model', type=str, default='pyannote/speaker-diarization-3.0', help='name of the diarization model')
    parser.add_argument('--load_files', type=bool, default=True, help='load the files instead of processing the audio')
    parser.add_argument('--save_asr', type=bool, default=True, help='save the asr files as separate json files')
    parser.add_argument('--save_diarization', type=bool, default=True, help='save the diarization files as separate rttm files')
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
    # get hyperparameters
    opt = args()

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

if __name__ == "__main__":
    main()