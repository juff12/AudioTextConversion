import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import json
from pathlib import Path
from presets.opt_audio_to_text import opt
import shutil
import sys
from tqdm import tqdm

def process_audio(pipe, file, generate_kwargs={"language": "english"}):
    result = pipe(file, generate_kwargs=generate_kwargs)
    return result

def create_and_move(files, subs, parent):
    # error has occured
    if len(files) != len(subs):
        print('Error has occured in creating subdirectories\n')
        print(f'Subdirectories: {subs}\n')
        print(f'Files: {files}\n')
        sys.exit()
    for i, name in enumerate(subs):
        dir = os.path.join(parent, name)
        Path(dir).mkdir(parents=True, exist_ok=True)
        shutil.move(os.path.join(parent, files[i]), os.path.join(dir, files[i]))
    return subs

def get_audio_files(directory, audio_file_endings):
    audio_files = []
    for file in os.listdir(directory):
        for ending in audio_file_endings:
            if file.endswith(ending):
                audio_files.append(os.path.join(directory, file))
    if len(audio_files) == 0:
        return None
    return audio_files[0] # return the first audio file

def remove_audio_file_ending(file_name, audio_file_endings):
    for ending in audio_file_endings:
        if file_name.endswith(ending):
            return file_name.replace(ending, '')
    return None # not a not file

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
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
    )

    directory = opt.dir


    sub_dirs = os.listdir(directory)

    # audio file endings
    audio_file_endings = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']
    
    # function to check that the file is an audio file ending
    check_audio_end = lambda x, endings: any([x.endswith(ending) for ending in endings])
    if opt.create_subdir:
        files = [x for x in os.listdir(directory) if check_audio_end(x, audio_file_endings)]
        csv_files = [x for x in os.listdir(directory) if x.endswith('.csv')]
        sub_dirs = [remove_audio_file_ending(x, audio_file_endings) for x in os.listdir(directory) if check_audio_end(x, audio_file_endings)]
        sub_dirs = [x for x in sub_dirs if x is not None]
        # merge the csv files with the audio files
        files = files + csv_files
        # if folders exit in the directory, start from the largest number
        sub_dirs = create_and_move(files, sub_dirs, directory)

    for dir in tqdm(sub_dirs):
        # skip already processed files
        if opt.reprocess_old is False and os.path.exists(os.path.join(directory,f"{dir}/audio_text_{dir}.json")):
            continue
        try:
            audio_file = get_audio_files(os.path.join(directory, dir), audio_file_endings)
            result = pipe(audio_file, generate_kwargs={"language": "english"})
            json.dump(result, open(os.path.join(directory,f"{dir}/audio_text_{dir}.json"), "w"), indent=4)
        except:
            print('Error')
            continue # if an error occurs while processing, continue processing
if __name__=='__main__':
    main()