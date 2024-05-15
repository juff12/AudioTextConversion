import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import json
from tqdm import tqdm
import argparse
from utils.functions import get_audio_files

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data/streamers/test', help='directory of files to be processed')
    parser.add_argument('--reprocess_old', type=bool, default=True, help='reprocess old files')
    return parser.parse_args()

def main():
    opt = args()
    
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
    
    for dir in tqdm(sub_dirs):
        # skip already processed files
        if opt.reprocess_old is False and os.path.exists(os.path.join(directory,f"{dir}/audio_text_{dir}.json")):
            continue
        try:
            audio_file = get_audio_files(os.path.join(directory, dir), audio_file_endings)
            result = pipe(audio_file, generate_kwargs={"language": "english"})
            with open(os.path.join(directory,f"{dir}/audio_text_{dir}.json"), "w") as f:
                json.dump(result, f, indent=4)
        except:
            print('Error')
            continue # if an error occurs while processing, continue processing
if __name__=='__main__':
    main()