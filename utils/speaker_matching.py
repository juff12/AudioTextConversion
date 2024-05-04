import json
import os
from presets.opt_audio_to_text import opt


def read_rttm(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    diarization = {}
    for line in lines:
        line = line.split(' ')
        speaker_id = line[7]
        start_time = float(line[3])
        end_time = start_time + float(line[4])
        if speaker_id not in diarization:
            diarization[speaker_id] = []
        diarization[speaker_id].append((start_time, end_time))
    return diarization

def overlapp(start_time, end_time, timestamp):
    if end_time < timestamp[0] or start_time > timestamp[1]:
        return False
    return True

def match_speakers(audio_text, speaker_times):
    for speaker_id, times in speaker_times.items():
        for i in range(len(audio_text)):
            for start_time, end_time in times:
                if overlapp(start_time, end_time, audio_text[i]['timestamp']):
                    if 'speaker_id' not in audio_text[i]:
                        audio_text[i]['speaker_id'] = [speaker_id]
                        break
                    elif speaker_id not in audio_text[i]['speaker_id']:
                        audio_text[i]['speaker_id'].append(speaker_id)
                        break
    return audio_text

def merge_speakers(matched_text):
    last_speakers = None
    new_text = []
    for i in range(len(matched_text)):
        # if the speaker id is not present, add the text to the last speaker
        if 'speaker_id' not in matched_text[i]:
            # if this is the first element, drop it
            if len(new_text) < 0:
                continue
            new_text[-1]['text'] += ' ' + matched_text[i]['text']
            continue
        # start of the loop
        if last_speakers is None:
            last_speakers = matched_text[i]['speaker_id']
            new_text.append(matched_text[i])
            continue
        # if the speakers are the same
        if last_speakers == matched_text[i]['speaker_id']:
            new_text[-1]['text'] += ' ' + matched_text[i]['text']
        else:
            last_speakers = matched_text[i]['speaker_id']
            new_text.append(matched_text[i])
    return new_text

def main():
    directory = opt.dir
    sub_dirs = os.listdir(directory)
    for dir in sub_dirs:
        with open(os.path.join(directory, f"{dir}/audio_text_{dir}.json"), 'r') as file:
            audio_text = json.load(file)

        speaker_times = read_rttm(os.path.join(directory, f"{dir}/diarization_{dir}.rttm"))
        matched_text = match_speakers(audio_text['chunks'], speaker_times)
        matched_text = merge_speakers(matched_text)
        with open(os.path.join(directory, f"{dir}/matched_speaker_{dir}.json"), 'w') as file:
            json.dump(matched_text, file, indent=4)
        break
if __name__ == '__main__':
    main()