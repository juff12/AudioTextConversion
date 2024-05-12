import os
import json
from utils.file_processor import TextCleaner
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
from tqdm import tqdm

class ASRDiarization():
    def __init__(self, asr_pipeline, diarization_pipeline):
        self.asr_pipeline = asr_pipeline
        self.diarization_pipeline = diarization_pipeline
        self.cleaner = TextCleaner()
    
    def process_audio(self, audio_file, save_dir=None, save_id=None, save_asr=False, save_diarization=False):
        # process the audio using the asr pipeline
        asr = self.asr_pipeline(audio_file)
        # save the asr data if save_asr is true and the save_dir is not None
        if save_asr and save_dir is not None and save_id is not None:
            with open(os.path.join(save_dir, f"audio_text_{save_id}.json"), 'w') as file:
                json.dump(asr, file, indent=4)
        
        # process the audio using the diarization pipeline, progress hook for visual aid
        with ProgressHook() as hook:
            # load with pytorch for faster processing
            waveform, sample_rate = torchaudio.load(audio_file)
            diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
        
        # save the diarization data if save_diarization is true and the save_dir is not None
        if save_diarization and save_dir is not None and save_id is not None:
            with open(os.path.join(save_dir, f"diarization_{save_id}.rttm"), 'w') as file:
                diarization.write_rttm(file)
        diarization = self.rttm_array(diarization)
        diarization = self.read_rttm(diarization)
        # match the processed audio and the diarization
        matched_text = self.match_speakers(asr['chunks'], diarization)
        return matched_text
    
    def process_from_files(self, audio_file, diarization_file):
        # open the diarization and audio files
        with open(audio_file, 'r') as file:
            audio_text = json.load(file)
        with open(diarization_file, 'r') as file:
            diarization = file.readlines()
            diarization = self.read_rttm(diarization)
        
        # match the processed audio and the diarization
        matched_text = self.match_speakers(audio_text['chunks'], diarization)

        # return the matched data
        return matched_text

    def rttm_array(self, diarization):
        array = []
        uri = diarization.uri if diarization.uri else "<NA>"
        for segment, _, label in diarization.itertracks(yield_label=True):
            # line that is in rttm file
            line = (
                f"SPEAKER {uri} 1 {segment.start:.3f} {segment.duration:.3f} "
                f"<NA> <NA> {label} <NA> <NA>\n"
            )
            array.append(line)
        return array
    
    def read_rttm(self, lines):
        diarization = {}
        for line in lines:
            line = line.split(' ')
            # speaker id
            speaker_id = line[7]
            # start time
            start_time = float(line[3])
            # end time, is start + duration
            end_time = start_time + float(line[4])
            # if the id is not in the list, add it
            if speaker_id not in diarization:
                diarization[speaker_id] = []
            diarization[speaker_id].append((start_time, end_time))
        return diarization

    def match_speakers(self, audio_text, speaker_times):
        # match the diarization and the audio text
        # go through all the speak times
        for speaker_id, times in tqdm(speaker_times.items(), total=len(speaker_times), 
                                      ncols=100, desc= 'Matching Speakers'):
            # go though each audio item
            for i in range(len(audio_text)):
                # check each time in the audio text
                for start_time, end_time in times:
                    if self.overlapp(start_time, end_time, audio_text[i]['timestamp']):
                        # create a speaker id key in the dictionary
                        if 'speaker_id' not in audio_text[i]:
                            audio_text[i]['speaker_id'] = [speaker_id]
                            break
                        # add the speaker id to the list
                        elif speaker_id not in audio_text[i]['speaker_id']:
                            audio_text[i]['speaker_id'].append(speaker_id)
                            break
        # fix missing speaker ids
        i = 0
        while i < len(audio_text):
            if i == 0 and 'speaker_id' not in audio_text[i]:
                audio_text[i]['speaker_id'] = ['unknown']
            elif 'speaker_id' not in audio_text[i]:
                audio_text[i]['speaker_id'] = audio_text[i-1]['speaker_id']
            i += 1
        return audio_text

    def overlapp(self, start_time, end_time, timestamp):
        # both stamps are invalid, false
        if timestamp[0] is None and timestamp[1] is None:
            return False
        # the first time stamp is invalid, but last is valid
        # assume true, else false if not satisfy condition
        elif timestamp[0] is None and timestamp[1] is not None:
            if start_time > timestamp[1]:
                return False
            return True
        # the first time stamp is vallid, but last is invalid
        # assume true, else false if not satisfy condition
        elif timestamp[0] is not None and timestamp[1] is None:
            if end_time < timestamp[0]:
                return False
            return True
        # check if the start time or end time is inside of the speaker time
        # checks if the speaker could be the detected speaker
        if end_time < timestamp[0] or start_time > timestamp[1]:
            return False
        return True
    
    def merge_speakers(self, matched_text):
        last_speakers = None
        new_text = []
        for i in range(len(matched_text)):
            # if the speaker id is not present, add the text to the last speaker
            if 'speaker_id' not in matched_text[i]:
                # if this is the first element, drop it
                if len(new_text) == 0:
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
                if len(new_text) == 0:
                    last_speakers = matched_text[i]['speaker_id']
                    new_text.append(matched_text[i])
                    continue
                new_text[-1]['text'] += ' ' + matched_text[i]['text']
            else:
                last_speakers = matched_text[i]['speaker_id']
                new_text.append(matched_text[i])
        return new_text