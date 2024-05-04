import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default="data/streamers/", help="path to data folder")

# file processing and directory creation
parser.add_argument("--create_subdir", type=bool, default=False, help="should the program create subdirectories")
parser.add_argument("--reprocess_old", type=bool, default=False, help="should the program reprocess old files")
parser.add_argument("--rename_yt", type=bool, default=False, help="rename youtube files")


parser.add_argument("--save_raw_audio_text", type=bool, default=False, help="save the raw audio text")
parser.add_argument("--save_raw_diarization", type=bool, default=False, help="save the raw diarization")



parser.add_argument("--audio_to_text", type=bool, default=True, help="convert audio to text")
parser.add_argument("--speaker_diarization", type=bool, default=True, help="diarize the speaker")
parser.add_argument("--cluster_text", type=bool, default=False, help="cluster the text")


opt = parser.parse_args()