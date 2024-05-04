import argparse

parser = argparse.ArgumentParser()

# text matching params
parser.add_argument("--dir", type=str, default="data/streamers/", help="path to data folder")
parser.add_argument("--create_subdir", type=bool, default=False, help="should the program create subdirectories")
parser.add_argument("--reprocess_old", type=bool, default=False, help="should the program reprocess old files")
opt = parser.parse_args()