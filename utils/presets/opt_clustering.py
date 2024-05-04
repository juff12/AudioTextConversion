import argparse

parser = argparse.ArgumentParser()

# text matching params
parser.add_argument("--dir", type=str, default="data/streamers/", help="path to data folder")
parser.add_argument("--threshhold", type=float, default=0.3, help="The the threshhold of % match for semantics")
parser.add_argument("--skip_complete", type=bool, default=False, help="skip the file has been processed")
opt = parser.parse_args()