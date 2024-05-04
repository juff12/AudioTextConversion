import argparse

parser = argparse.ArgumentParser()

# text matching params
parser.add_argument("--dir", type=str, default="data/streamers/", help="path to data folder")
parser.add_argument("--threshhold", type=float, default=0.55, help="The the threshhold of % match for semantics")
parser.add_argument("--forward_chunks", type=str, default=0, help="number of preceeding messages from match")
parser.add_argument("--backward_chunks", type=int, default=10, help="number of following messages from match")
parser.add_argument("--history_limit", type=int, default=50, help="number of chat messages in the chat history")
parser.add_argument("--delay", type=int, default=10, help="delay on the chat relative streamer speech")
parser.add_argument("--multi_match", type=bool, default=False, help="should the model match multiple chat messages")
parser.add_argument("--sem_model", type=str, default='multi-qa-MiniLM-L6-cos-v1', help="The semantic model to use for analysis")
parser.add_argument("--skip_complete", type=bool, default=False, help="skip the file has been processed")
opt = parser.parse_args()