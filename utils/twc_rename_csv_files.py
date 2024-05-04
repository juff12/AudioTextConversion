import os
import json
import re

dir = 'data/streamers/'
for item in os.listdir(dir):
    if os.path.isdir(item):
        continue
    if item.endswith('.csv'):
        os.rename(os.path.join(dir,item), os.path.join(dir, item.replace('twitch-chat-', '')))