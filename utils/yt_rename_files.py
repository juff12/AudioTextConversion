import os
import re

def rename_file(file, extensions):
    # find the ending file extension
    for ext in extensions:
        if file.endswith(ext):
            # rename the file
            result = re.findall(r'\[([^]]+)\]'+ext, file)
            if result == None or len(result) == 0:
                return None
            return result[0] + ext

def main():
    # audio file endings
    audio_file_endings = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma', '.m4a']
    dir = 'data/streamers/'
    parents = [x for x in os.listdir(dir) if os.path.isdir(dir)]
    for parent in parents:
        parent = os.path.join(dir, parent)
        for file in os.listdir(parent):
            # rename the files
            new_name = rename_file(file, audio_file_endings)
            if new_name is None:
                continue
            try:
                os.rename(os.path.join(parent,file), os.path.join(parent, new_name))
            except:
                pass

if __name__ == "__main__":
    main()