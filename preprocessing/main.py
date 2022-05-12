import os
import time
import pandas as pd

from detect_lines import *

def fetch_files(path):
    """
    Fetch all files in a directory
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-4:] == ".png":
                files.append(os.path.join(r, file))
    return files

if __name__ == '__main__':
    files = fetch_files("./data")
    
    index = 0
    results = []
    for file in files:
        if index > 50:
            break
        index += 1
        lines = detect_lines(file)
        # results.append({"file": file, "lines": lines})
        # if index > 0 and index % 10 == 0:
        #     print(f"Processed {index} files out of {len(files)}")
        # index += 1

    # f = files[0]
    # lines = detect_lines(f)
