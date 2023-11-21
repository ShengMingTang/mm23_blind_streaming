'''
move directoy with only *.json files to pureExpDirs/
'''

from pathlib import Path
import shutil
from Exp_Gen import *

LEN = STATIC_VIDEO_LEN

dst = Path('pureExpDirs')
dst.mkdir(exist_ok=True)
pattern = f'Trace_FPS{FPS}_LEN{LEN}_s*'
filtered = []
for dir in list(Path('.').glob(pattern)):
    if (dir/'qual').is_dir() == False and (dir/'video').is_dir() == False:
        # print(dir)
        # print(dir.parts[-1])
        filtered.append(dir)
        shutil.move(dir, dst/dir.parts[-1])
print(len(filtered))