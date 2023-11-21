#%%
'''
Collect encoded data to statistic files
'''
#%%
from Exp import *
from Exp_Gen import *
W, H = 960, 540
FPS = 50
LEN = STATIC_VIDEO_LEN
exp = Exp()
#%%
exp.encodedToSummary(Path(f'Trace_FPS{FPS}_LEN{LEN}'), Path('encode_merge.csv'))
#%%
exp.encodedSeparateToSummary(Path(f'Trace_FPS{FPS}_LEN{LEN}'), Path('encode_merge_separate.csv'))