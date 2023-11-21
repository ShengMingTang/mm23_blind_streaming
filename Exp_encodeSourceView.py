#%%
'''
Encode source views to *.264
'''
#%%
from Exp import *
from Exp_Gen import *
W, H = 960, 540
FPS = 50
LEN = STATIC_VIDEO_LEN
exp = Exp()
#%%
for scene in ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']:
    poseDir = Path(f'Trace_FPS{FPS}_LEN{LEN}')/scene
    exp.encodeTargetView(poseDir, W, H, FPS)
#%%
for scene in ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']:
    poseDir = Path(f'Trace_FPS{FPS}_LEN{LEN}')/scene
    exp.encodeSourceView(poseDir, W, H, 4)
#%%
for scene in ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']:
    poseDir = Path(f'Trace_FPS{FPS}_LEN{LEN}')/scene
    exp.encodeSourceViewSeparate(poseDir, W, H, 4)