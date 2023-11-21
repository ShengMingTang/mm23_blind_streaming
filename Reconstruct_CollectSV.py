'''
This collect the leakage source view to a directory for reconstruction
'''

from pathlib import Path
import numpy as np
import json
from Exp import *
from Exp_Gen import *
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dynamic', type=bool)
args = parser.parse_args()
isDynamic = args.dynamic

if isDynamic:
    VIDEO_LEN = DYNAMIC_VIDEO_LEN # secs
    root = Path(f'Trace_FPS{FPS}_LEN{VIDEO_LEN}')
    outputRoot = Path('Reconstruction')/'Dynamic'
    outputRoot.mkdir(exist_ok=True)
    DEFAULT_PARAM['m'] = 0.01
    DEFAULT_PARAM['h'] = 0.05
else:
    VIDEO_LEN = STATIC_VIDEO_LEN # secs
    root = Path(f'Trace_FPS{FPS}_LEN{VIDEO_LEN}')
    outputRoot = Path('Reconstruction')/'Static'
    outputRoot.mkdir(exist_ok=True)

for scene in SCENES:
    poseDir = root/generatePoseDirName(FPS, VIDEO_LEN, scene)
    param = dict(DEFAULT_PARAM)
    param['poseDir'] = poseDir
    expDir = root/generateExpDirName(
        FPS, VIDEO_LEN, scene, 
        **param
    )
    param['expDir'] = expDir
    print(str(param['expDir']))
    with open(expDir/'Exp.json', 'r') as f:
        expJ = json.load(f)
        sols = np.array(expJ['sols'])
        print(sols.shape)
    
    outDir = outputRoot/scene/generateCandidateDirName(**param)/'images'
    outDir.mkdir(exist_ok=True, parents=True)
    cddDir = root/scene/generateCandidateDirName(**param)
    
    if isDynamic == False:
        for i in range(sols.shape[0]):
            print(f'scene: {scene}, group: {i}')
            nz = np.nonzero(sols[i])[0]
            for j in nz:
                fr = cddDir/f'rgb_sv{i}_{j}.png'
                to = outDir/fr.name
                shutil.copyfile(fr, to)
    else:
        for i in range(sols.shape[0]):
            print(f'scene: {scene}, group: {i}')
            nz = np.nonzero(sols[i])[0]
            for j in nz:
                for k in range(FPS):
                    fr = cddDir/f'rgb_sv{j}_{FPS*i + k}.png'
                    to = outDir/fr.name
                    shutil.copyfile(fr, to)
    
    