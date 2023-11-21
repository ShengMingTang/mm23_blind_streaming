from Exp import *
from pathlib import Path
import sys
from Exp_Gen import *

VIDEO_LEN = STATIC_VIDEO_LEN # secs
POSS_WORKS = ['generateCandidates', 'prepareCandidates', 'prune']

if __name__ == '__main__':
    expName = Path(__file__).stem
    root = Path('.')
    WORKS = sys.argv[1:]
    for work in WORKS:
        assert work in POSS_WORKS
    
    expName = input('input filename prefix\n')
    
    fnCount = 0
    for scene in SCENES:
        with open(f'{expName}_{scene}.{ext}', 'w') as f:
            poseDir = root/generatePoseDirName(FPS, VIDEO_LEN, scene)
            # TMM
            if GenComment:
                f.write('@REM cdd=TMM\n')
            for m in [0.01, 0.02, 0.03, 0.04, 0.05]:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['m'] = m
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # IXR
            if GenComment:
                f.write('@REM cdd=IXR\n')
            for m in [0.01]:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                
                # cdd
                param['psPolicy'] = 'IXR'
                param['placePolicy'] = 'IXR'
                param['m'] = m
                
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            