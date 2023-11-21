from Exp import *
from pathlib import Path
import sys
from Exp_Gen import *

VIDEO_LEN = STATIC_VIDEO_LEN # secs
POSS_WORKS = ['exp', 'synthesize', 'qual']

if __name__ == '__main__':
    root = Path('.')
    WORKS = sys.argv[1:]
    IXR_SOLVERS = ['C2G', 'C2I']
    TMM_SOLVERS = ['BB', 'Uni', 'UnB', 'Opt']
    print(f'IXR solvers = {IXR_SOLVERS}')
    
    expName = input('input filename prefix\n')
    
    for work in WORKS:
        assert work in POSS_WORKS
    fnCount = 0
    for scene in ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']:
        with open(f'{expName}_{scene}.{ext}', 'w') as f:
            poseDir = root/generatePoseDirName(FPS, VIDEO_LEN, scene)
            # cdd=TMM,solver=IXR (2)
            if GenComment:
                f.write('@REM cdd=TMM,solver=IXR\n')
            for solverPolicy in IXR_SOLVERS:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['solverPolicy'] = solverPolicy
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # cdd=IXR,solver=TMM (4)
            if GenComment:
                f.write('@REM cdd=IXR,solver=TMM\n')
            for solverPolicy in TMM_SOLVERS:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                
                # cdd
                param['psPolicy'] = 'IXR'
                param['placePolicy'] = 'IXR'
                param['m'] = 0.01
                
                param['solverPolicy'] = solverPolicy
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # cdd=IXR,solver=IXR (2)
            if GenComment:
                f.write('@REM cdd=IXR,solver=IXR\n')
            for solverPolicy in IXR_SOLVERS:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir

                # cdd
                param['psPolicy'] = 'IXR'
                param['placePolicy'] = 'IXR'
                param['m'] = 0.01
                
                param['solverPolicy'] = solverPolicy
                                
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
        os.system(f'type {expName}_{scene}.{ext} >> {expName}.{ext}')