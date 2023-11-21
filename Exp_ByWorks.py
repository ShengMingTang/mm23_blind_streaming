from Exp import *
from pathlib import Path
import sys
from Exp_Gen import *

VIDEO_LEN = STATIC_VIDEO_LEN # secs
POSS_WORKS = ['exp', 'synthesize', 'qual']


if __name__ == '__main__':
    root = Path('.')
    WORKS = sys.argv[1:]
    TMM_SOLVERS = ['BB', 'Uni', 'UnB', 'Opt']
    solver = input(f"input default solver in {TMM_SOLVERS}\n")
    DEFAULT_PARAM["solverPolicy"] = solver
    print(f'default solver = {DEFAULT_PARAM["solverPolicy"]}')
    assert solver in TMM_SOLVERS
    TMM_SOLVERS.remove(solver)
    print(f'solvers = {TMM_SOLVERS}')
    
    expName = input('input filename prefix\n')
    
    for work in WORKS:
        assert work in POSS_WORKS
    fnCount = 0
    for scene in ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']:
        with open(f'{expName}_{scene}.{ext}', 'w') as f:
            poseDir = root/generatePoseDirName(FPS, VIDEO_LEN, scene)
            # default
            if GenComment:
                f.write('@REM Default\n')
            param = dict(DEFAULT_PARAM)
            param['poseDir'] = poseDir
            expDir = generateExpDirName(
                FPS, VIDEO_LEN, scene, 
                **param
            )
            param['expDir'] = expDir
            f.write(f'mkdir {expDir}\n')
            for work in WORKS:
                param['work'] = work
                f.write(genCmd(**param))
        
            # Quality (5)
            if GenComment:
                f.write('@REM Quality\n')
            # pure m variant
            for m in [0.01, 0.02, 0.04, 0.05]:
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
            
            # Compute (3)
            if GenComment:
                f.write('@REM Compute\n')
            for maxNumNodes in [48, 192]:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['maxNumNodes'] = maxNumNodes
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # Solver (4)
            if GenComment:
                f.write('@REM Solver\n')
            for solverPolicy in TMM_SOLVERS:
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
            # FFR (1)
            if GenComment:
                f.write('@REM FFR\n')
            for ffrMask in ['On']:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['ffrMask'] = ffrMask
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # Run time depth res (2)
            if GenComment:
                f.write('@REM Runtime Depth\n')
            for runTimeResDs in [5, 10]:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['runTimeResDs'] = runTimeResDs
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # a (2)
            if GenComment:
                f.write('@REM a\n')
            for a in [1e-4, 1e-6]:
                param = dict(DEFAULT_PARAM)
                param['poseDir'] = poseDir
                param['a'] = a
                expDir = generateExpDirName(
                    FPS, VIDEO_LEN, scene, 
                    **param
                )
                param['expDir'] = expDir
                f.write(f'mkdir {expDir}\n')
                for work in WORKS:
                    param['work'] = work
                    f.write(genCmd(**param))
            # opt (1)
            # if GenComment:
            #     f.write('@REM opt\n')
            # for m, h in zip([0.01], [0.15]):
            #     param = dict(DEFAULT_PARAM)
            #     param['poseDir'] = poseDir
            #     param['m'] = m
            #     param['h'] = h
            #     param['maxNumNodes'] = 'inf'
            #     expDir = generateExpDirName(
            #         FPS, VIDEO_LEN, scene, 
            #         **param
            #     )
            #     param['expDir'] = expDir
            #     f.write(f'mkdir {expDir}\n')
            #     for work in WORKS:
            #         param['work'] = work
            #         f.write(genCmd(**param))
            
        os.system(f'type {expName}_{scene}.{ext} >> {expName}.{ext}')