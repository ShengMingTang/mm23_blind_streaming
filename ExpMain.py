from Exp import *
from pathlib import Path
from Exp_miv_util import *
import argparse
import platform
from Exp_Gen import *

# ! constants
SCENES = ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']
RESOLUTION = (540, 960) # (H, W)
H, W = RESOLUTION
OPS = [
    # * Once
    'truncate', 'targetViews',
    # * Once for each candidate
    'generateCandidates', 'prepareCandidates', 'prune',
    # * For every experiment
    'exp', 'synthesize', 'qual',
]

def main(
    work:str, # must be one of the OPS
    poseDir: Path, expDir: Path, 
    runTimeResDs: int, m: float, h: float, 
    ffrMask:str, maxNumNodes: int, depthThres: float,
    psPolicy: str, cspPolicy: str,
    windowSize: int,
    solverPolicy: str,
    useRVS: bool,
    placePolicy: str,
    aa: float,
    # [dynamic]
    isDynamic: bool,
):
    assert work in OPS
    assert ffrMask in ['On', 'Off']
    assert placePolicy in ['average', 'IXR']
    
    VIDEO_LEN = STATIC_VIDEO_LEN if isDynamic == False else DYNAMIC_VIDEO_LEN # secs
    
    runTimeH, runTimeW = round(H / runTimeResDs), round(W / runTimeResDs)
    
    if ffrMask == 'Off':
        ffrMask = 1.0
    elif ffrMask == 'On':
        ffrMask = makeFoveationWeights(
            h=runTimeH, w=runTimeW,
            sizeY=0.35, sizeX=0.4,
            shiftY=0.0, shiftX=0.0,
            edgeRatioY=5.0, edgeRatioX=4.0
        )
    exp = Exp()
    '''
    Pose directory: Trace_FPS{FPS}_LEN{VIDEO_LEN}/{SCENE}
    '''
    # * Generate poses
    if work == 'truncate':
        for p in Path('Trace_Raw').glob('*'):
            truncatePoseDir2Dir(p, Path(f'Trace_FPS{FPS}_LEN{VIDEO_LEN}')/p.name, -FPS*VIDEO_LEN, None)
    # * Prepare for the target views
    if work == 'targetViews':
        for scene in SCENES:
            # don't use /scene as the directory to avoid overwrite old data
            exp.makeTargetViews(Path(f'Trace_FPS{FPS}_LEN{VIDEO_LEN}')/scene)
    # * Candidates
    if work == 'generateCandidates':
        exp.exp(
            # common
            outDir=expDir, cddOnly=True,
            # PoseFeeder
            poseDir=poseDir, windowSize=windowSize,
            # ContentCreator
            objPath=poseDir/'scene.obj', H=runTimeH, W=runTimeW,
            # PoseSplitter
            psPolicy=psPolicy, m=m, h=h,
            # Solver
            a=np.log(aa), ffrMask=ffrMask, maxNumNodes=maxNumNodes,
            # Cloud Service Provider
            cspPolicy=cspPolicy, depthThres=depthThres,
            solverPolicy=solverPolicy,
            placePolicy=placePolicy,
            # [dynamic]
            isDynamic=isDynamic,
            FPS=FPS,
        )
    if work == 'prepareCandidates':
        duplicates = windowSize if useRVS else 0
        exp.prepareCandidates(
            # PoseFeeder
            poseDir=poseDir, windowSize=windowSize,
            # ContentCreator
            H=H, W=W,
            # PoseSplitter
            psPolicy=psPolicy, m=m, h=h,
            # Cloud Service Provider
            cspPolicy=cspPolicy,
            duplicates=duplicates,
            pruneOnly=False,
            placePolicy=placePolicy,
        )
    if work == 'prune':
        duplicates = windowSize if useRVS else 0
        # [dynamic]
        if isDynamic:
            exp.prepareCandidatesDynamic(
                # PoseFeeder
                poseDir=poseDir, windowSize=windowSize,
                # ContentCreator
                H=H, W=W,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                duplicates=duplicates,
                pruneOnly=True,
                placePolicy=placePolicy,
            )
        else:
            exp.prepareCandidates(
                # PoseFeeder
                poseDir=poseDir, windowSize=windowSize,
                # ContentCreator
                H=H, W=W,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                duplicates=duplicates,
                pruneOnly=True,
                placePolicy=placePolicy,
            )
    # * Run exp
    if work == 'exp':
        exp.exp(
            # common
            outDir=expDir, cddOnly=False,
            # PoseFeeder
            poseDir=poseDir, windowSize=windowSize,
            # ContentCreator
            objPath=poseDir/'scene.obj', H=runTimeH, W=runTimeW,
            # PoseSplitter
            psPolicy=psPolicy, m=m, h=h,
            # Solver
            a=np.log(aa), ffrMask=ffrMask, maxNumNodes=maxNumNodes,
            # Cloud Service Provider
            cspPolicy=cspPolicy, depthThres=depthThres,
            solverPolicy=solverPolicy,
            placePolicy=placePolicy,
            # [dynamic]
            isDynamic=isDynamic,
            FPS=FPS,
        )
    # * Synthesis
    if work == 'synthesize':
        if useRVS:
            RENDERER_PATH = './RVS' if 'Linux' in platform.platform() else 'RVS.exe'
        else:
            RENDERER_PATH = Path('run_miv')/'tmiv_install'/'bin'/'Renderer'
        CONFIG_DIR = None if 'Linux' in platform.platform() else (Path('run_miv')/'config')
        # [dynamic]
        if isDynamic:
            exp.prepareSynthesisDynamic(
                # common
                expDir=expDir,
                # PoseFeeder
                poseDir=poseDir, windowSize=windowSize,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                useRVS=useRVS,
                placePolicy=placePolicy,
            )
            exp.synthesizeDynamic(
                RENDERER_PATH=RENDERER_PATH,
                CONFIG_DIR=CONFIG_DIR,
                expDir=expDir, poseDir=poseDir, windowSize=windowSize,
                H=H, W=W,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                useRVS=useRVS,
                placePolicy=placePolicy,
            )
        else:
            exp.prepareSynthesis(
                # common
                expDir=expDir,
                # PoseFeeder
                poseDir=poseDir, windowSize=windowSize,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                useRVS=useRVS,
                placePolicy=placePolicy,
            )
            exp.synthesize(
                RENDERER_PATH=RENDERER_PATH,
                CONFIG_DIR=CONFIG_DIR,
                expDir=expDir, poseDir=poseDir, windowSize=windowSize,
                H=H, W=W,
                # PoseSplitter
                psPolicy=psPolicy, m=m, h=h,
                # Cloud Service Provider
                cspPolicy=cspPolicy,
                useRVS=useRVS,
                placePolicy=placePolicy,
            )
    # * Group
    # if work == 'group':
    #     exp.groupTargetViewAfterSynthesis(
    #         expDir=expDir,
    #         poseDir=poseDir,
    #         H=H, W=W,
    #         useRVS=useRVS,
    #     )
    if work == 'qual':
        vmafPath = './vmaf' if 'Linux' in platform.platform() else 'vmaf.exe'
        exp.runQual(
            vmafPath=vmafPath,
            expDir=expDir, poseDir=poseDir,
            H=H, W=W,
        )
    print(f'{expDir} finished')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=str)
    parser.add_argument('--poseDir', type=Path)
    parser.add_argument('--expDir', type=Path)
    parser.add_argument('--runTimeResDs', type=int)
    parser.add_argument('--m', type=float)
    parser.add_argument('--h', type=float)
    parser.add_argument('--ffrMask', type=str)
    parser.add_argument('--maxNumNodes', type=str)
    parser.add_argument('--depthThres', type=float)
    parser.add_argument('--psPolicy', type=str)
    parser.add_argument('--cspPolicy', type=str)
    parser.add_argument('--windowSize', type=int)
    parser.add_argument('--solverPolicy', type=str)
    parser.add_argument('--useRVS', type=bool)
    parser.add_argument('--placePolicy', type=str)
    parser.add_argument('--aa', type=float)
    # [dynamic]
    parser.add_argument('--isDynamic', type=bool)
    
    args = parser.parse_args()
    args.maxNumNodes = int(args.maxNumNodes) if args.maxNumNodes != 'inf' else args.maxNumNodes
    
    args.expDir.mkdir(exist_ok=True, parents=True)
    args.poseDir.mkdir(exist_ok=True, parents=True)
    
    main(**vars(args))
    