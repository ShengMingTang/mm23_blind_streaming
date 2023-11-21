import platform

pythonCmd = 'python3' if 'Linux' in platform.system() else 'python'
ext = 'sh' if 'Linux' in platform.system() else 'bat'
useRVS = True
GenComment = True

FPS = 50
DEFAULT_PARAM = {
    'pythonCmd': pythonCmd, 
    'work': None, 
    'poseDir': None,
    'expDir': None,
    'runTimeResDs': 4,
    'm': 0.03,
    'h': 0.15,
    'ffrMask': 'Off',
    'maxNumNodes': 96,
    'depthThres': 1e-2,
    'psPolicy': 'uniform',
    'cspPolicy': '1Prob',
    'windowSize': FPS,
    'solverPolicy': 'UnB',
    'useRVS': True,
    'placePolicy': 'average',
    'a': 1e-5,
    'isDynamic': False,
}

STATIC_VIDEO_LEN = 30
# ! dynamic scene not implemeneted
DYNAMIC_VIDEO_LEN = 8

SCENES = ['FurnishedCabin', 'ScifiTraceBigroom', 'ScifiTraceSmallroom']

def genCmd(
    pythonCmd, work, poseDir, expDir, runTimeResDs,
    m, h, ffrMask, maxNumNodes, depthThres, psPolicy,
    cspPolicy, windowSize, solverPolicy, useRVS, placePolicy, a,
    **kwargs,
):
    return f'{pythonCmd} ExpMain.py --work {work} --poseDir {poseDir} --expDir {expDir} --runTimeResDs {runTimeResDs} --m {m} --h {h} --ffrMask {ffrMask} --maxNumNodes {maxNumNodes} --depthThres {depthThres} --psPolicy {psPolicy} --cspPolicy {cspPolicy} --windowSize {windowSize} --solverPolicy {solverPolicy} --useRVS {useRVS} --placePolicy {placePolicy} --aa {a} > {expDir}/stdout-{work}.txt 2> {expDir}/stderr-{work}.txt\n'


def genCmdDynamic(
    pythonCmd, work, poseDir, expDir, runTimeResDs,
    m, h, ffrMask, maxNumNodes, depthThres, psPolicy,
    cspPolicy, windowSize, solverPolicy, useRVS, placePolicy, a,
    isDynamic,
    **kwargs,
):
    return f'{pythonCmd} ExpMain.py --work {work} --poseDir {poseDir} --expDir {expDir} --runTimeResDs {runTimeResDs} --m {m} --h {h} --ffrMask {ffrMask} --maxNumNodes {maxNumNodes} --depthThres {depthThres} --psPolicy {psPolicy} --cspPolicy {cspPolicy} --windowSize {windowSize} --solverPolicy {solverPolicy} --useRVS {useRVS} --placePolicy {placePolicy} --aa {a} --isDynamic {isDynamic} > {expDir}/stdout-{work}.txt 2> {expDir}/stderr-{work}.txt\n'