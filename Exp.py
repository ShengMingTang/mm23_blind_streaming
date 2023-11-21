import Exp_miv_util
from pathlib import Path
from PoseFeeder import *
from ContentCreator import *
from PoseSplitter import *
from CamPlace import *
from CloudServiceProvider import *
from Solver import *
from Yuv import *
from Common import *
import time
import os
import shutil
import re
import pandas as pd
import platform
import ffmpeg
import math

def generateCandidateDirName(
    m: float, h: float,
    psPolicy: str, cspPolicy: str,
    windowSize: int,
    placePolicy: str,
    **kwargs,
):
    return Path(f'cdd_csp{cspPolicy}_w{windowSize}_p{psPolicy}_m{m}_h{h}_pl{placePolicy}')
def generatePoseDirName(
    fps: int, videoLen: int,
    scene: str,
    **kwargs,
):
    return Path(f'Trace_FPS{fps}_LEN{videoLen}')/scene
def generateExpDirName(
    fps: int, videoLen: int,
    scene: str,
    runTimeResDs: int, m: float, h: float, 
    ffrMask:str, maxNumNodes: int, depthThres: float,
    psPolicy: str, cspPolicy: str,
    windowSize: int,
    solverPolicy: str,
    placePolicy: str,
    a: float,
    **kwargs,
):
    return Path(f'Trace_FPS{fps}_LEN{videoLen}_s{scene}_rtR{runTimeResDs}_ffr{ffrMask}_maxNodes{maxNumNodes}_dep{depthThres}_slvr{solverPolicy}_a{a}_{generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)}')
def destructExpDirName(path: Path):
    path = str(path)
    pattern = 'Trace_FPS([0-9]*)_LEN([0-9]*)_s(.*)_rtR([0-9]*)_ffr(On|Off)_maxNodes([0-9]*|inf)_dep([0-9]*.[0-9]*)_slvr([a-zA-Z0-9]*)_a([0-9]*.?[0-9]*e?-?[0-9]*)'
    groups = re.findall(pattern, path)[0]
    match = re.search(pattern, path)
    expDirDict = {
        "fps": int(groups[0]),
        "videoLen": int(groups[1]),
        "scene": groups[2],
        "runTimeResDs": int(groups[3]),
        "ffr": groups[4],
        "maxNodes": groups[5],
        "depthThres": float(groups[6]),
        "solverPolicy": groups[7],
        "a": float(groups[8]),
    }
    if expDirDict['maxNodes'] == 'inf':
        expDirDict['maxNodes'] = math.inf
    else:
        expDirDict['maxNodes'] = int(expDirDict['maxNodes'])
    cddDirDict = destructCandidateDirName(path[match.end():])
    return {**expDirDict, **cddDirDict}
def destructCandidateDirName(path: Path):
    path = str(path)
    pattern = 'cdd_csp([0-9a-zA-Z]*)_w([0-9]*)_p([a-zA-Z]*)_m([0-9]*.[0-9]*)_h([0-9]*.[0-9]*)_pl([a-zA-Z]*)'
    groups = re.findall(pattern, path)[0]
    return {
        "cspPolicy": groups[0],
        "windowSize": int(groups[1]),
        "psPolicy": groups[2],
        "m": float(groups[3]),
        "h": float(groups[4]),
        "placePolicy": groups[5],
    }
def expDirName2CddDirName(expDirName: Path):
    path = str(expDirName)
    pattern = 'Trace_FPS([0-9]*)_LEN([0-9]*)_s(.*)_rtR([0-9])_ffr(On|Off)_maxNodes([0-9]*|inf)_dep([0-9]*.[0-9]*)_slvr([a-zA-Z]*)_'
    match = re.search(pattern, path)
    return path[match.end():]

class Exp():
    '''
    poseDir (static) = Scene dir, Source view dir
    |# * (the very sources of experiment)
    |    - pose*.csv
    |    - scene.obj
    |    - cameraParam.json
    |# * (unity generated target frames)
    |    - rgb_{u}_{i}.png (i^th frame of the u^th user)
    |# * (target views) call self.makeTargetViews(poseDir)
    |    - pose*_texture_*.yuv (converted from *.png)
    |# * (candidate cache)
    |   - cdd_*
    |   |    - cameraParam.json
    |   |    - # * (unity generated)
    |   |    - rgb_sv{g}_{i}.png (RGB source view of the i^th candidate for group g)
    |   |    - d_sv{g}_{i}.raw   (D source view of the i^th candidate for group g)
    |   |    - # * candidates cache derived by calling self.prepareCandidates(cddDir)
    |   |    - generated
    |   |    |   - sv{g}.csv (candidates of group g)
    |   |    |   - sv{g}.json (json describing each cameras in miv.json)
    |   |    |   - g{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv (depth yuv corresponds to the i^th selection in group g)
    |   |    |   - g{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv (color yuv corresponds to the i^th selection in group g)
    
    poseDir (dynamic) = Scene dir, Source view dir
    |# ! call self.dynamicExp()
    |# * (the very sources of experiment)
    |    - pose*.csv
    |    - scene.obj
    |    - cameraParam.json
    |# * (unity generated target frames)
    |    - rgb_{u}_{i}.png (i^th frame of the u^th user)
    |# * (target views) call self.makeTargetViews(poseDir)
    |    - pose*_texture_*.yuv (converted from *.png)
    |# * (candidate cache)
    |   - #! Note that multiple cdd_* must be shot together to guarantee deterministic results
    |   - cdd_*
    |   |    - cameraParam.json
    |   |    - # * (unity generated)
    |   |    - rgb_sv{c}_{i}.png (RGB source view of the cth camera for frame i)
    |   |    - d_sv{c}_{i}.raw   (D source view of the cth camera for frame i)
    |   |    - # * candidates cache derived by calling self.self.prepareCandidatesDynamic(cddDir)
    |   |    - generated
    |   |    |   - sv{c}.csv (source view pose of the c^th camera)
    |   |    |   - sv{g}.json (json describing each cameras (in unit of groups) in miv.json)
    |   |    |   - # * g{c}_{i} will be aggregated to g{c}_{g} for i corresponding to group g
    |   |    |   - g{c}_{g}_depth_{W}x{H}_yuv420p16le.yuv (depth yuv corresponds to the c^th camera for frame i)
    |   |    |   - g{c}_{g}_texture_{W}x{H}_yuv420p10le.yuv (color yuv corresponds to the c^th camera for frame i)
    
    |# * (encoded h264 bit stream)
    |   - encoded (encoded in unit of group)
    |   |    - pose*_texture_*.264 (converted from *.yuv)
    |   |    - cdd_*
    |   |    |   - g{g}_depth_{W}x{H}_yuv420p16le.264 (depth bitstream corresponds to group g)
    |   |    |   - g{g}_texture_{W}x{H}_yuv420p10le.264 (color bitstream corresponds to group g)
    |   - encodedSeparate (encoded in unit of frame)
    |   |    - pose*_texture_*.264 (converted from *.yuv)
    |   |    - cdd_*
    |   |    |   - g{g}_{i}_depth_{W}x{H}_yuv420p16le.264
    |   |    |   - g{g}_{i}_texture_{W}x{H}_yuv420p10le.264
    
    expDir
    |# * (generated by running self.exp)
    |    - ClassName.json (generated by each of the classes)
    |    - Exp.json (describes everything in this experiment)
    |      {'sols': describe the boolean mask to select the candidates}
    |# * call self.prepareSynthesis
    |    - config (static/dynamic) # * call self.prepareSynthesis or call self.prepareSynthesisDynamic
    |    |    - # * (useRVS=False) call self.prepareSynthesis
    |    |    - miv_{g}.json (let TMIV know which batch to synthesize)
    |    |    - miv_pose{g}.csv (batch to synthesize)
    |    |    - # * (useRVS=True) call self.prepareSynthesis
    |    |    - rvs_config_{g}_{u}.json (g: group, u: whichuser)
    |    |    - rvs_pose_{g}_{u}.csv (batch to sythesize)
    |    |    - rvs_{g}.json (source view camera data)
    |# * (after TMIV) call self.synthesize or self.synthesizeDynamic
    |    - output (static/dynamic)
    |    - # * (useRVS=False)
    |    |    - gOut{g}_depth_{W}x{H}_yuv420p16le.yuv (depth yuv for group g, user0 concat user 1 concat ... concat user last)
    |    |    - gOut{g}_texture_{W}x{H}_yuv420p10le.yuv (color yuv for group g, user0 concat user 1 concat ... concat user last)
    |    - # * (useRVS=True)
    |    |    - gOut{g}_{u}_depth_{W}x{H}_yuv420p16le.yuv (depth yuv for group g, user0 concat user 1 concat ... concat user last)
    |    |    - gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv (color yuv for group g, user0 concat user 1 concat ... concat user last)
    |# * (after synthesis) call self.groupTargetViewAfterSynthesis
    |    - video
    |    |    - pose{i}_texture_{W}x{H}_yuv420p10le.yuv (color yuv for user i)
    |# * (after vmaf) call self.runQual
    |    - qual
    |    |    - qual*.csv (vmaf, ssim, psnr)
    '''
    def __init__(self) -> None:
        self.f2d = Exp_miv_util.fDepthPlannarFactory(1000)
    def makeTargetViews(self, inputDir: Path):
        '''
        Call this to generate all target view needs
        '''
        Exp_miv_util.generateTMIVUsersStaticSceneFromDir(inputDir)
    def makeSourceViews(self, inputDir: Path, contentName="helloWorld"):
        '''
        inputDir: after capturing using MIV_Main
        contentName: may be set to experiement related so that we can undetstand what it is quickly
        '''
        Exp_miv_util.generateTMIVInputsStaticSceneFromDir(inputDir, contentName, self.f2d)
    def exp(self
            # common
            , outDir: Path, cddOnly:bool
            # PoseFeeder
            , poseDir: Path, windowSize: int
            # ContentCreator
            , objPath: Path, H:int, W: int
            # PoseSplitter
            , psPolicy: str, m: float, h: float
            # Solver
            , a: float, ffrMask: np.array, maxNumNodes: int
            # Cloud Service Provider
            , cspPolicy: str, depthThres: float,
            # solver
            solverPolicy: str,
            placePolicy: str,
            # [Dynamic]
            isDynamic: bool,
            FPS: int,
        ):
        '''
        '''
        # start session
        outDir.mkdir(exist_ok=True, parents=True)
        
        # PoseFeeder
        pfdSt = PoseFeeder.GetDefaultSettings()
        pfdSt['poseDir'] = Path(poseDir)
        pfdSt['size'] = windowSize
        pfd = PoseFeeder(pfdSt)
        U = next(iter(pfd)).shape[0]
        P = next(iter(pfd)).reshape((-1, 7)).shape[0]
        # ContentCreator
        ccSt = ContentCreator.GetDefaultSettings()
        ccSt['obj'] = objPath
        ccSt['height'] = H
        ccSt['width'] = W
        cc = ContentCreator(ccSt)
        # CloudServiceProvider
        cspSt = CloudServiceProvider.GetDefaultSettings()
        cspSt['policy'] = cspPolicy
        cspSt['depthThres'] = depthThres
        cspSt['maxNumNodes'] = math.inf if maxNumNodes == 'inf' else maxNumNodes # the case that unlimited BB
        psSt = PoseSplitter.GetDefaultSettings()
        psSt['policy'] = psPolicy
        # round to multiple of U
        if psSt['policy'] == 'uniform':
            psSt[psPolicy]['numPart'] = int(optProbRatio(cspPolicy, m, h, P) * m * P)
            psSt[psPolicy]['numPart'] = int(np.floor(psSt[psPolicy]['numPart'] / U) * U)
            psSt[psPolicy]['numPart'] = int(max(m*P, psSt[psPolicy]['numPart']))
            if isDynamic == False:
                assert psSt[psPolicy]['numPart'] % U == 0
            if cddOnly:
                print(psSt[psPolicy]['numPart'])
        elif psSt['policy'] == 'IXR':
            psSt[psPolicy]['ccSt'] = ccSt
            psSt[psPolicy]['depthThres'] = depthThres
            psSt[psPolicy]['numPart'] = -1 # M is meaningless in this case
            if cddOnly:
                print(psSt[psPolicy]['numPart'])
        camSt = CamPlace.GetDefaultSettings()
        camSt['policy'] = placePolicy
        slvrSt = Solver.GetDefaultSettings()
        slvrSt['a'] = a
        slvrSt['ffrMask'] = ffrMask
        slvrSt['policy'] = solverPolicy
        csp = CloudServiceProvider(cspSt, psSt, camSt, slvrSt)
        
        # param
        N = int(np.around(m * P))
        M = psSt[psPolicy]['numPart']
                
        # make mesh cache
        print(f'Making mesh cache')
        makeMeshFromRaysDepth(np.zeros((H, W, 6)), np.zeros((H, W)))
        # running session
        timeSplit = []
        timeCdd = []
        timeEsted = [] # time for estimating coverage table
        timePlace = [] # total time for CloudServiceProvider.place()
        sols = []
        opts = []
        optSearchSequences = []
        ubs = []
        ccCosts = []
        callbacks = {
            "onEsted": lambda x: timeEsted.append(x),
        }
        # [dynamic]
        svPoses = [[] for _ in range(M)]
        
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        # throw if the generated already
        if cddOnly:
            cddDir.mkdir(parents=True)
        
        for i, poses in enumerate(pfd):
            print(f'running {i}, with {N} source views, overhead {M}')
            tic = time.process_time()
            indices = csp.split(poses)
            timeSplit.append(time.process_time() - tic)
            print(f'splitted takes {timeSplit[-1]} sec')
            
            tic = time.process_time()
            cdds = csp.candidates(poses, indices)
            timeCdd.append(time.process_time() - tic)
            print(f'candidates takes {timeCdd[-1]} sec')
            
            if cddOnly:
                # cache candidates, they are uniquely determined by poseDir, windowSize, psPolicy, m, h
                # [dynamic]
                cddWithT = np.zeros((cdds.shape[0], cdds.shape[1] + 1))
                cddWithT[:, 1:] = cdds
                if isDynamic:
                    for j in range(cdds.shape[0]):
                        for _ in range(FPS):
                            svPoses[j].append(cddWithT[j])
                        np.savetxt(
                            str(cddDir/f'sv{j}.csv'),
                            svPoses[j],
                            fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw',
                            comments=''
                        )
                else: # [static]
                    np.savetxt(
                        str(cddDir/f'sv{i}.csv'),
                        cddWithT,
                        fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw',
                        comments=''
                    )
                continue
            
            tic = time.process_time()
            optSearchSequence = []
            callbacks["solverCallback"] = {
                "onOptUpdate": lambda x: optSearchSequence.append(x),    
            }
            sol, opt, ub = csp.place(cdds, N, cc, callbacks)
            timePlace.append(time.process_time() - tic)
            print(f'solver takes {timePlace[-1]} sec')
            print(f'sol = {sol}')
            sols.append(sol.tolist())
            opts.append(opt)
            optSearchSequences.append(optSearchSequence)
            ubs.append(ub)
            
            print(f'Rendering RGB source views')
            for i in range(N):
                cc.renderRGB(cdds[0])
            
            ccCosts.append(cc.getCost())
        
            # summary session
            pfd.summary(outDir)
            cc.summary(outDir)
            csp.summary(outDir)
            with open(outDir/'Exp.json', 'w') as f:
                json.dump({
                        'timeSplit': timeSplit,
                        'timeCdd': timeCdd,
                        'timeEsted': timeEsted,
                        'timePlace': timePlace,
                        'sols': sols,
                        'opts': opts,
                        'optSearchSequences': optSearchSequences,
                        'ubs': ubs,
                        'ccCosts': ccCosts,
                        # other params
                        'H': H,
                        'W': W,
                        'N': N,
                        'M': M,
                    }
                    , f
                )
    def prepareCandidates(self,
            # PoseFeeder
            poseDir: Path, windowSize: int,
            # ContentCreator
            H:int, W: int,
            # PoseSplitter
            psPolicy: str, m: float, h: float,
            # Cloud Service Provider
            cspPolicy: str,
            duplicates: int,
            pruneOnly: bool,
            placePolicy: str,
        ):
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        cddFns = list(cddDir.glob(f'sv*.csv'))
        cddGenDir = cddDir/'generated'
        cddGenDir.mkdir(exist_ok=True)
        assert len(cddFns) > 0
        cddFns = sorted(cddFns, key=lambda x: int(re.findall(f'sv(.*).csv', x.name)[0]))
        # (G, M, 8)
        cdds = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in cddFns]
        # cdds = np.array(cdds)
        # # t is omitted
        # cdds = cdds[:, :, 1:] # unity frame
        cdds = [cdd_i[:, 1:] for cdd_i in cdds]
        
        with open(cddDir/'cameraParam.json') as f:
            camParam = json.load(f)
        for g in range(len(cdds)): # sv{g}.csv
            cddsJ = {}
            for i in range(len(cdds[g])):
                # prepare texture
                if pruneOnly == False:
                    with open(cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv', mode='wb') as f:
                        out, _ = (
                            ffmpeg
                            .input(str(cddDir/f'rgb_sv{g}_{i}.png'))
                            .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                            .run(capture_stdout=True)
                        )
                        f.write(out)
                    # prepare duplicates for RVS
                    # if duplicates >= 1:
                    #     for _ in range(duplicates):
                    #         os.system(f"type {cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'} >> {cddGenDir/f'dup_sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'}")
                        # with open(cddGenDir/f'dup_sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv', mode='wb') as f:
                        #     yuv = Yuv(cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv')
                        #     for _ in range(duplicates):
                        #         f.write(yuv[0])
                
                # prepare depth
                data = np.fromfile(str(cddDir/f'd_sv{g}_{i}.raw'), dtype=np.float32)
                if data.size == H*W*4:
                    data = data.reshape((H, W, 4))
                    data = data[::-1, :, 0]
                    data.tofile(str(cddDir/f'd_sv{g}_{i}.raw'))
                else:
                    assert data.size == H*W
                    # load directly
                    data = np.fromfile(str(cddDir/f'd_sv{g}_{i}.raw'), dtype=np.float32)
                    data = data.reshape((H, W))
                
                if pruneOnly:
                    continue
                
                # ! replacing 1.0 with max encountered
                data[data == 1.0] = np.max(data[data != 1.0])
                
                depth = self.f2d(data)
                zmin, zmax = float(np.min(depth)), float(np.max(depth))
                with open(cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv', mode='wb') as f:
                    depth_16bit = (((1/depth-1/zmax) / (1/zmin-1/zmax)) * 65535)
                    depth_16bit = depth_16bit.astype(np.int16)
                    depth_16bit = np.append(depth_16bit, np.full(int(depth_16bit.size/2), 32768, dtype=np.int16))
                    f.write(depth_16bit.tobytes())
                # prepare duplicates for RVS
                # if duplicates >= 1:
                #     for _ in range(duplicates):
                #             os.system(f"type {cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv'} >> {cddGenDir/f'dup_sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv'}")
                    # with open(cddGenDir/f'dup_sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv', mode='wb') as f:
                    #     yuv = Yuv(cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv')
                    #     for _ in range(duplicates):
                    #         f.write(yuv[0]) 
                # prepare cameraJson
                camera = {}
                camera["BitDepthColor"] = 10
                camera["BitDepthDepth"] = 16
                camera["Name"] = f'sv{g}_{i}'
                camera["Depth_range"] = [zmin, zmax]
                camera["DepthColorSpace"] = "YUV420"
                camera["ColorSpace"] = "YUV420"
                MIV_camera_pose = convertUnityPoses7ToMIVCoord(cdds[g][i]).reshape((-1,))
                camera["Position"] = list(MIV_camera_pose[:3])
                camera["Rotation"] = list(MIV_camera_pose[3:])
                camera["Resolution"] = [W, H]
                camera["Projection"] = "Perspective"
                camera["HasInvalidDepth"] = False
                camera["Depthmap"] = 1
                camera["Background"] = 0
                # F = w / (2 * tan(FOV/2))
                # Use horizontal Fov in calculation, vertical Fov is determined automatically by aspect ratio
                camera["Focal"] = [
                    camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180)), camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180))
                ]
                # w / 2, h / 2
                camera["Principle_point"] = [
                    camera["Resolution"][0]/2, camera["Resolution"][1]/2
                ]        
                cddsJ[f'sv{g}_{i}'] = dict(camera)
            if pruneOnly == False:
                with open(cddGenDir/f'sv{g}.json', 'w') as f:
                    json.dump(cddsJ, f)
    def prepareCandidatesDynamic(self,
            # PoseFeeder
            poseDir: Path, windowSize: int,
            # ContentCreator
            H:int, W: int,
            # PoseSplitter
            psPolicy: str, m: float, h: float,
            # Cloud Service Provider
            cspPolicy: str,
            duplicates: int,
            pruneOnly: bool,
            placePolicy: str,
            groupSize: int,
        ):
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        cddFns = list(cddDir.glob(f'sv*.csv'))
        cddGenDir = cddDir/'generated'
        cddGenDir.mkdir(exist_ok=True)
        assert len(cddFns) > 0
        cddFns = sorted(cddFns, key=lambda x: int(re.findall(f'sv(.*).csv', x.name)[0]))
        # (G, M, 8)
        cdds = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in cddFns]
        # cdds = np.array(cdds)
        # # t is omitted
        # cdds = cdds[:, :, 1:] # unity frame
        cdds = [cdd_i[:, 1:] for cdd_i in cdds]
        
        with open(cddDir/'cameraParam.json') as f:
            camParam = json.load(f)
        assert cdds.shape[1] % groupSize == 0
        nGroups = cdds.shape[1] // groupSize
        return NotImplemented
        # TODO
        for g in range(nGroups): # sv{g}.csv
            cddsJ = {}
            for i in range(groupSize):
                # prepare texture
                if pruneOnly == False:
                    with open(cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv', mode='wb') as f:
                        out, _ = (
                            ffmpeg
                            .input(str(cddDir/f'rgb_sv{g}_{i}.png'))
                            .output('pipe:', format='rawvideo', pix_fmt='yuv420p10le')
                            .run(capture_stdout=True)
                        )
                        f.write(out)
                    # prepare duplicates for RVS
                    # if duplicates >= 1:
                    #     for _ in range(duplicates):
                    #         os.system(f"type {cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'} >> {cddGenDir/f'dup_sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'}")
                        # with open(cddGenDir/f'dup_sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv', mode='wb') as f:
                        #     yuv = Yuv(cddGenDir/f'sv{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv')
                        #     for _ in range(duplicates):
                        #         f.write(yuv[0])
                
                # prepare depth
                data = np.fromfile(str(cddDir/f'd_sv{g}_{i}.raw'), dtype=np.float32)
                if data.size == H*W*4:
                    data = data.reshape((H, W, 4))
                    data = data[::-1, :, 0]
                    data.tofile(str(cddDir/f'd_sv{g}_{i}.raw'))
                else:
                    assert data.size == H*W
                    # load directly
                    data = np.fromfile(str(cddDir/f'd_sv{g}_{i}.raw'), dtype=np.float32)
                    data = data.reshape((H, W))
                
                if pruneOnly:
                    continue
                
                # ! replacing 1.0 with max encountered
                data[data == 1.0] = np.max(data[data != 1.0])
                
                depth = self.f2d(data)
                zmin, zmax = float(np.min(depth)), float(np.max(depth))
                with open(cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv', mode='wb') as f:
                    depth_16bit = (((1/depth-1/zmax) / (1/zmin-1/zmax)) * 65535)
                    depth_16bit = depth_16bit.astype(np.int16)
                    depth_16bit = np.append(depth_16bit, np.full(int(depth_16bit.size/2), 32768, dtype=np.int16))
                    f.write(depth_16bit.tobytes())
                # prepare duplicates for RVS
                # if duplicates >= 1:
                #     for _ in range(duplicates):
                #             os.system(f"type {cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv'} >> {cddGenDir/f'dup_sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv'}")
                    # with open(cddGenDir/f'dup_sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv', mode='wb') as f:
                    #     yuv = Yuv(cddGenDir/f'sv{g}_{i}_depth_{W}x{H}_yuv420p16le.yuv')
                    #     for _ in range(duplicates):
                    #         f.write(yuv[0]) 
                # prepare cameraJson
                camera = {}
                camera["BitDepthColor"] = 10
                camera["BitDepthDepth"] = 16
                camera["Name"] = f'sv{g}_{i}'
                camera["Depth_range"] = [zmin, zmax]
                camera["DepthColorSpace"] = "YUV420"
                camera["ColorSpace"] = "YUV420"
                MIV_camera_pose = convertUnityPoses7ToMIVCoord(cdds[g][i]).reshape((-1,))
                camera["Position"] = list(MIV_camera_pose[:3])
                camera["Rotation"] = list(MIV_camera_pose[3:])
                camera["Resolution"] = [W, H]
                camera["Projection"] = "Perspective"
                camera["HasInvalidDepth"] = False
                camera["Depthmap"] = 1
                camera["Background"] = 0
                # F = w / (2 * tan(FOV/2))
                # Use horizontal Fov in calculation, vertical Fov is determined automatically by aspect ratio
                camera["Focal"] = [
                    camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180)), camera["Resolution"][0] / (2 * math.tan(camParam['horizontalFov']/2 * math.pi/180))
                ]
                # w / 2, h / 2
                camera["Principle_point"] = [
                    camera["Resolution"][0]/2, camera["Resolution"][1]/2
                ]        
                cddsJ[f'sv{g}_{i}'] = dict(camera)
            if pruneOnly == False:
                with open(cddGenDir/f'sv{g}.json', 'w') as f:
                    json.dump(cddsJ, f)
    def prepareSynthesis(self,
            # common
            expDir: Path,
            # PoseFeeder
            poseDir: Path, windowSize: int,
            # PoseSplitter
            psPolicy: str, m: float, h: float,
            # Cloud Service Provider
            cspPolicy: str,
            useRVS: bool,
            placePolicy: str,
        ):
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        cddFns = list(cddDir.glob(f'sv*.csv'))
        cddGenDir = cddDir/'generated'
        cddGenDir.mkdir(exist_ok=True)
        assert len(cddFns) > 0
        cddFns = sorted(cddFns, key=lambda x: int(re.findall(f'sv(.*).csv', x.name)[0]))
        # (G, M, 8)
        cdds = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in cddFns]
        # cdds = np.array(cdds)
        # # t is omitted
        # cdds = cdds[:, :, 1:] # unity frame
        cdds = [cdd_i[:, 1:] for cdd_i in cdds]
        
        with open(str(expDir/'Exp.json')) as f:
            expJ = json.load(f)
        with open(str(expDir/'PoseFeeder.json')) as f:
            pfJ = json.load(f)
        # (G, M)
        # sols = np.array(expJ['sols'], dtype=bool)
        sols = expJ['sols'] # list of lists
        
        # user poses
        userPoseFns = list(poseDir.glob('pose*.csv'))
        userPoseFns = sorted(userPoseFns, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
        # t is omitted
        # (U, P, 7)
        userPoses = [np.loadtxt(fn, skiprows=1, delimiter=',').reshape((-1, 8))[:, 1:] for fn in userPoseFns]
        P = pfJ['size']
        userPoses = np.array(userPoses)
        
        configDir = expDir/'config'
        configDir.mkdir(exist_ok=True)
        outputDir = expDir/'output'
        
        if useRVS: # use RVS, we need to prepare the config files
            with open(cddDir/'cameraParam.json', 'r') as f:
                cameraParamJ = json.load(f)
                W, H = cameraParamJ['width'], cameraParamJ['height']
            for g in range(len(sols)):
                # prepare rvs_{g}.json
                camera_parameter = {
                    "Version": "3.0",
                    "Content_name": expDir.stem,
                    "BoundingBox_center": [0,0,0],
                    "Fps": 1,
                    "Frames_number": 1,
                    "cameras": [] # ! override
                }
                # ! override
                camera_parameter["cameras"] = []
                sel = np.nonzero(sols[g])[0]
                # cameras
                with open(str(cddGenDir/f'sv{g}.json')) as f:
                    cddsJ = json.load(f)
                for idx in sel:
                    camera_parameter["cameras"].append(cddsJ[f'sv{g}_{idx}'])
                # ! always add viewport
                viewport_parameter = camera_parameter["cameras"][0].copy()
                viewport_parameter["Name"] = "viewport" # 0 for all pose terms, we don't need additional transform
                viewport_parameter["Position"] = [0.0, 0.0, 0.0]
                viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
                viewport_parameter["HasInvalidDepth"] = True
                camera_parameter["cameras"].append(viewport_parameter)
                with open(configDir/f'rvs_{g}.json', 'w') as f:
                    json.dump(camera_parameter, f)
                for u in range(userPoses.shape[0]):
                    sel = np.nonzero(sols[g])[0]
                    config_json = {
                        "Version": "2.0",
                        "InputCameraParameterFile": "*.json", # ! override 0
                        "VirtualCameraParameterFile": "*.json", # ! override 1
                        "InputCameraNames": ["list of camera names in Json"], # ! override 2
                        "VirtualCameraNames": ["viewport"],
                        "ViewImageNames": ["path_to_source_view_texture"], # ! override 3
                        "DepthMapNames": ["path_to_source_view_depth"], # ! override 4
                        "OutputFiles": ["output.yuv"], # ! override 5
                        "StartFrame": 0,
                        "NumberOfFrames": 1,
                        "NumberOfOutputFrames": 1, # ! override 6
                        "Precision": 2.0,
                        "ColorSpace": "YUV",
                        "ViewSynthesisMethod": "Triangles",
                        "BlendingMethod": "Simple",
                        "BlendingLowFreqFactor": 1.0,
                        "BlendingHighFreqFactor": 4.0,
                        "VirtualPoseTraceName": "path_to_pose_csv" # ! override 7
                    }
                    # ! RVS does not implement string escape, translate all '\\' instances to '/'
                    # ! override 0, 1
                    cameraParamFile = str(configDir/f'rvs_{g}.json')
                    cameraParamFile = cameraParamFile.replace('\\', '/')
                    config_json['InputCameraParameterFile'] = cameraParamFile
                    config_json['VirtualCameraParameterFile'] = cameraParamFile
                    # ! override 2, 3, 4
                    config_json['InputCameraNames'] = []
                    config_json['ViewImageNames'] = []
                    config_json['DepthMapNames'] = []
                    for idx in sel:
                        name = cddsJ[f'sv{g}_{idx}']['Name']
                        config_json['InputCameraNames'].append(name)
                        # texName = str(cddGenDir/f'dup_{name}_texture_{W}x{H}_yuv420p10le.yuv')
                        texName = str(cddGenDir/f'{name}_texture_{W}x{H}_yuv420p10le.yuv')
                        texName = texName.replace('\\', '/')
                        config_json['ViewImageNames'].append(texName)
                        # depName = str(cddGenDir/f'dup_{name}_depth_{W}x{H}_yuv420p16le.yuv')
                        depName = str(cddGenDir/f'{name}_depth_{W}x{H}_yuv420p16le.yuv')
                        depName = depName.replace('\\', '/')
                        config_json['DepthMapNames'].append(depName)
                    # ! override 5
                    config_json['OutputFiles'] = [str(outputDir/f'gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv').replace('\\', '/')]
                    # ! override 6
                    config_json['NumberOfOutputFrames'] = P
                    # ! override 7
                    posesForThisGroup = userPoses[u, g*windowSize:(g+1)*windowSize, :] # (W, 7)
                    mivPoses = convertUnityPoses7ToMIVCoord(posesForThisGroup) # (W, 7)
                    np.savetxt(str(configDir/f'rvs_pose_{g}_{u}.csv'), mivPoses, fmt='%.4f', header='X,Y,Z,Yaw,Pitch,Roll', comments='', delimiter=',')
                    poseName = str(configDir/f'rvs_pose_{g}_{u}.csv')
                    poseName = poseName.replace('\\', '/')
                    config_json['VirtualPoseTraceName'] = poseName
                    with open(str(configDir/f'rvs_config_{g}_{u}.json'), 'w') as f:
                        json.dump(config_json, f)
        else: # use TMIV
            # generate camera param json
            camera_parameter = {}
            camera_parameter['Version'] = '4.0'
            camera_parameter["BoundingBox_center"] = [0, 0, 0]
            camera_parameter["Fps"] = Exp_miv_util.UNITY_FPS
            camera_parameter["Content_name"] = expDir.stem
            camera_parameter["Frames_number"] = 1
            camera_parameter["lengthsInMeters"] = True
            # camera_parameter["sourceCameraNames"] = [svCsv.stem for svCsv in svCsvs]
            camera_parameter["sourceCameraNames"] = []
            camera_parameter["cameras"] = []
            for g in range(len(sols)):
                sel = np.nonzero(sols[g])[0]
                # prepare target pose
                posesForThisGroup = userPoses[:, g*windowSize:(g+1)*windowSize, :] # (U, W, 7)
                mivPoses = convertUnityPoses7ToMIVCoord(posesForThisGroup) # (U*W, 7)
                np.savetxt(str(configDir/f'miv_pose{g}.csv'), mivPoses, fmt='%.4f', header='X,Y,Z,Yaw,Pitch,Roll', comments='', delimiter=',')
                # cameras
                with open(str(configDir/f'sv{g}.json')) as f:
                    cddsJ = json.load(f)
                for idx in sel:
                    camera_parameter['sourceCameraNames'].append(cddsJ[f'sv{g}_{idx}']['Name'])
                    camera_parameter["cameras"].append(cddsJ[f'sv{g}_{idx}'])
                # ! always add viewport
                viewport_parameter = camera_parameter["cameras"][0].copy()
                viewport_parameter["Name"] = "viewport"
                viewport_parameter["Position"] = [0.0, 0.0, 0.0]
                viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
                viewport_parameter["HasInvalidDepth"] = True
                camera_parameter["cameras"].append(viewport_parameter)
                with open(configDir/f'miv_{g}.json', 'w') as f:
                    json.dump(camera_parameter, f)
    def prepareSynthesisDynamic(self,
            # common
            expDir: Path,
            # PoseFeeder
            poseDir: Path, windowSize: int,
            # PoseSplitter
            psPolicy: str, m: float, h: float,
            # Cloud Service Provider
            cspPolicy: str,
            useRVS: bool,
            placePolicy: str,
        ):
        return NotImplemented
        # TODO
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        cddFns = list(cddDir.glob(f'sv*.csv'))
        cddGenDir = cddDir/'generated'
        cddGenDir.mkdir(exist_ok=True)
        assert len(cddFns) > 0
        cddFns = sorted(cddFns, key=lambda x: int(re.findall(f'sv(.*).csv', x.name)[0]))
        # (G, M, 8)
        cdds = [np.loadtxt(fn, skiprows=1, delimiter=',') for fn in cddFns]
        # cdds = np.array(cdds)
        # # t is omitted
        # cdds = cdds[:, :, 1:] # unity frame
        cdds = [cdd_i[:, 1:] for cdd_i in cdds]
        
        with open(str(expDir/'Exp.json')) as f:
            expJ = json.load(f)
        with open(str(expDir/'PoseFeeder.json')) as f:
            pfJ = json.load(f)
        # (G, M)
        # sols = np.array(expJ['sols'], dtype=bool)
        sols = expJ['sols'] # list of lists
        
        # user poses
        userPoseFns = list(poseDir.glob('pose*.csv'))
        userPoseFns = sorted(userPoseFns, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
        # t is omitted
        # (U, P, 7)
        userPoses = [np.loadtxt(fn, skiprows=1, delimiter=',').reshape((-1, 8))[:, 1:] for fn in userPoseFns]
        P = pfJ['size']
        userPoses = np.array(userPoses)
        
        configDir = expDir/'config'
        configDir.mkdir(exist_ok=True)
        outputDir = expDir/'output'
        
        if useRVS: # use RVS, we need to prepare the config files
            with open(cddDir/'cameraParam.json', 'r') as f:
                cameraParamJ = json.load(f)
                W, H = cameraParamJ['width'], cameraParamJ['height']
            for g in range(len(sols)):
                # prepare rvs_{g}.json
                camera_parameter = {
                    "Version": "3.0",
                    "Content_name": expDir.stem,
                    "BoundingBox_center": [0,0,0],
                    "Fps": 1,
                    "Frames_number": 1,
                    "cameras": [] # ! override
                }
                # ! override
                camera_parameter["cameras"] = []
                sel = np.nonzero(sols[g])[0]
                # cameras
                with open(str(cddGenDir/f'sv{g}.json')) as f:
                    cddsJ = json.load(f)
                for idx in sel:
                    camera_parameter["cameras"].append(cddsJ[f'sv{g}_{idx}'])
                # ! always add viewport
                viewport_parameter = camera_parameter["cameras"][0].copy()
                viewport_parameter["Name"] = "viewport" # 0 for all pose terms, we don't need additional transform
                viewport_parameter["Position"] = [0.0, 0.0, 0.0]
                viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
                viewport_parameter["HasInvalidDepth"] = True
                camera_parameter["cameras"].append(viewport_parameter)
                with open(configDir/f'rvs_{g}.json', 'w') as f:
                    json.dump(camera_parameter, f)
                for u in range(userPoses.shape[0]):
                    sel = np.nonzero(sols[g])[0]
                    config_json = {
                        "Version": "2.0",
                        "InputCameraParameterFile": "*.json", # ! override 0
                        "VirtualCameraParameterFile": "*.json", # ! override 1
                        "InputCameraNames": ["list of camera names in Json"], # ! override 2
                        "VirtualCameraNames": ["viewport"],
                        "ViewImageNames": ["path_to_source_view_texture"], # ! override 3
                        "DepthMapNames": ["path_to_source_view_depth"], # ! override 4
                        "OutputFiles": ["output.yuv"], # ! override 5
                        "StartFrame": 0,
                        "NumberOfFrames": 1,
                        "NumberOfOutputFrames": 1, # ! override 6
                        "Precision": 2.0,
                        "ColorSpace": "YUV",
                        "ViewSynthesisMethod": "Triangles",
                        "BlendingMethod": "Simple",
                        "BlendingLowFreqFactor": 1.0,
                        "BlendingHighFreqFactor": 4.0,
                        "VirtualPoseTraceName": "path_to_pose_csv" # ! override 7
                    }
                    # ! RVS does not implement string escape, translate all '\\' instances to '/'
                    # ! override 0, 1
                    cameraParamFile = str(configDir/f'rvs_{g}.json')
                    cameraParamFile = cameraParamFile.replace('\\', '/')
                    config_json['InputCameraParameterFile'] = cameraParamFile
                    config_json['VirtualCameraParameterFile'] = cameraParamFile
                    # ! override 2, 3, 4
                    config_json['InputCameraNames'] = []
                    config_json['ViewImageNames'] = []
                    config_json['DepthMapNames'] = []
                    for idx in sel:
                        name = cddsJ[f'sv{g}_{idx}']['Name']
                        config_json['InputCameraNames'].append(name)
                        # texName = str(cddGenDir/f'dup_{name}_texture_{W}x{H}_yuv420p10le.yuv')
                        texName = str(cddGenDir/f'{name}_texture_{W}x{H}_yuv420p10le.yuv')
                        texName = texName.replace('\\', '/')
                        config_json['ViewImageNames'].append(texName)
                        # depName = str(cddGenDir/f'dup_{name}_depth_{W}x{H}_yuv420p16le.yuv')
                        depName = str(cddGenDir/f'{name}_depth_{W}x{H}_yuv420p16le.yuv')
                        depName = depName.replace('\\', '/')
                        config_json['DepthMapNames'].append(depName)
                    # ! override 5
                    config_json['OutputFiles'] = [str(outputDir/f'gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv').replace('\\', '/')]
                    # ! override 6
                    config_json['NumberOfOutputFrames'] = P
                    # ! override 7
                    posesForThisGroup = userPoses[u, g*windowSize:(g+1)*windowSize, :] # (W, 7)
                    mivPoses = convertUnityPoses7ToMIVCoord(posesForThisGroup) # (W, 7)
                    np.savetxt(str(configDir/f'rvs_pose_{g}_{u}.csv'), mivPoses, fmt='%.4f', header='X,Y,Z,Yaw,Pitch,Roll', comments='', delimiter=',')
                    poseName = str(configDir/f'rvs_pose_{g}_{u}.csv')
                    poseName = poseName.replace('\\', '/')
                    config_json['VirtualPoseTraceName'] = poseName
                    with open(str(configDir/f'rvs_config_{g}_{u}.json'), 'w') as f:
                        json.dump(config_json, f)
        else: # use TMIV
            # generate camera param json
            camera_parameter = {}
            camera_parameter['Version'] = '4.0'
            camera_parameter["BoundingBox_center"] = [0, 0, 0]
            camera_parameter["Fps"] = Exp_miv_util.UNITY_FPS
            camera_parameter["Content_name"] = expDir.stem
            camera_parameter["Frames_number"] = 1
            camera_parameter["lengthsInMeters"] = True
            # camera_parameter["sourceCameraNames"] = [svCsv.stem for svCsv in svCsvs]
            camera_parameter["sourceCameraNames"] = []
            camera_parameter["cameras"] = []
            for g in range(len(sols)):
                sel = np.nonzero(sols[g])[0]
                # prepare target pose
                posesForThisGroup = userPoses[:, g*windowSize:(g+1)*windowSize, :] # (U, W, 7)
                mivPoses = convertUnityPoses7ToMIVCoord(posesForThisGroup) # (U*W, 7)
                np.savetxt(str(configDir/f'miv_pose{g}.csv'), mivPoses, fmt='%.4f', header='X,Y,Z,Yaw,Pitch,Roll', comments='', delimiter=',')
                # cameras
                with open(str(configDir/f'sv{g}.json')) as f:
                    cddsJ = json.load(f)
                for idx in sel:
                    camera_parameter['sourceCameraNames'].append(cddsJ[f'sv{g}_{idx}']['Name'])
                    camera_parameter["cameras"].append(cddsJ[f'sv{g}_{idx}'])
                # ! always add viewport
                viewport_parameter = camera_parameter["cameras"][0].copy()
                viewport_parameter["Name"] = "viewport"
                viewport_parameter["Position"] = [0.0, 0.0, 0.0]
                viewport_parameter["Rotation"] = [0.0, 0.0, 0.0]
                viewport_parameter["HasInvalidDepth"] = True
                camera_parameter["cameras"].append(viewport_parameter)
                with open(configDir/f'miv_{g}.json', 'w') as f:
                    json.dump(camera_parameter, f)
    def synthesize(self,
        RENDERER_PATH: Path,
        CONFIG_DIR: Path, # not used for RVS
        expDir: Path, poseDir: Path, windowSize:int,
        H: int, W: int,
        # PoseSplitter
        psPolicy: str, m: float, h: float,
        # Cloud Service Provider
        cspPolicy: str,
        useRVS: bool,
        placePolicy: str,
    ):
        '''
        Run synthesizer for each group
        includes group after synthesizer each chunk
        '''
        with open(str(expDir/'Exp.json')) as f:
            expJ = json.load(f)
        with open(str(expDir/'PoseFeeder.json')) as f:
            pfdSt = json.load(f)
        nUsers = len(pfdSt['poseFns'])
        # sols = np.array(expJ['sols'], dtype=bool)
        sols = expJ['sols']
        numSamplesPerWindow = pfdSt['size'] * nUsers
        outputDir = expDir/'output'
        outputDir.mkdir(exist_ok=True, parents=True)
        cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        assert cddDir.is_dir()
        
        cddGenDir = cddDir/'generated'
        configDir = expDir/'config'
        videoDir = expDir/'video'
        videoDir.mkdir(exist_ok=True)
        
        if useRVS:
            for u in range(len(list(configDir.glob(f'rvs_config_0_*.json')))):
                print(f'synthesizing user user {u} RVS')
                groupedVideo = videoDir/f'pose{u}_texture_{W}x{H}_yuv420p10le.yuv'
                open(str(groupedVideo), 'w').close()
                tic = time.process_time()
                for g in range(len(sols)):
                    ret = os.system(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1')
                    print(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1')
                    if ret != 0:
                        print(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1 returns {ret}')
                    assert ret == 0
                    outUngrouped = outputDir/f'gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv'
                    os.system(f"type {outUngrouped} >> {groupedVideo}")
                    (configDir/f"rvs_config_{g}_{u}.json").unlink()
                    outUngrouped.unlink()
        else: # use TMIV
            return NotImplemented
        
            DEFAULT_SYNTHESIZER = 'AdditiveSynthesizer'
            synthesizerConfig = CONFIG_DIR/f'TMIV_{DEFAULT_SYNTHESIZER}_renderer_config.json'
            shutil.copyfile(
                str(synthesizerConfig),
                str(outputDir/'rendererConfig.json'),        
            )
            for g in range(len(sols)):
                print(f'synthesizing group {g}')
                tic = time.process_time()
                '''
                https://www.notion.so/TMIV-Usage-4ee5d4df09724acead7a1500dadc4ad4
                Path specification
                '''
                os.system(
                    f"{RENDERER_PATH} \
                    -n 1 -N {numSamplesPerWindow} -s {'.'} -f 0 -r rec_0 -P p01 \
                    -c {synthesizerConfig} \
                    -p configDirectory {CONFIG_DIR} \
                    -p inputDirectory {cddGenDir} \
                    -p inputSequenceConfigPathFmt {Path('..')/'..'/configDir}/miv_{g}.json \
                    -p inputViewportParamsPathFmt {Path('..')/'..'/configDir}/miv_{g}.json \
                    -p inputPoseTracePathFmt {Path('..')/'..'/configDir}/miv_pose{g}.csv \
                    -p outputDirectory {outputDir} \
                    -p outputViewportGeometryPathFmt gOut{g}_depth_{W}x{H}_yuv420p16le.yuv \
                    -p outputViewportTexturePathFmt gOut{g}_texture_{W}x{H}_yuv420p10le.yuv \
                    > {outputDir}/g{g}.log 2>&1"
                )
            print(f'takes {time.process_time() - tic} secs')
    def synthesizeDynamic(self,
        RENDERER_PATH: Path,
        CONFIG_DIR: Path, # not used for RVS
        expDir: Path, poseDir: Path, windowSize:int,
        H: int, W: int,
        # PoseSplitter
        psPolicy: str, m: float, h: float,
        # Cloud Service Provider
        cspPolicy: str,
        useRVS: bool,
        placePolicy: str,
    ):
        '''
        Run synthesizer for each group
        includes group after synthesizer each chunk
        '''
        return NotImplemented
        # TODO
        # with open(str(expDir/'Exp.json')) as f:
        #     expJ = json.load(f)
        # with open(str(expDir/'PoseFeeder.json')) as f:
        #     pfdSt = json.load(f)
        # nUsers = len(pfdSt['poseFns'])
        # # sols = np.array(expJ['sols'], dtype=bool)
        # sols = expJ['sols']
        # numSamplesPerWindow = pfdSt['size'] * nUsers
        # outputDir = expDir/'output'
        # outputDir.mkdir(exist_ok=True, parents=True)
        # cddDir = poseDir/generateCandidateDirName(m, h, psPolicy, cspPolicy, windowSize, placePolicy)
        # assert cddDir.is_dir()
        
        # cddGenDir = cddDir/'generated'
        # configDir = expDir/'config'
        # videoDir = expDir/'video'
        # videoDir.mkdir(exist_ok=True)
        
        # if useRVS:
        #     for u in range(len(list(configDir.glob(f'rvs_config_0_*.json')))):
        #         print(f'synthesizing user user {u} RVS')
        #         groupedVideo = videoDir/f'pose{u}_texture_{W}x{H}_yuv420p10le.yuv'
        #         open(str(groupedVideo), 'w').close()
        #         tic = time.process_time()
        #         for g in range(len(sols)):
        #             ret = os.system(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1')
        #             print(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1')
        #             if ret != 0:
        #                 print(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1 returns {ret}')
        #             assert ret == 0
        #             outUngrouped = outputDir/f'gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv'
        #             os.system(f"type {outUngrouped} >> {groupedVideo}")
        #             (configDir/f"rvs_config_{g}_{u}.json").unlink()
        #             outUngrouped.unlink()
        # else: # use TMIV
        #     return NotImplemented
        
        #     DEFAULT_SYNTHESIZER = 'AdditiveSynthesizer'
        #     synthesizerConfig = CONFIG_DIR/f'TMIV_{DEFAULT_SYNTHESIZER}_renderer_config.json'
        #     shutil.copyfile(
        #         str(synthesizerConfig),
        #         str(outputDir/'rendererConfig.json'),        
        #     )
        #     for g in range(len(sols)):
        #         print(f'synthesizing group {g}')
        #         tic = time.process_time()
        #         '''
        #         https://www.notion.so/TMIV-Usage-4ee5d4df09724acead7a1500dadc4ad4
        #         Path specification
        #         '''
        #         os.system(
        #             f"{RENDERER_PATH} \
        #             -n 1 -N {numSamplesPerWindow} -s {'.'} -f 0 -r rec_0 -P p01 \
        #             -c {synthesizerConfig} \
        #             -p configDirectory {CONFIG_DIR} \
        #             -p inputDirectory {cddGenDir} \
        #             -p inputSequenceConfigPathFmt {Path('..')/'..'/configDir}/miv_{g}.json \
        #             -p inputViewportParamsPathFmt {Path('..')/'..'/configDir}/miv_{g}.json \
        #             -p inputPoseTracePathFmt {Path('..')/'..'/configDir}/miv_pose{g}.csv \
        #             -p outputDirectory {outputDir} \
        #             -p outputViewportGeometryPathFmt gOut{g}_depth_{W}x{H}_yuv420p16le.yuv \
        #             -p outputViewportTexturePathFmt gOut{g}_texture_{W}x{H}_yuv420p10le.yuv \
        #             > {outputDir}/g{g}.log 2>&1"
        #         )
        #     print(f'takes {time.process_time() - tic} secs')
    def groupTargetViewAfterSynthesis(self,
            expDir: Path, poseDir: Path,
            H: int, W: int,
            useRVS: bool,
        ):
        '''
        Group the results of each group
        
        expDir
        ...
            - output
        |    |    - gOut{g}_depth_{W}x{H}_yuv420p16le.yuv (depth yuv for group g, user0 concat user 1 concat ... concat user last)
        |    |    - gOut{g}_texture_{W}x{H}_yuv420p10le.yuv (color yuv for group g, user0 concat user 1 concat ... concat user last)
        |    |    -# * (after TMIV has synthesized) call self.groupTargetViewAfterSynthesis
        |    |    - pose{i}_texture_{W}x{H}_yuv420p10le.yuv (color yuv for user i)
    
        '''
        return NotImplemented
    
        with open(str(expDir/'PoseFeeder.json')) as f:
            pfdSt = json.load(f)
        with open(str(expDir/'Exp.json')) as f:
            expJ = json.load(f)
        nUsers = len(pfdSt['poseFns'])
        # sols = np.array(expJ['sols'], dtype=bool) # (G, M)
        sols = expJ['sols']
        
        videoDir = expDir/'video'
        videoDir.mkdir(exist_ok=True)
        outputDir = expDir/'output'
        
        for i in range(nUsers):
            print(f'grouping user {i}')
            if useRVS:
                # create empty file
                groupedVideo = videoDir/f'pose{i}_texture_{W}x{H}_yuv420p10le.yuv'
                open(str(groupedVideo), 'w').close()
                for g in range(len(sols)):
                    outUngrouped = outputDir/f'gOut{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'
                    os.system(f"type {outUngrouped} >> {groupedVideo}")
                    outUngrouped.unlink()
                # with open(videoDir/f'pose{i}_texture_{W}x{H}_yuv420p10le.yuv', 'wb') as f:
                #     for g in range(len(sols)):
                #         print(f'data group {g}')
                #         outUngrouped = outputDir/f'gOut{g}_{i}_texture_{W}x{H}_yuv420p10le.yuv'
                #         yuv = Yuv(outUngrouped)
                #         for j in range(len(yuv)):
                #             f.write(yuv[j])
            else:
                return NotImplemented
                with open(videoDir/f'pose{i}_texture_{W}x{H}_yuv420p10le.yuv', 'wb') as f:
                    for g in range(len(sols)):
                        print(f'data group {g}')
                        outUngrouped = outputDir/f'gOut{g}_texture_{W}x{H}_yuv420p10le.yuv'
                        if outUngrouped.exists():
                            yuv = Yuv(outUngrouped)
                            for frameIdx in range(i*pfdSt['size'], (i+1)*pfdSt['size']):
                                f.write(yuv[frameIdx])
    def runQual(self,
            vmafPath: str,
            expDir: Path, poseDir: Path,
            H: int, W: int,
        ):
        '''
        See https://github.com/Netflix/vmaf/blob/master/resource/doc/python.md#command-line-tools
        
        # ! Add these lines to make run_vmaf.py work
        # [SM]
        import os
        import sys
        cur = os.path.abspath(os.path.dirname(__file__))
        root = os.path.split(os.path.split(cur)[0])[0]
        sys.path.append(root)
        # [SM]
        '''        
        videoDir = expDir/'video'
        qualDir = expDir/'qual'

        qualDir.mkdir(exist_ok=True)
        
        with open(str(expDir/'PoseFeeder.json')) as f:
            pfdSt = json.load(f)
        nUsers = len(pfdSt['poseFns'])
        pixFmt = 'yuv420p10le'
        for i in range(nUsers):           
            assert '420' in pixFmt
            assert '10le' in pixFmt
            os.system(f"{vmafPath} \
                --reference {poseDir/f'pose{i}_texture_{W}x{H}_{pixFmt}.yuv'} \
                --distorted {videoDir/f'pose{i}_texture_{W}x{H}_{pixFmt}.yuv'} \
                --width {W} --height {H} --pixel_format 420 --bitdepth 10 \
                --model version=vmaf_v0.6.1 \
                --feature psnr \
                --feature float_ssim \
                --o {qualDir/f'qual{i}.csv'} --csv"
            )
    def checkMissing(self, 
        pattern: str, 
        windowSize: int,H: int, W:int, Nuser: int,
        poseDir: Path, RENDERER_PATH: Path, CONFIG_DIR: Path,
        useRVS: bool,
    ):
        '''
        This function is used to search missing groups during synthesis by checking if the file size matched the expected
        # ! format must be yuv420p10le or yuv420p16le
        print the commands for synthesizing missing groups
        return list of cmds in str
        '''
        # ! not maintained
        return NotImplemented
    
        allDirs = list(Path('.').glob(pattern))
        expDirs = []
        counts = 0
        cmds = []
        for expDir in allDirs:
            configDir = expDir/'config'
            if useRVS:
                expectedGroups = len(list(configDir.glob('rvs_*.json')))
                with open(str(expDir/'PoseFeeder.json')) as f:
                    pfdSt = json.load(f)
                nUsers = len(pfdSt['poseFns'])
                numSamplesPerWindow = pfdSt['size']
                outputDir = expDir/'output'
                outputDir.mkdir(exist_ok=True, parents=True)
                cddDirName = expDirName2CddDirName(expDir)
                cddDir = poseDir/cddDirName
                assert cddDir.is_dir()
                for g in range(expectedGroups):
                    for u in range(nUsers):
                        fn = (outputDir/f'gOut{g}_{u}_texture_{W}x{H}_yuv420p10le.yuv')
                        if fn.is_file() == False or fn.stat().st_size != expectedSize:
                            expDirs.append(str(expDir))
                            os.system(f'{RENDERER_PATH} {str(configDir/f"rvs_config_{g}_{u}.json")} > {outputDir}/g{g}_{u}.log 2>&1')
                            cmds.append(
                                f"{RENDERER_PATH} {str(configDir/f'rvs_config_{g}_{u}.json')} > {outputDir}/g{g}_{u}.log 2>&1\n\
                                    echo \"{counts} done\" \n"
                            )
                            counts += 1
            else:
                expectedGroups = len(list(configDir.glob('miv_*.json')))
                expectedSize = (H * W * 3 ) * windowSize * Nuser
                with open(str(expDir/'PoseFeeder.json')) as f:
                    pfdSt = json.load(f)
                nUsers = len(pfdSt['poseFns'])
                numSamplesPerWindow = pfdSt['size'] * nUsers
                DEFAULT_SYNTHESIZER = 'AdditiveSynthesizer'
                synthesizerConfig = CONFIG_DIR/f'TMIV_{DEFAULT_SYNTHESIZER}_renderer_config.json'
                outputDir = expDir/'output'
                outputDir.mkdir(exist_ok=True, parents=True)
                cddDirName = expDirName2CddDirName(expDir)
                cddDir = poseDir/cddDirName
                assert cddDir.is_dir()
                for g in range(expectedGroups):
                    fn = (outputDir/f'gOut{g}_texture_{W}x{H}_yuv420p10le.yuv')
                    if fn.is_file() == False or fn.stat().st_size != expectedSize:
                        expDirs.append(str(expDir))
                        cmds.append(
                            f"{RENDERER_PATH} \
                            -n 1 -N {numSamplesPerWindow} -s {'.'} -f 0 -r rec_0 -P p01 \
                            -c {synthesizerConfig} \
                            -p configDirectory {CONFIG_DIR} \
                            -p inputDirectory {cddDir} \
                            -p inputSequenceConfigPathFmt {Path('..')/'..'/expDir}/miv_{g}.json \
                            -p inputViewportParamsPathFmt {Path('..')/'..'/expDir}/miv_{g}.json \
                            -p inputPoseTracePathFmt {Path('..')/'..'/expDir}/miv_pose{g}.csv \
                            -p outputDirectory {outputDir} \
                            -p outputViewportGeometryPathFmt gOut{g}_depth_{W}x{H}_yuv420p16le.yuv \
                            -p outputViewportTexturePathFmt gOut{g}_texture_{W}x{H}_yuv420p10le.yuv \
                            > {outputDir}/g{g}.log 2>&1 \n\
                            echo \"{counts} done\" \n"
                        )
                        counts += 1
        return cmds
    def mergeAllQualFiles(self, nFramePerCsv: int, pattern: str, outputCsv: Path, outputExpJson: Path):
        '''
        merge all qual*.csv under all directories that satisfies pattern
        format = 
        expDir user (qual*.csv)...
        nFramePerCsv: expected number of frames in a single qual*.csv
        '''
        allDirs = list(Path('.').glob(pattern))
        print(len(allDirs))
        data =  []
        expDirs = []
        users = []
        dirKeys = {}
        outJ = {}
        for dir in allDirs:
            print(f'merging {dir}')
            keys = destructExpDirName(dir)
            dirStr = dir.parts[-1]
            outJ[dirStr] = {}
            # merge Exp.json into that term
            with open(str(dir/'Exp.json'), 'r') as f:
                j = json.load(f)
                for key in j:
                    outJ[dirStr][key] = j[key]
            # merge directory key into that term
            for key in keys:
                keys[key] = keys[key]
                outJ[dirStr][key] = keys[key]
            if dirKeys == {}:
                for key in keys:
                    dirKeys[key] = []            
            qualDir = dir/'qual'
            assert len(list(qualDir.glob('qual*.csv'))) == 16
            for qualFn in qualDir.glob('qual*.csv'):
                user = int(re.findall('qual([0-9]*)', str(qualFn.stem))[0])
                data.append(pd.read_csv(str(qualFn)))
                print(data[-1].shape[0])
                if data[-1].shape[0] != nFramePerCsv:
                    print(f'{dir}/{qualFn} does not satisfy {nFramePerCsv} frames, got {data[-1].shape[0]}')
                    return
                expDirs.extend([str(dir.parts[-1]) for _ in range(data[-1].shape[0])])
                users.extend([str(user) for _ in range(data[-1].shape[0])])
                for key in keys:
                    dirKeys[key].extend([keys[key]] * data[-1].shape[0])
        mergedData = pd.concat(data)
        mergedData.reset_index(inplace=True, drop=True)
        expDirs = pd.Series(expDirs, name='expDir').reset_index(drop=True)
        users = pd.Series(users, name='user').reset_index(drop=True)
        keysDf = pd.DataFrame(dirKeys)
        keysDf.reset_index(inplace=True, drop=True)
        print(mergedData.shape, expDirs.shape, users.shape, keysDf.shape)
        mergedData = pd.concat([expDirs, users, mergedData, keysDf], axis=1)
        mergedData.to_csv(str(outputCsv))
        
        with open(str(outputExpJson), 'w') as f:
            json.dump(outJ, f)
        
        return mergedData.shape
    def encodeTargetView(self, poseDir: Path, W: int, H:int, fps: int):
        pattern = 'pose*.yuv'
        encodeDir = poseDir/'encoded'
        encodeDir.mkdir(exist_ok=True)
        poseYuvs = list(poseDir.glob(pattern))
        for yuv in poseYuvs:
            print(f'encoding {yuv}')
            logFile = encodeDir/f"{yuv.stem}.txt"
            os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v {W}x{H} -r {fps} -i {yuv} -c:v libx264 -qp 0 {encodeDir/f"{yuv.stem}"}.264 > {logFile} 2>&1')
    def encodeSourceView(self, poseDir: Path, W: int, H:int, runTimeResDs: int):
        encodeDir = poseDir/'encoded'
        encodeDir.mkdir(exist_ok=True)
        pattern = 'cdd_*'
        cddDirs = list(poseDir.glob(pattern))
        for cddDir in cddDirs:
            print(f'encoding {cddDir}')
            outDir = encodeDir/cddDir.parts[-1]
            outDir.mkdir(exist_ok=True, parents=True)
            isIXR = 'IXR' in str(cddDir.parts[-1])
            if isIXR == False:
                # encode {*}_{i} to {i}, encode effective downsampling of a single user
                nViews = len(list((cddDir/'generated').glob(f'sv0_*_depth_{W}x{H}_yuv420p16le.yuv')))
                for i in range(nViews):
                    print(f'view {i}')
                    # depth
                    grouped = outDir/f'sv{i}_depth_{W}x{H}_yuv420p16le.yuv'
                    grouped.unlink(missing_ok=True)
                    outFile = outDir/f'sv{i}_depth_{W}x{H}_yuv420p16le.264'
                    auxFile = outDir/f'aux_{runTimeResDs}_sv{i}_depth_{W}x{H}_yuv420p16le.264'
                    # concat
                    files = list((cddDir/'generated').glob(f'sv*_{i}_depth_{W}x{H}_yuv420p16le.yuv'))
                    files = sorted(files, key=lambda x: int(re.findall(f'sv(.*)_{i}_depth_{W}x{H}_yuv420p16le.yuv', x.name)[0]))
                    for file in files:
                        os.system(f'type {file} >> {grouped}')
                    # enocde
                    logFile = encodeDir/f"{outFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264  -qp 0 {outFile} > {logFile} 2>&1')
                    roundedH = H//runTimeResDs
                    if roundedH % 2:
                        roundedH += 1
                    logFile = encodeDir/f"{auxFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264 -vf scale=-1:{roundedH}  -qp 0 {auxFile} > {logFile} 2>&1')
                    grouped.unlink()
                    # color
                    grouped = outDir/f'sv{i}_texture_{W}x{H}_yuv420p10le.yuv'
                    grouped.unlink(missing_ok=True)
                    outFile = outDir/f'sv{i}_texture_{W}x{H}_yuv420p10le.264'
                    # concat
                    files = list((cddDir/'generated').glob(f'sv*_{i}_texture_{W}x{H}_yuv420p10le.yuv'))
                    files = sorted(files, key=lambda x: int(re.findall(f'sv(.*)_{i}_texture_{W}x{H}_yuv420p10le.yuv', x.name)[0]))
                    for file in files:
                        os.system(f'type {file} >> {grouped}')
                    # enocde
                    logFile = encodeDir/f"{outFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264 -qp 0 {outFile} > {logFile} 2>&1')
                    grouped.unlink()
            else:
                # encode {g}_{*} to {g}, they don't have dependency for IXR cdds
                nGroups = len(list((cddDir/'generated').glob(f'sv*_0_depth_{W}x{H}_yuv420p16le.yuv')))
                for g in range(nGroups):
                    print(f'group {g}')
                    # depth
                    grouped = outDir/f'sv{g}_depth_{W}x{H}_yuv420p16le.yuv'
                    grouped.unlink(missing_ok=True)
                    outFile = outDir/f'sv{g}_depth_{W}x{H}_yuv420p16le.264'
                    auxFile = outDir/f'aux_{runTimeResDs}_sv{g}_depth_{W}x{H}_yuv420p16le.264'
                    # concat
                    files = list((cddDir/'generated').glob(f'sv{g}_*_depth_{W}x{H}_yuv420p16le.yuv'))
                    files = sorted(files, key=lambda x: int(re.findall(f'sv{g}_(.*)_depth_{W}x{H}_yuv420p16le.yuv', x.name)[0]))
                    for file in files:
                        os.system(f'type {file} >> {grouped}')
                    # enocde
                    logFile = encodeDir/f"{outFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264 -qp 0 {outFile} > {logFile} 2>&1')
                    roundedH = H//runTimeResDs
                    if roundedH % 2:
                        roundedH += 1
                    logFile = encodeDir/f"{auxFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264 -vf scale=-1:{roundedH} -qp 0  {auxFile} > {logFile} 2>&1')
                    grouped.unlink()
                    # color
                    grouped = outDir/f'sv{g}_texture_{W}x{H}_yuv420p10le.yuv'
                    grouped.unlink(missing_ok=True)
                    outFile = outDir/f'sv{g}_texture_{W}x{H}_yuv420p10le.264'
                    # concat
                    files = list((cddDir/'generated').glob(f'sv{g}_*_texture_{W}x{H}_yuv420p10le.yuv'))
                    files = sorted(files, key=lambda x: int(re.findall(f'sv{g}_(.*)_texture_{W}x{H}_yuv420p10le.yuv', x.name)[0]))
                    for file in files:
                        os.system(f'type {file} >> {grouped}')
                    # enocde
                    logFile = encodeDir/f"{outFile.stem}.txt"
                    os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v {W}x{H} -r {1} -i {grouped} -c:v libx264 -qp 0 {outFile} > {logFile} 2>&1')
                    grouped.unlink()
    def encodedToSummary(self, rootDir: Path, outputCsv: Path):
        data = {
            'scene': [], # pose directory
            'cddDir': [], # candidate directory, empty for target views
            'type': [], # str = 'tv'(target view) | 'texture' | 'depth' (source view) | 'aux'(auxliary)
            'group': [], # int: -1 for target view, >= 0 for source view (0~29) in our case
            'size': [], # int: represent file size
            # **cddDirKeys
        }
        temp = rootDir/'FurnishedCabin'/'encoded'
        cddDirKeys = destructCandidateDirName(list(temp.glob('cdd_*'))[0])
        for key in cddDirKeys:
            data[key] = []
        for sceneDir in rootDir.glob('*'):
            scene = sceneDir.parts[-1]
            dir = sceneDir/'encoded'
            cddDirs = list(dir.glob('cdd_*'))
            # target views
            for tv in dir.glob('pose*.264'):
                data['scene'].append(scene)
                data['cddDir'].append('N/A')
                data['type'].append('tv')
                data['group'].append(int(re.findall('pose([0-9]+)', tv.stem)[0]))
                data['size'].append(tv.stat().st_size)
                for key in cddDirKeys:
                    data[key].append('N/A')
            # source views
            for cddDir in cddDirs:
                keys = destructCandidateDirName(cddDir.parts[-1])
                print(cddDir)
                for tex in cddDir.glob('sv*_texture*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('texture')
                    data['group'].append(int(re.findall('sv([0-9]+)', tex.stem)[0]))
                    data['size'].append(tex.stat().st_size)
                    for key in keys:
                        data[key].append(keys[key])
                for dep in cddDir.glob('sv*_depth*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('depth')
                    data['group'].append(int(re.findall('sv([0-9]+)', dep.stem)[0]))
                    data['size'].append(dep.stat().st_size)
                    for key in keys:
                        data[key].append(keys[key])
                for aux in cddDir.glob('aux*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('aux')
                    data['group'].append(int(re.findall('aux_([0-9]+)_sv([0-9]+)', aux.stem)[0][1]))
                    data['size'].append(aux.stat().st_size)
                    for key in keys:
                        data[key].append(keys[key])
        df = pd.DataFrame(data=data)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(outputCsv)
    def encodeSourceViewSeparate(self, poseDir: Path, W: int, H:int, runTimeResDs: int):
        encodeDir = poseDir/'encodedSeparate'
        encodeDir.mkdir(exist_ok=True)
        pattern = 'cdd_*'
        cddDirs = list(poseDir.glob(pattern))
        for cddDir in cddDirs:
            print(f'encoding {cddDir}')
            outDir = encodeDir/cddDir.parts[-1]
            outDir.mkdir(exist_ok=True, parents=True)
            for tex in list((cddDir/'generated').glob(f'sv*_texture_{W}x{H}_yuv420p10le.yuv')):
                outFile = outDir/f'{tex.stem}.264'
                logFile = outDir/f'{tex.stem}.txt'
                os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v {W}x{H} -r {1} -i {tex} -c:v libx264 -qp 0 {outFile} > {logFile} 2>&1')
            for dep in list((cddDir/'generated').glob(f'sv*_depth_{W}x{H}_yuv420p16le.yuv')):
                outFile = outDir/f'{dep.stem}.264'
                logFile = outDir/f'{dep.stem}.txt'
                os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {dep} -c:v libx264  -qp 0 {outFile} > {logFile} 2>&1')
                # aux
                outFile = outDir/f'aux_{runTimeResDs}_{dep.stem}.264'
                logFile = outDir/f'aux_{runTimeResDs}_{dep.stem}.txt'
                roundedH = H//runTimeResDs
                if roundedH % 2:
                    roundedH += 1
                os.system(f'ffmpeg -y -f rawvideo -pix_fmt yuv420p16le -s:v {W}x{H} -r {1} -i {dep} -c:v libx264 -vf scale=-1:{roundedH}  -qp 0 {outFile} > {logFile} 2>&1')
    def encodedSeparateToSummary(self, rootDir: Path, outputCsv: Path):
        data = {
            'scene': [], # pose directory
            'cddDir': [], # candidate directory, empty for target views
            'type': [], # str = 'tv'(target view) | 'texture' | 'depth' (source view) | 'aux'(auxliary)
            'group': [], # int: -1 for target view, >= 0 for source view (0~29) in our case
            'viewId': [], # int: userID for target view, >= 0 for source view id
            'bytes': [], # int: represent file size
            'path': [], # path to this file
            # **cddDirKeys
        }
        temp = rootDir/'FurnishedCabin'/'encodedSeparate'
        cddDirKeys = destructCandidateDirName(list(temp.glob('cdd_*'))[0])
        for key in cddDirKeys:
            data[key] = []
        for sceneDir in rootDir.glob('*'):
            scene = sceneDir.parts[-1]
            dir = sceneDir/'encodedSeparate'
            cddDirs = list(dir.glob('cdd_*'))
            # target views
            for tv in dir.glob('pose*.264'):
                data['scene'].append(scene)
                data['cddDir'].append('N/A')
                data['type'].append('tv')
                data['group'].append(-1)
                data['viewId'].append(int(re.findall('pose([0-9]+)', tv.stem)[0]))
                data['bytes'].append(tv.stat().st_size)
                data['path'].append(str(tv))
                for key in cddDirKeys:
                    data[key].append('N/A')
            # source views
            for cddDir in cddDirs:
                keys = destructCandidateDirName(cddDir.parts[-1])
                print(cddDir)
                for tex in cddDir.glob('sv*_texture*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('texture')
                    data['group'].append(int(re.findall('sv([0-9]+)', tex.stem)[0]))
                    data['viewId'].append(int(re.findall('sv([0-9]+)_([0-9]+)', tex.stem)[0][1]))
                    data['bytes'].append(tex.stat().st_size)
                    data['path'].append(str(tex))
                    for key in keys:
                        data[key].append(keys[key])
                for dep in cddDir.glob('sv*_depth*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('depth')
                    data['group'].append(int(re.findall('sv([0-9]+)', dep.stem)[0]))
                    data['viewId'].append(int(re.findall('sv([0-9]+)_([0-9]+)', dep.stem)[0][1]))
                    data['bytes'].append(dep.stat().st_size)
                    data['path'].append(str(dep))
                    for key in keys:
                        data[key].append(keys[key])
                for aux in cddDir.glob('aux*.264'):
                    data['scene'].append(scene)
                    data['cddDir'].append(cddDir.parts[-1])
                    data['type'].append('aux')
                    data['group'].append(int(re.findall('aux_([0-9]+)_sv([0-9]+)', aux.stem)[0][1]))
                    data['viewId'].append(int(re.findall('aux_([0-9]+)_sv([0-9]+)_([0-9]+)', aux.stem)[0][2]))
                    data['bytes'].append(aux.stat().st_size)
                    data['path'].append(str(aux))
                    for key in keys:
                        data[key].append(keys[key])
        df = pd.DataFrame(data=data)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(outputCsv)        