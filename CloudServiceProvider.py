from Common import *
from PoseSplitter import *
from CamPlace import *
from Solver import *
from pathlib import Path
from ContentCreator import *
import time

class CloudServiceProvider:
    def __init__(self, settings, psSettings, camSettings, solverSettings):
        '''
        * psSettings: pose splitter settings
        * camSettings: CamPlace settings
        * solverSettings Solver settings
        '''
        self.settings = settings
        self.psSettings = psSettings
        self.camSettings = camSettings
        self.solverSettings = solverSettings
        self.ps = PoseSplitter(psSettings)
        self.camPlace = CamPlace(camSettings)
        self.solver = Solver(solverSettings)
    @classmethod
    def GetDefaultSettings(cls):
        return {
            # policy = '1Prob' assumes that the "correct candidate selection ratio for single candidate" grows linear with load
            # policy = 'IXR'
            'policy': '1Prob',
            'depthThres': None,
            'timeLimit':None,
        }
    def summary(self, outDir: Path):
        with open(outDir/'CloudServiceProvider.json', 'w') as f:
            j = dict(self.settings)
            json.dump(j, f)
        self.ps.summary(outDir)
        self.camPlace.summary(outDir)
        self.solver.summary(outDir)
    def split(self, poses):
        '''
        * poses: (U, P, 7) in unity frame
        '''
        return self.ps.splitPose(poses)
    def candidates(self, poses, indices):
        '''
        * poses: (N, 7) in unity frame
        * indices: (X, 7) returned by self.split
        This func partitions, and generate candidates for each of the partition
        return indices, cdds (M, 7) in unity frame
        '''
        policy = self.settings['policy']
        if policy == '1Prob':
            poses = poses.reshape((-1, 7))
            places = []
            for i in range(indices.shape[0]):
                places.append(self.camPlace.localPlace(poses[indices[i, 0]:indices[i, 1]]))
            return np.array(places)
        elif policy == 'IXR':
            places = []
            for i in range(indices.shape[0]):
                places.append(poses[indices[i]])
            return np.array(poses)
        else:
            return NotImplemented            
    def place(self, cdds, N: int, cc: ContentCreator, callbacks: dict = {}):
        '''
        place the source views
        
        * cdds: (M, 7) candidates returned by self.candidates
        * N: number of views to be selected
        * cc: content creator
        * callbacks:
            {
                "onEsted": f(float), # called on estimated finished
            }        
        return sol (M,) , opt, ub
        '''
        policy = self.settings['policy']
        depthThres = self.settings['depthThres']
        ccSt = cc.getSettings() # a copy of Content Creator settings
        H, W = ccSt['height'], ccSt['width']
        M = cdds.shape[0]
        if policy == '1Prob' or policy == 'IXR':
            # build coverage between cdds
            cddsD = np.array([cc.renderD(cdds[i]) for i in range(M)])
            # ! Assume that the coverage array can fit in
            # C = np.zeros((M, M, H, W), dtype=bool)
            C = []
            tic = time.process_time()
            for i in range(M):
                C.append([])
                eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(cdds[i]))
                rays_i = cc.createRays(eye, center, up)
                d_i = cddsD[i]
                ccSt['obj'] = makeMeshFromRaysDepth(rays_i.numpy(), d_i)
                cc_i = ContentCreator(ccSt)
                for j in range(M):
                    if i == j:
                        # C[i, j] = 1
                        C[-1].append(np.ones((H, W), dtype=bool))
                    else:
                        d_j = cddsD[j]
                        d_ij = cc_i.renderD(cdds[j])
                        # C[i, j] = makeCoverageMap(d_ij, d_j, depthThres)
                        C[-1].append(makeCoverageMap(d_ij, d_j, depthThres))
            C = np.array(C, dtype=bool)
            if "onEsted" in callbacks:
                callbacks["onEsted"](time.process_time() - tic)
            # print(f'Building coverage table takes {time.process_time() - tic} secs')
            # solve
            sol, opt, ub = self.solver.solve(C, N, self.settings['maxNumNodes'], callbacks['solverCallback'])
            return sol, opt, ub
        else:
            return NotImplemented