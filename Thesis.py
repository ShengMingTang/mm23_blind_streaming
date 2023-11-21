from Exp import *
from PoseFeeder import *
from ContentCreator import *
from PoseSplitter import *
from CamPlace import *
from CloudServiceProvider import *
from Solver import *
from pathlib import Path
import json
'''D:\TMM\doc_TMM22_Code\Trace_FPS50_LEN30\Trace_FPS50_LEN30_sFurnishedCabin_rtR4_ffrOff_maxNodes96_dep0.01_slvrC2I_a1e-05_cdd_csp1Prob_w50_puniform_m0.03_h0.15_plaverage'''
'''D:\TMM\Trace_FPS50_LEN30\Trace_FPS50_LEN30_sFurnishedCabin_rtR4_ffrOff_maxNodes48_dep0.01_slvrUnB_a1e-05_cdd_csp1Prob_w50_puniform_m0.03_h0.15_plaverage'''

paths = [
    # Path('D:\TMM\doc_TMM22_Code\Trace_FPS50_LEN30\Trace_FPS50_LEN30_sFurnishedCabin_rtR4_ffrOff_maxNodes96_dep0.01_slvrC2I_a1e-05_cdd_csp1Prob_w50_puniform_m0.03_h0.15_plaverage'),
    # Path('D:\TMM\Trace_FPS50_LEN30\Trace_FPS50_LEN30_sFurnishedCabin_rtR4_ffrOff_maxNodes48_dep0.01_slvrUnB_a1e-05_cdd_csp1Prob_w50_puniform_m0.03_h0.15_plaverage'),
    Path('D:\TMM\Trace_FPS50_LEN30\Trace_FPS50_LEN30_sFurnishedCabin_rtR4_ffrOff_maxNodes96_dep0.01_slvrC2G_a1e-05_cdd_csp1Prob_w50_puniform_m0.03_h0.15_plaverage')
]
saveJsons = [
    # 'distrC2I.json',
    # 'distrUnB.json',
    'distrC2G.json'
]
counts = [{}, {}]
H, W = 540, 960

# PoseFeeder
pfdSt = PoseFeeder.GetDefaultSettings()
pfdSt['poseDir'] = Path('D:\TMM\Trace_FPS50_LEN30\FurnishedCabin')
pfdSt['size'] = 50
pfd = PoseFeeder(pfdSt)
U = next(iter(pfd)).shape[0]
P = next(iter(pfd)).reshape((-1, 7)).shape[0]
M = 48
depthThres = 1e-2
# ContentCreator
ccSt = ContentCreator.GetDefaultSettings()
ccSt['height'] = H
ccSt['width'] = W
ccSt['obj'] = Path('D:\TMM\Trace_FPS50_LEN30\FurnishedCabin\scene.obj')
cc = ContentCreator(ccSt)
# CloudServiceProvider
cspSt = CloudServiceProvider.GetDefaultSettings()
cspSt['depthThres'] = depthThres
cspSt['maxNumNodes'] = 0
psSt = PoseSplitter.GetDefaultSettings()
psPolicy = 'uniform'
psSt['policy'] = psPolicy
psSt[psPolicy]['numPart'] = M
camSt = CamPlace.GetDefaultSettings()
camSt['policy'] = 'average'
slvrSt = Solver.GetDefaultSettings()
slvrSt['a'] = np.log(1e-5)
slvrSt['ffrMask'] = 1.0
slvrSt['policy'] = 'UnB'
csp = CloudServiceProvider(cspSt, psSt, camSt, slvrSt)

sample = 0

for i, p in enumerate(paths):
    with open(str(p/'Exp.json'), 'r') as f:
        expJ = json.load(f)
    sols = np.array(expJ['sols'], dtype=bool)
    counts = {}
    iterMax = 1
    for j, poses in enumerate(pfd):
        # if iterMax == 0:
        #     break
        # else:
        #     iterMax -= 1
        print(f'group {j}')
        indices = csp.split(poses)
        cdds = csp.candidates(poses, indices)
        cddsD = np.array([cc.renderD(cdds[_]) for _ in range(M)])
        C = np.zeros((H, W), dtype=int)
        for k in range(1, cdds.shape[0]):
            if sols[j][k] == True:
                print(f'    sol {k}')
                eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(cdds[k]))
                rays_i = cc.createRays(eye, center, up)
                d_i = cddsD[i]
                d_j = cddsD[sample]
                ccSt['obj'] = makeMeshFromRaysDepth(rays_i.numpy(), d_i)
                cc_i = ContentCreator(ccSt)
                d_ij = cc_i.renderD(cdds[sample])
                # C[i, j] = makeCoverageMap(d_ij, d_j, depthThres)
                C_ij = makeCoverageMap(d_ij, d_j, depthThres)
                C += C_ij
        uni = np.unique(C)
        for c in uni:
            if (c in counts) == False:
                counts[int(c)] = 0
            counts[int(c)] += int(np.sum(C == c))
    with open(saveJsons[i], 'w') as f:
        json.dump(counts, f)