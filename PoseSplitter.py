import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json
from Common import *
from ContentCreator import *

'''
angle mean: https://rosettacode.org/wiki/Averages/Mean_angle
'''

class PoseSplitter():
    @classmethod
    def GetDefaultSettings(cls):
        return {
            # policy = 'uniform' | 'fixedNumber' | 'motion'
            'policy': 'uniform',
            'uniform': {
                'numPart': None, # number of partitions, must be set
            },
            'IXR': {
                'thres': 0.75, # theshold for yield a partition
                'ccSt': None,
                'depthThres': None,
            },
            'fixedNumber': { # dp
                'numPart': None, # number of partitions, must be set
            },
            # motion-base settngs
            'motion': {
                'posThres': [0.1**2, 0.1**2, 0.1**2, 0.1], # [xThres, yThres, zThres, captured determinant]
                'rotThres': 5, # angle of rotation from a quaternion to the other, in degree (relared)
                'minSample': 5, # grab this amount then collect more
            },
            # ! force to use experimental modes
            "force": False,
        }
    def __init__(self, settings):
        '''
        '''
        if settings['policy'] == 'motion':
            settings['motion']['rotThres'] = (1 - np.cos(settings['motion']['rotThres'] * np.pi/180))/2
        self.settings = settings
    def summary(self, outDir: Path):
        outDir.mkdir(exist_ok=True)
        with open(outDir/'PoseSplitter.json', 'w') as f:
            ccSt = self.settings['IXR']['ccSt']
            self.settings['IXR'].pop('ccSt')
            json.dump(self.settings, f)
            self.settings['IXR']['ccSt'] = ccSt
    def splitPose(self, pose):
        '''
        motion:
            * pose: (N, 7), each = (x, y, z, qw, qx, qy, qz)
            return:
                np.array of (M, 2), # each row is [startIdx, endIdx), M = number of segments
        uniform | fixedNumber:
            * pose: (U, N, 7), same as output of PoseFeeder
            return:
                np.array of (M, 2), # each row is [startIdx, endIdx) indexing U axis, M = number of segments
        IXR:
            * pose: (U, N, 7), same as output of PoseFeeder
            return:
                np.array of (M,) # each is index of the pose
        '''
        if self.settings['policy'] == 'motion':
            if self.settings['force'] != True:
                return NotImplemented
            pose = pose.reshape((-1, 7))
            splitted = []
            vars = []
            idx = 0
            minSample = self.settings['motion']['minSample']
            criterion = np.array([*self.settings['motion']['posThres'], self.settings['motion']['rotThres']])
            criterion = np.square(criterion)
            while(idx + minSample < pose.shape[0]):
                cond = False
                tail = idx
                while cond == False and tail < pose.shape[0]:
                    tail = min(tail + minSample, pose.shape[0])
                    p = pose[idx:tail, :3]
                    x = np.arange(idx, tail)
                    lsef = [makeLSE(x, p[:, i]) for i in range(p.shape[1])]
                    ypred = np.array([f(x) for f in lsef])
                    yerr = np.square(p.T - ypred)
                    yMeanErr = np.mean(yerr, axis=1)
                    thres = yMeanErr.tolist()
                    thres.append(np.abs(np.prod(yMeanErr)))
                    
                    q = R.from_quat(pose[idx:tail, 3:])
                    q2 = R.from_quat(pose[[idx, tail - 1], 3:])
                    slerp = Slerp([0, 1], q2)
                    # measure rotation from one q to another
                    # https://math.stackexchange.com/questions/90081/quaternion-distance
                    qPred = slerp(np.linspace(0, 1, num=q.as_quat().shape[0]))
                    inner = np.sum(qPred.as_quat() * q.as_quat(), axis=1)
                    theta = 1 - np.square(inner)
                    theta = np.mean(theta)
                    thres.append(theta)
                    cond = np.sum(thres > criterion) > 0
                # cond has one truth, we should yield this fragment
                splitted.append([idx, tail])
                vars.append(thres)
                idx = tail
                
            # collect the last fragment
            if idx < pose.shape[0]:
                splitted.append([idx, pose.shape[0]])
            return np.array(splitted, dtype=int)
        elif self.settings['policy'] == 'uniform':
            u, p = pose.shape[0], pose.shape[1]
            longPoses = np.array(pose).reshape((-1, 7))
            partSize = p / (self.settings['uniform']['numPart'] // u)
            indices = []
            # ! cannot index by partSize directly to avoid partition across users
            for i in range(u): # the ith user
                j = 0
                cnt = 0
                while round(j) < p:
                    idx = round(j)
                    indices.append([p*i + round(j), min(p*i + round(j + partSize), p*(i+1))])
                    j += partSize
                    cnt += 1
                assert cnt == self.settings['uniform']['numPart'] // u
                # for j in range(p * i, p * (i + 1), partSize):
                #     indices.append([j, min(p * (i+1), j + partSize)])
            return np.array(indices)
        elif self.settings['policy'] == 'fixedNumber':
            '''
            https://stackoverflow.com/questions/26949246/a-strategy-to-partition-a-set-to-get-the-minimum-sum-of-variances-from-subsets
            '''
            return NotImplemented
            def partCost(part):
                ''' part: (3) '''
                return np.sqrt(np.sum(np.square(part)))
            longPoses = np.array(pose).reshape((-1, 7))
            numPart = self.settings['fixedNumber']['numPart']
            # position
            xP = longPoses[:, :3]
            uP = np.zeros((longPoses.shape[0] + 1, 3)) # (N, [mean_x, mean_y, mean_z])
            vP = np.zeros((longPoses.shape[0] + 1, 3)) # [var_x, var_y, var_z]
            xQ = R.from_quat(longPoses[:, 3:]).as_euler('xyz').astype(np.complex128) # orientation
            uQ = np.ones((longPoses.shape[0] + 1, 3), dtype=np.complex128) # (N, [mean_roll, mean_pitch, mean_yaw])
            vQ = np.ones((longPoses.shape[0] + 1, 3), dtype=np.complex128) # [var_roll, var_pitch, var_yaw]
            # build up the table
            for i in range(1, xP.shape[0] + 1):
                uP[i, :] = scalarMeanUpdate(xP[i - 1, :], uP[i - 1, :], i)
                vP[i, :] = scalarVarUpdate(vP[i-1, :], uP[i, :], uP[i-1, :], i)
                uQ[i, :] = angularMeanUpdate(xQ[i - 1, :], uQ[i - 1, :], i)
                vQ[i, :] = angularVarUpdate(vQ[i-1, :], uQ[i, :], uQ[i-1, :], i)
            # dpP[c, p] = remain c cuts to drop, finishes pose p
            dpP = np.zeros((np.around(numPart/2).astype(np.int) - 1, longPoses.shape[0]))
            solP = np.zeros((np.around(numPart/2).astype(np.int) - 1, longPoses.shape[0]), dtype=np.int)
            dpP.fill(np.inf)
            dpP[0, :] = np.max(vP[1:, :], axis=1)
            dpP[:, 0] = 0
            for i in range(1, dpP.shape[0]):
                for j in range(1, dpP.shape[1]):
                    for k in range(j):
                        v = max(dpP[i-1, k], np.max(scalarVarMN(vP, uP, k + 1, j + 1)))
                        if v < dpP[i, j]:
                            dpP[i, j] = v
                            solP[i, j] = k + 1
            indicesP = [dpP.shape[1]]
            m, n = dpP.shape[0] - 1, dpP.shape[1] - 1
            while m != 0:
                indicesP.append(solP[m, n])
                m, n = m - 1, solP[m, n] - 1
            # print(dpP)
            # print(indicesP)
            # import matplotlib.pyplot as plt
            # plt.imshow(dpP)
            # plt.show()
            
            # orientation
            
            indicesP = np.array(indicesP[::-1])
            return indicesP
        elif self.settings['policy'] == 'IXR':
            u, p = pose.shape[0], pose.shape[1]
            longPoses = np.array(pose).reshape((-1, 7))
            thres = self.settings['IXR']['thres']
            ccSt = self.settings['IXR']['ccSt']
            depthThres = self.settings['IXR']['depthThres']
            ccProx = ContentCreator(ccSt)
            tail = 0
            indices = []
            eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(longPoses[tail]))
            rays_tail = ccProx.createRays(eye, center, up)
            d_tail = ccProx.renderD(longPoses[tail])
            ccSt['obj'] = makeMeshFromRaysDepth(rays_tail.numpy(), d_tail)
            cc_tail = ContentCreator(ccSt)
            for head in range(longPoses.shape[0]):
                d_tail_head = cc_tail.renderD(longPoses[head])
                d_head = ccProx.renderD(longPoses[head])
                cvg = np.mean(makeCoverageMap(d_tail_head, d_head, depthThres))
                # ensure every user at least generate a partition
                if cvg < thres or (head > 0 and (head % p == 0)):
                    indices.append([tail, head])
                    tail = head
                    eye, center, up = convertO3d7ToO3dEyeCenterUp(convertUnityPoses7ToO3d7(longPoses[tail]))
                    rays_tail = ccProx.createRays(eye, center, up)
                    d_tail = ccProx.renderD(longPoses[tail])
                    ccSt['obj'] = makeMeshFromRaysDepth(rays_tail.numpy(), d_tail)
                    cc_tail = ContentCreator(ccSt)
                    print(f'partition {indices[-1]}')
            # append last to make it closed
            indices.append([tail, longPoses.shape[0]])
            print(f'partition {indices[-1]}')
            return np.array(indices)
        else:
            return NotImplemented
