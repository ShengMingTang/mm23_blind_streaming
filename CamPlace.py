import numpy as np
from Common import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json
from pathlib import Path

class CamPlace:
    '''
    Convert (-1, 7) a group of poses in Unity frame to a source view candidates (7,)
    '''
    def __init__(self, settings) -> None:
        self.settings = settings
    @classmethod
    def GetDefaultSettings(cls) -> dict:
        return {
            # local placement policy = 'interpolation' | 'average' | 'backMost' | 'IXR'
            # 'interpolation': interprete the group of poses as a linear function of position and orientation from [0, 1], outputting position and orientation at 0.5
            # 'average': return average position and orientation
            # 'backMost': Intended to output the back-most pose relative to all poses in that group (under dev)
            # 'IXR': return the last pose
            'policy': 'average',
        }
    def summary(self, outDir: Path):
        outDir.mkdir(exist_ok=True)
        with open(outDir/'CamPlace.json', 'w') as f:
            j = dict(self.settings)
            json.dump(j, f)
    def localPlace(self, splitedPose: np.array) -> np.array:
        '''
        * splitedpose: (M, 7) in Unity frame (actually any frame for motion only placement)
        
        Average quaternions
        http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
        '''
        if self.settings['policy'] == 'interpolation':
            ret = np.zeros((7,))
            ret[:3] = np.mean(splitedPose[:, :3], axis=0)
            q = R.from_quat(splitedPose[[0, -1], 3:])
            slerp = Slerp([0, 1], q)
            ret[3:] = slerp(0.5).as_quat()
            return ret
        elif self.settings['policy'] == 'average':
            ret = np.zeros((7,))
            ret[:3] = np.mean(splitedPose[:, :3], axis=0)
            q = splitedPose[:, 3:]
            M = np.array([np.outer(q[i], q[i]) for i in range(q.shape[0])])
            M = np.average(M, axis=0)
            qw, qv = np.linalg.eig(M)
            qu = qv[:, np.argmax(qw)]
            ret[3:] = qu
            # ! complex type often result in a + 0j, which is safe to cast to real number
            # try:
            #     assert qu.dtype != np.complex128
            # except AssertionError:
            #     print(f'Compldex eigenvector {qu}')
            return ret
        elif self.settings['policy'] == 'backMost':
            return NotImplemented
            ret = np.zeros((7,))
            # use rotation average
            q = splitedPose[:, 3:]
            M = np.array([np.outer(q[i], q[i]) for i in range(q.shape[0])])
            M = np.average(M, axis=0)
            qw, qv = np.linalg.eig(M)
            qu = qv[:, np.argmax(qw)]
            ret[3:] = qu
            # use back-most pose position
            initFront = np.array([0, 0, 1], dtype=np.float32) # initial front (0, 0, 1) in Unity frame
            front = R.from_quat(qu).apply(initFront)
            pos = splitedPose[:, :3].copy()
            inners = np.zeros((pos.shape[0]))
            for i in range(inners.shape[0]):
                pose = pos[i].copy()
                pos = pos - pose
                inner = np.sum(np.dot(pos, front))
                inners[i] = inner
                pos = pos + pose
            idx = np.argmax(inners)
            ret[:3] = pos[idx]
            return ret
        elif self.settings['policy'] == 'IXR':
            return splitedPose[-1]
        else:
            return NotImplemented
        