import numpy as np
import re
from pathlib import Path
import json

class PoseFeeder:
    '''
        This may serve as pose predicter in the future
        unity raw pose are x, y, z, qx, qy, qz, qw (scalar last)
    '''
    @classmethod
    def GetDefaultSettings(cls):
        return {
            'size': 50,
            'poseDir': None,
        }
    def __init__(self, settings: dict) -> None:
        '''
        * poseDir: directory contains pose*.csv
        '''
        self.settings = settings
        fns = list(self.settings['poseDir'].glob('pose*.csv'))
        fns = sorted(fns, key=lambda x: int(re.findall('pose(.*).csv', x.name)[0]))
        self.fns = fns
        if len(self.fns) == 0:
            raise IOError('0 file loaded at PoseFeeder')
        self.poses = np.array([np.loadtxt(fn, skiprows=1, delimiter=',') for fn in self.fns])
    def summary(self, outDir: Path):
        outDir.mkdir(exist_ok=True)
        with open(outDir/'PoseFeeder.json', 'w') as f:
            j = dict(self.settings)
            j['poseDir'] = str(j['poseDir'])
            j['poseFns'] = [str(fn) for fn in self.fns]
            json.dump(j, f)
    def __iter__(self):
        idx = 0
        sz = self.settings['size']
        while idx < self.poses.shape[1] // sz:
            yield self.poses[:, idx * sz : (idx + 1) * sz, 1:]
            idx += 1