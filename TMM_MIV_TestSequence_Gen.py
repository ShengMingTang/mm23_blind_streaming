import numpy as np
from pathlib import Path
import argparse
'''
Use to generate verification sequence for miv renderer
Split pose only

Require single user pose csv,
pick some of the poses as the source views,
another some of them as target views,
source/target views are non-overlapping
'''
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, help='test sequence directory')
args = parser.parse_args()

longPoseCsv = Path(args.dir)/'pose0.csv'
longPose = np.loadtxt(longPoseCsv, delimiter=',', skiprows=1)
shortPose = longPose[::50]
userDir = Path(args.dir)/'TargetView'
userDir.mkdir(exist_ok=True)
np.savetxt(str(userDir/f'pose0.csv'), shortPose, fmt='%.4f', delimiter=',', header='t,x,y,z,qx,qy,qz,qw', comments='')

svDir = Path(args.dir)/'SourceView'
svDir.mkdir(exist_ok=True)
shortPose = longPose[::25]
for i in range(shortPose.shape[0]):
    np.savetxt(str(svDir/f'sv{i}.csv'), shortPose[i].reshape((-1, 8)), delimiter=',', header='t,x,y,z,qw,qx,qy,qz', comments='', fmt='%.4f')

