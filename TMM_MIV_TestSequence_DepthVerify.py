'''
Verify depth shader
Use Depth Plannar in TMM_Unity_VR/Assets/Shader/*.shader
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rgb', type=str, help='path to rgb (*.png)')
parser.add_argument('-d', '--dep', type=str, help='path to depth (*.raw)')
args = parser.parse_args()

rgb = Path(args.rgb)
d = Path(args.dep)

img = cv2.imread(str(rgb), cv2.IMREAD_UNCHANGED)
data = np.fromfile(d, dtype=np.float32)
if img is None:
    print('img not loaded')
if data is None:
    print('binary not loaded')

print(f'Number of unique value in depth {np.unique(data[..., 0]).shape}')
data = data.reshape((*img.shape[:2], 4))
data = data[..., :3] * 255 # for visualization

plt.subplot(121)
plt.imshow(data[::-1, ...])
plt.subplot(122)
plt.imshow(img)
plt.show()