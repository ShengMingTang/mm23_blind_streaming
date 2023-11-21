'''
This file is not maintained
'''
from miv_util import *
import re
import json

f2d = fDepthPlannarFactory(1000)
for p in Path('Trace_Raw').glob('*'):
    truncatePoseDir2DirFolded(p, Path('Trace_Processed')/p.name, -50*30, None, 4)