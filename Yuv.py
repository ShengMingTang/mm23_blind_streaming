import numpy as np
import re
import os

PATTERN='(.*)_(texture|depth)_([0-9]*)x([0-9]*)_(.*)p(.*)le.yuv'
class Yuv:
    def __init__(self, fn) -> None:
        self.fn = str(fn)
        tokens = re.findall(PATTERN, self.fn)[0]
        self.W = int(tokens[2])
        self.H = int(tokens[3])
        self.pixFmt = tokens[4]
        self.bitDepth = int(tokens[5])
        self.step = self.W * self.H * 3
        self.f = open(self.fn, 'rb')
        assert tokens[1] == 'texture' or tokens[1] == 'depth'
        assert self.pixFmt == 'yuv420'
        assert self.bitDepth == 10 or self.bitDepth == 16
        assert self.f != None
    def __len__(self):
        size = self.f.seek(0, os.SEEK_END)
        return size // self.step
    def __getitem__(self, key)->bytes:
        assert type(key) == int
        self.f.seek(key * self.step, os.SEEK_SET)
        return self.f.read(self.step)