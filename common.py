
from pathlib import Path
import numpy as np
import functools
import random

# Returns arrays of simple and detailed images
def get_data(spath=Path("../ts.npy"), dpath=Path("../td.npy")):
    return np.load(spath), np.load(dpath)

def get_random_cut(shape, size):
    def do_random_cut(imgs, cut, size):
        return imgs[:,cut[0]:cut[0]+size,cut[1]:cut[1]+size,:]
    def X(imgs):
        return imgs
    if shape[1] == size and shape[2] == size:
        return X
    cut_shape = (random.randint(0, shape[1] - size), random.randint(0, shape[2] - size))
    return functools.partial(do_random_cut, cut=cut_shape, size=size)
