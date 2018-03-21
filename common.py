
from pathlib import Path
import numpy as np

# Returns arrays of simple and detailed images
def get_data(spath=Path("../ts.npy"), dpath=Path("../td.npy")):
    return np.load(spath), np.load(dpath)
