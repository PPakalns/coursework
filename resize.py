#!/usr/bin/env python3

import sys
import common
from PIL import Image

if len(sys.argv) != 5:
    raise "Expected arguments: [SIZE] [IMAGE] [CAT] [OUT_IMAGE]"

_, size, imgp, catp, outp = sys.argv

size = int(size)
p = common.readcat(catp)

# Center on nose
nx, ny = p[2]

img = Image.open(imgp)
w, h = img.size

ms = min(w, h)
msdiv2 = ms // 2
nx, ny = nx + msdiv2, ny + msdiv2
nx, ny = min(nx, w), min(ny, h)
nx, ny = nx - ms, ny - ms
nx, ny = max(nx, 0), max(ny, 0)

# Crop box
crop = (nx, ny, nx + ms, ny + ms)

img = img.crop(crop)
img = img.resize((size, size))
img.save(outp)
