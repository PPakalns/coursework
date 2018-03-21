#!/usr/bin/env python3

import sys
from pathlib import Path, PosixPath
from PIL import Image
import numpy as np

if len(sys.argv) != 3:
    raise "Expected arguments: [DIR] [OUT_FILE]"

dirPath = Path(sys.argv[1])
outp = sys.argv[2]
files = []

for entry in dirPath.iterdir():
    if not entry.is_file:
        continue
    files.append(entry)

files.sort()

data = []

print(f"Processing {len(files)} images from {str(dirPath)}, saving to {outp}")

cnt, tcnt = 0, 0

for imgPath in files:
    cnt += 1
    tcnt += 1
    if cnt == 1000:
        print(f"\tProcessed: {tcnt}/{len(files)}")
        cnt = 0
    img = Image.open(imgPath)
    img.load()
    data.append(np.asarray(img, dtype="int32"))
data = np.asarray(data)
print(f"Shape {data.shape}")
np.save(outp, data)
print("DONE")

