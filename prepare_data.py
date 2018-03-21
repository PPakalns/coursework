#!/usr/bin/env python3

import sys
from pathlib import Path
from PIL import Image, ImageDraw
from typing import List, Tuple
import numpy as np

CatPoints = List[Tuple[int, int]]

# Read cat file from dataset
def readcat(catPath):
    p = []
    with open(catPath, "r") as f:
        inp = [int(x) for x in f.readline().split()]
        cnt = inp[0]
        assert(cnt == 9)
        for i in range(1, 1 + 2 * 9, 2):
            p.append((inp[i], inp[i + 1]))
    return p


# Returns drawed simple Image
def gen_img(imgfile: Path, p:CatPoints):
    with Image.open(imgfile) as img:
        out = Image.new("L", img.size, color=255)
        draw = ImageDraw.Draw(out)
        weight = (sum(img.size) // 50)

        draw.line(p[3:10], fill=0, width=weight)

        hweight = weight // 2
        for i in range(3):
            mi = tuple((x - hweight) for x in p[i])
            ma = tuple((x + hweight) for x in p[i])
            draw.ellipse([mi, ma], fill=0)
    return out


def resizeImage(img:Image, p:CatPoints, size):
    # Center on nose
    nx, ny = p[2]

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
    return img


def prepare_images(sizes:int, sized:int, path:Path, outs:Path, outd:Path):
    files = []
    for entry in path.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".jpg":
            continue
        files.append(entry)
    files.sort()

    print(f"Processing {len(files)} files from {str(path)} directory!")

    processed = 0
    sdata, ddata = [], []
    for file in files:
        processed += 1
        if (processed & 0x3FF) == 0:
            print(f"\t{processed} / {len(files)}")
        file:Path
        catpoints = readcat(file.parent / (file.name + ".cat"))
        simpleimg = gen_img(file, catpoints)
        simpleimg = resizeImage(simpleimg, catpoints, sizes)
        with Image.open(file) as img:
            detailimg = resizeImage(img, catpoints, sized)
        sdata.append(np.asarray(simpleimg, dtype="int32"))
        ddata.append(np.asarray(detailimg, dtype="int32"))
    print("Images processed")
    np.save(outs, sdata)
    np.save(outd, ddata)
    print(f"Images saved as numpy arrays simple:{str(outs)}, detailed:{str(outd)}")

if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise "Expected arguments: [size simple] [size detailed] [DIR] [out simple] [out detailed]"
    sizes, sized, path, outs, outd = sys.argv
    prepare_images(sizes, sized, Path(path), Path(outs), Path(outd))

