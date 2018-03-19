#!/usr/bin/env python3

import sys
from PIL import Image, ImageDraw

if len(sys.argv) != 4:
    raise "Expected arguments: [IMAGE] [CAT] [OUT_IMAGE]"

_, imgp, cat, outp = sys.argv

p = []

with open(cat, "r") as f:
    inp = [int(x) for x in f.readline().split()]
    cnt = inp[0]
    assert(cnt == 9)
    for i in range(9):
        p.append((inp[1 + 2 * i], inp[2 + 2 * i]))

img = Image.open(imgp)

out = Image.new("L", img.size, color=255)
draw = ImageDraw.Draw(out)
weight = ((img.size[0] + img.size[1]) // 50)

draw.line(p[3:10], fill=0, width=weight)

for i in range(3):
    mi = tuple((x - weight // 2) for x in p[i])
    ma = tuple((x + weight // 2) for x in p[i])
    draw.ellipse([mi, ma], fill=0)

out.save(outp)
