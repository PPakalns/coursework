#!/usr/bin/env python3

import sys
import common
from PIL import Image, ImageDraw

if len(sys.argv) != 4:
    raise "Expected arguments: [IMAGE] [CAT] [OUT_IMAGE]"

_, imgp, cat, outp = sys.argv

p = common.readcat(cat)

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
