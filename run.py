#!/usr/bin/env python3.6

import sys

import matplotlib as mpl
mpl.use('Agg')

import gan
import keras
import time
g = gan.GAN(gray=False, size=128, dis_depth=64, gen_depth=128)

g.generator(silent=True)
g.discriminator(silent=True)

if len(sys.argv) > 1:
    c = 3 if len(sys.argv) <= 2 else int(sys.argv[2])
    if c & 1:
        g.G = keras.models.load_model(f"../model_{sys.argv[1]}_G.h5")
        g.G.summary()
        print("GENERATOR LOADED")
        time.sleep(1)
    else:
        g.generator(silent=True)

    if c & 2:
        g.D = keras.models.load_model(f"../model_{sys.argv[1]}_D.h5")
        g.D.summary()
        print("DISCRIMINATOR LOADED")
        time.sleep(1)
    else:
        g.discriminator(silent=True)

    print("MODELS LOADED")
    time.sleep(1)

try:
    g.train(1000000, batch=16, save=50, label_flipping=0.01)
finally:
    g.save()

