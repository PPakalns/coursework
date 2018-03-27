
import cnn
from cnn import CNN

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import Activation, Dropout, Input, Concatenate
from keras.layers import UpSampling2D, Reshape, Flatten, BatchNormalization

class GAN(CNN):

    def __init__(self, size=64, gray=False):
        CNN.__init__(self, size=size, gray=gray)
        self.size = size
        self.depth = 4
        self.D = None

    def discriminator(self, silent=False):
        if self.D:
            if not silent:
                self.D.summary()
            return self.D

        depth = self.depth

        inp = Input((self.size, self.size, 1 if self.gray else 3))
        d1 = cnn.downsample(inp, depth)     # 32
        d2 = cnn.downsample(d1, depth * 2)  # 16
        d3 = cnn.downsample(d2, depth * 4)  # 8
        d4 = cnn.downsample(d3, depth * 6)  # 4

        m0 = Flatten()(d4)
        m1 = Dense(self.size)(m0)
        m2 = Dropout(0.1)(m1)
        m3 = LeakyReLU(0.1)(m2)
        b0 = BatchNormalization(momentum=0.8)(m3)

        m4 = Dense(16)(b0)
        m5 = Dropout(0.1)(m4)
        m6 = LeakyReLU(0.1)(m5)

        o0 = Dense(1)(m6)
        o1 = Activation('sigmoid')(o0)

        self.D = Model(inp, o1)
        if not silent:
            self.D.summary()
        return self.D

    def train(self, epochs, batch=32, save = 150):
        self.load_data()

        if self.G is None:
            raise "Initialize generator"

        if self.D is None:
            raise "Initialize discriminator"

        optimizer = Adam()

        self.D.trainable = True

        self.D.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        simg = Input((self.size, self.size, 1 if self.gray else 3))

        self.D.trainable = False
        self.combined = Model(simg, self.D(self.G(simg)))

        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

        for epoch in range(1, 1 + epochs):

            # Discriminator
            idxs = np.random.randint(0, self.ddata.shape[0], batch // 2)
            dimgs = self.ddata[idxs]

            idxs = np.random.randint(0, self.sdata.shape[0], batch // 2)
            simgs = self.sdata[idxs]
            gimgs = self.G.predict(simgs)

            d_inp = np.concatenate((dimgs, gimgs))
            d_out = np.concatenate((np.ones((batch // 2,)), np.zeros((batch // 2,))))
            d_out = np.reshape(d_out, (*d_out.shape, 1))

            d_loss = self.D.train_on_batch(d_inp, d_out)

            # Generator
            idxs = np.random.randint(0, self.sdata.shape[0], batch)
            g_inp = self.sdata[idxs]
            g_out = np.ones((batch, 1))
            g_loss = self.combined.train_on_batch(g_inp, g_out)

            print(f"{epoch}\tD: [loss: {d_loss[0]:.2f} acc: {d_loss[1]:.2f}]\tG: [loss: {g_loss:.2f}]")

            if epoch % save == 0 or epoch == epochs:
                timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.G.save(f"../model_{timenow}_G.h5")
                self.D.save(f"../model_{timenow}_D.h5")

                self.show_img(epoch)

    def show_img(self, epoch):
        cnt = 3
        idxs = np.random.randint(0, self.sdata.shape[0], cnt)
        simg = self.sdata[idxs]
        dimg = self.ddata[idxs]
        gimg = self.G.predict(simg)

        # Rescale images 0 - 1
        simg = simg * 0.5 + 0.5
        gimg = gimg * 0.5 + 0.5

        fig, axs = plt.subplots(cnt, 3)
        for i in range(cnt):
            axs[i,0].imshow(simg[i,:,:,0], cmap='gray')
            axs[i,0].axis('off')
            show_gimg = gimg[i,:,:,0] if self.gray else gimg[i,:,:,:]
            axs[i,1].imshow(show_gimg, cmap='gray' if self.gray else None)
            axs[i,1].axis('off')
            show_dimg = dimg[i,:,:,0] if self.gray else dimg[i,:,:,:]
            axs[i,2].imshow(show_dimg, cmap='gray' if self.gray else None)
            axs[i,2].axis('off')
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.savefig(f"../gan_{epoch}_{timenow}.png")
        plt.show()


