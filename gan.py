
import cnn
from cnn import CNN

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import Activation, Dropout, Input, Concatenate, GaussianNoise
from keras.layers import UpSampling2D, Reshape, Flatten, BatchNormalization

class GAN(CNN):

    def __init__(self, size=64, gray=False, gen_depth=16, dis_depth=2):
        CNN.__init__(self, size=size, gray=gray, gen_depth=gen_depth)
        self.dis_depth = dis_depth
        self.D = None

    def discriminator(self, silent=False):
        if self.D:
            if not silent:
                self.D.summary()
            return self.D

        depth = self.dis_depth

        inp = Input((self.size, self.size, (1 if self.gray else 3) + 1))
        d1 = cnn.downsample(inp, depth)     # 32
        d2 = cnn.downsample(d1, depth * 2)  # 16
        d3 = cnn.downsample(d2, depth * 4)  # 8
        d4 = cnn.downsample(d3, depth * 6)  # 4

        m0 = Flatten()(d4)
        m1 = Dense(self.size)(m0)
        m2 = LeakyReLU(0.1)(m1)
        b0 = BatchNormalization(momentum=0.8)(m2)

        m3 = Dropout(0.2)(b0)

        m4 = Dense(16)(m3)
        m5 = LeakyReLU(0.1)(m4)
        b1 = BatchNormalization(momentum=0.8)(m5)
        m6 = Dropout(0.2)(b1)

        o0 = Dense(1)(m6)
        o1 = Activation('sigmoid')(o0)

        self.D = Model(inp, o1)
        if not silent:
            self.D.summary()
        return self.D

    def train(self, epochs, batch=32, save = 150):
        self.load_data()

        if (batch % 4 != 0):
            raise "Batch size must be multiple of 4"

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

        self.D.trainable = False

        simg = Input((self.size, self.size, 1))
        gimg = GaussianNoise(0.5)(simg) # Add noise to input simple image
        discr = self.D(Concatenate()([self.G(gimg), gimg]))

        self.combined = Model(simg, discr)

        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )

        for epoch in range(1, 1 + epochs):

            # Discriminator
            idxs = np.random.randint(0, self.ddata.shape[0], batch // 2)
            good_inp = np.concatenate([self.ddata[idxs], self.sdata[idxs]], axis=3)

            idxs = np.random.randint(0, self.sdata.shape[0], batch // 4)
            simgs = self.sdata[idxs]
            gimgs = self.G.predict(simgs)
            bad_inp_1 = np.concatenate([gimgs, simgs], axis=3)

            idxs_1 = np.random.randint(0, self.ddata.shape[0], batch // 4)
            idxs_2 = np.random.randint(0, self.ddata.shape[0], batch // 4)
            simgs = self.sdata[idxs_1]
            dimgs = self.ddata[idxs_2]
            bad_inp_2 = np.concatenate([dimgs, simgs], axis=3)
            bad_inp = np.concatenate([bad_inp_1, bad_inp_2], axis=0)

            d_inp = np.concatenate((good_inp, bad_inp))
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


