
import cnn
from cnn import CNN

import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import Activation, Dropout, Input, Concatenate, GaussianNoise
from keras.layers import UpSampling2D, Reshape, Flatten

class GAN(CNN):

    def __init__(self, size=64, gray=False, gen_depth=16, dis_depth=2):
        CNN.__init__(self, size=size, gray=gray, gen_depth=gen_depth)
        self.dis_depth = dis_depth
        self.D = None
        self.combined = None

    def discriminator(self, silent=False):
        if self.D:
            if not silent:
                self.D.summary()
            return self.D

        depth = self.dis_depth

        inp = Input((self.size, self.size, (1 if self.gray else 3) + 1))
        d1 = cnn.downsample(inp, depth, norm=False)     # 32
        d2 = cnn.downsample(d1, depth * 2)  # 16
        d3 = cnn.downsample(d2, depth * 4)  # 8
        d4 = cnn.downsample(d3, depth * 8)  # 4
        d4 = cnn.downsample(d4, depth * 8)  # 2

        m0 = Flatten()(d4)

        o0 = Dense(1)(m0)
        o1 = Activation('sigmoid')(o0)

        self.D = Model(inp, o1)
        if not silent:
            self.D.summary()
        return self.D

    def train(self, epochs, batch=32, save = 150, label_flipping=0.1):
        self.load_data()

        if (batch % 2 != 0):
            raise "Batch size must be multiple of 2"

        if self.G is None:
            raise "Initialize generator"

        if self.D is None:
            raise "Initialize discriminator"

        if self.combined is None:
            D_optimizer = Adam(0.0002, 0.5, 0.999)
            G_optimizer = Adam(0.0002, 0.5, 0.999)

            self.D.trainable = False
            self.G.trainable = True

            simg = Input((self.size, self.size, 1))
            generated_image = self.G(simg)
            discr = self.D(Concatenate()([generated_image, simg]))

            self.combined = Model(inputs=simg, outputs=[generated_image, discr])

            self.combined.compile(
                loss=['mae', 'binary_crossentropy'],
                loss_weights=[20, 1],
                optimizer=G_optimizer
            )

            self.combined.summary()

            self.D.trainable = True
            self.G.trainable = False
            
            ssimg = Input((self.size, self.size, 1))
            ggimg = self.G(ssimg)
            ddd = self.D(Concatenate()([ggimg, ssimg]))
            self.combinedRev = Model(inputs=ssimg, outputs=[ddd])

            self.combinedRev.compile(
                loss='binary_crossentropy',
                optimizer=D_optimizer,
                metrics=['binary_accuracy']
            )
            
            self.D.compile(
                loss='binary_crossentropy',
                optimizer=D_optimizer,
                metrics=['binary_accuracy']
            )


        print(self.combinedRev.metrics_names, self.combined.metrics_names)
        for epoch in range(1, 1 + epochs):

            # Discriminator
            idxs = np.random.randint(0, self.ddata.shape[0], batch // 2)
            good_inp = np.concatenate([self.ddata[idxs], self.sdata[idxs]], axis=3)

            idxs = np.random.randint(0, self.sdata.shape[0], batch // 2)
            bad_inp = self.sdata[idxs]

            if label_flipping is not None:
                d_good = np.random.binomial(1, 1-label_flipping, (batch // 2,))
                d_bad = np.random.binomial(1, label_flipping, (batch // 2,))
            else:
                d_good = np.ones((batch // 2,))
                d_bad = np.zeros((batch // 2,))
            d_good = np.reshape(d_good, (*d_good.shape, 1))
            d_bad = np.reshape(d_bad, (*d_bad.shape, 1))

            d_lossg = self.D.train_on_batch(good_inp, d_good)
            d_lossb = self.combinedRev.train_on_batch(bad_inp, d_bad)
            d_loss = [((d_lossg[x] + d_lossb[x]) / 2) for x in range(len(d_lossb))]

            # Generator
            idxs = np.random.randint(0, self.sdata.shape[0], batch)
            g_inp = self.sdata[idxs]
            g_real = self.ddata[idxs]
            g_out = np.ones((batch, 1))
            g_loss = self.combined.train_on_batch(g_inp, [g_real, g_out])

            print(f"{epoch}\tD: [loss: {d_loss[0]:.4f} binary_acc: {d_loss[1]:.2f}]\tG: [loss: {g_loss[0]:.2f} loss_mae: {g_loss[1]:.2f} loss_D: {g_loss[2]:.6f}]")

            if epoch % save == 0 or epoch == epochs:
                self.show_img(epoch)


    def save(self):
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.show_img("-")
        self.G.save(f"../model_{timenow}_G.h5")
        self.D.save(f"../model_{timenow}_D.h5")


    def show_img(self, epoch):
        cnt = 5
        idxs = np.random.randint(0, self.sdata.shape[0], cnt)
        simg = self.sdata[idxs]
        # pimg = simg # + np.random.normal(scale=0.5, size=simg.shape)
        dimg = self.ddata[idxs]
        gimg = self.G.predict(simg)

        # Rescale images 0 - 1
        simg = simg * 0.5 + 0.5
        gimg = gimg * 0.5 + 0.5
        dimg = dimg * 0.5 + 0.5
        np.clip(simg, 0, 1, out=simg)
        np.clip(gimg, 0, 1, out=gimg)
        np.clip(dimg, 0, 1, out=dimg)

        def hideTicks(fig):
            fig.axes.get_xaxis().set_ticks([])
            fig.axes.get_yaxis().set_ticks([])

        fig, axs = plt.subplots(3, cnt)
        for i in range(cnt):
            axs[0,i].imshow(simg[i,:,:,0], cmap='gray')
            show_gimg = gimg[i,:,:,0] if self.gray else gimg[i,:,:,:]
            axs[1,i].imshow(show_gimg, cmap='gray' if self.gray else None)
            show_dimg = dimg[i,:,:,0] if self.gray else dimg[i,:,:,:]
            axs[2,i].imshow(show_dimg, cmap='gray' if self.gray else None)
            for j in range(3):
                hideTicks(axs[j,i])
        plt.subplots_adjust(wspace=0.02,hspace=0.02)
        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.savefig(f"../gan_{epoch}_{timenow}.png", bbox_inches='tight')
        plt.close(fig)


