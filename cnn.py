
from nn import NN

from keras import Model
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Input, Concatenate
from keras.layers import UpSampling2D, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization

def downsample(input_layer, filters, kernel = 4, dropout = 0.2, norm=True):
    t0 = Conv2D(filters, kernel, strides=2, padding='same')(input_layer)
    t1 = LeakyReLU(0.2)(t0)
    if norm:
        t1 = BatchNormalization()(t1)
    t2 = Dropout(dropout)(t1)
    return t2

def upsample(input_layer, filters, skip_layer = None, kernel = 4, batch_normalization = True):
    t0 = UpSampling2D()(input_layer)
    t0 = Conv2D(filters, kernel, strides=1, padding='same')(t0)
    if batch_normalization:
        t0 = BatchNormalization()(t0)
    t0 = Dropout(0.5)(t0)
    t0 = LeakyReLU(0.2)(t0)
    if skip_layer is not None:
        t0 = Concatenate()([t0, skip_layer])
    return t0

class CNN(NN):

    def __init__(self, size=64, gray=False, gen_depth=16):
        NN.__init__(self, gray=gray)

        # input image size
        self.size = size
        self.gen_depth = gen_depth
        self.G = None


    def generator(self, silent=False):
        if self.G:
            if not silent:
                self.G.summary()
            return self.G

        depth = self.gen_depth

        inp = Input(shape=(self.size, self.size, 1))
        g1 = GaussianNoise(0.1)(inp)
        d1 = downsample(g1, depth * 1, norm=False)
        d2 = downsample(d1, depth * 2)
        d3 = downsample(d2, depth * 4)

        m0 = Flatten()(d3)
        m1 = Dense(8 * 8 * depth * 2)(m0)
        m2 = Dropout(0.5)(m1)
        m3 = LeakyReLU(0.2)(m2)
        m4 = Dense(8 * 8 * depth * 2)(m3)
        m5 = Dropout(0.5)(m4)
        m6 = Activation('relu')(m5)
        m7 = Reshape((8, 8, depth * 2))(m6)

        u1 = upsample(m7, depth * 8)
        u2 = upsample(u1, depth * 4)
        u3 = upsample(u2, depth * 4)

        lc = Conv2D((1 if self.gray else 3), 4, strides=1, padding='same')(u3)
        output = Activation('tanh')(lc)

        self.G = Model(inp, output)
        if not silent:
            self.G.summary()
        return self.G



