
from nn import NN

from keras import Model
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Input, Concatenate
from keras.layers import UpSampling2D, Reshape, Flatten

def downsample(input_layer, filters, kernel = 4, dropout = 0.1):
    t0 = Conv2D(filters, kernel, strides=2, padding='same')(input_layer)
    t1 = Activation('relu')(t0)
    t2 = Dropout(dropout)(t1)
    return t2

def upsample(input_layer, skip_layer, filters, kernel = 4):
    t0 = UpSampling2D()(input_layer)
    t1 = Conv2D(filters, kernel, strides=1, padding='same')(t0)
    t2 = Activation('relu')(t1)
    t3 = Concatenate()([t2, skip_layer])
    return t3

class CNN(NN):

    def __init__(self, size=64, gray=False):
        NN.__init__(self, gray=gray)

        # input image size
        self.size = size
        self.depth = 8
        self.G = None


    def generator(self, silent=False):
        if self.G:
            if not silent:
                self.G.summary()
            return self.G

        depth = self.depth

        inp = Input(shape=(self.size, self.size, 1))
        d1 = downsample(inp, depth * 1)
        d2 = downsample(d1, depth * 2)
        d3 = downsample(d2, depth * 4)

        m0 = Flatten()(d3)
        m1 = Dense(8 * 8 * depth * 2)(m0)
        m2 = Activation('tanh')(m1)
        m3 = Reshape((8, 8, depth * 2))(m2)

        u1 = upsample(m3, d2, depth * 8)
        u2 = upsample(u1, d1, depth * 4)
        u3 = upsample(u2, inp, depth * 4)

        lc = Conv2D((1 if self.gray else 3), 4, strides=1, padding='same')(u3)
        output = Activation('tanh')(lc)

        self.G = Model(inp, output)
        if not silent:
            self.G.summary()
        return self.G



