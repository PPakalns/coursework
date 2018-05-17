
from nn import NN

from keras import Model
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Input, Concatenate
from keras.layers import UpSampling2D, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization, GaussianNoise

def downsample(input_layer, filters, kernel = 4, dropout = 0.5, norm=True, strides=2):
    t0 = Conv2D(filters, kernel, strides=strides, padding='same')(input_layer)
    t0 = LeakyReLU(0.2)(t0)
    if norm:
        t0 = BatchNormalization()(t0)
    t0 = Dropout(dropout)(t0)
    return t0

def upsample(input_layer, filters, skip_layer = None, kernel = 4, batch_normalization = True, upsample=True):
    t0 = input_layer
    if upsample:
        t0 = UpSampling2D()(t0)
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
        l1 = inp
        l = downsample(inp, depth * 1, norm=False)
        l2 = l
        l = downsample(l, depth * 2)
        l3 = l
        l = downsample(l, depth * 4)
        l4 = l
        l = downsample(l, depth * 4)
        l = upsample(l, depth * 4, l4)
        l = upsample(l, depth * 4, l3)
        l = upsample(l, depth * 4, l2)
        l = upsample(l, depth * 2, l1)

        l = Conv2D((1 if self.gray else 3), 4, strides=1, padding='same')(l)
        output = Activation('tanh')(l)

        self.G = Model(inp, output)
        if not silent:
            self.G.summary()
        return self.G



