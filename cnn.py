
from nn import NN

from keras import Model
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Input, Concatenate
from keras.layers import UpSampling2D, Reshape, Flatten, LeakyReLU
from keras.layers import BatchNormalization, GaussianNoise, Lambda
from keras_contrib.layers import InstanceNormalization
import tensorflow as tf

def_kernel = 4

def downsample(input_layer, filters, kernel = def_kernel, dropout = 0.5, norm=True, strides=2, padding="same"):
    t0 = Conv2D(filters, kernel, strides=strides, padding=padding)(input_layer)
    if norm:
        t0 = InstanceNormalization()(t0)
    t0 = LeakyReLU(0.2)(t0)
    t0 = Dropout(dropout)(t0, training=True)
    return t0

def resize_like(input_tensor):
    H, W = input_tensor.get_shape()[1], input_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [2*H.value, 2*W.value])

def upsample(input_layer, filters, skip_layer = None, kernel = def_kernel, norm = True):
    t0 = input_layer
    t0 = UpSampling2D()(t0)
    # t0 = Lambda(resize_like)(t0)
    t0 = Conv2D(filters, kernel, padding='same', strides=1)(t0)
    # t0 = Conv2DTranspose(filters, kernel, padding='same', strides=2)(t0)
    if norm:
        t0 = InstanceNormalization()(t0)
    # t0 = LeakyReLU(0.2)(t0)
    t0 = Activation('relu')(t0)
    t0 = Dropout(0.5)(t0, training=True)
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
        # l = GaussianNoise(0.5)(inp)
        l = inp
        l = downsample(l, depth * 1, kernel=5, norm=False) #64
        l = downsample(l, depth * 2) #32
        l3 = l
        l = downsample(l, depth * 4) #16
        l4 = l
        l = downsample(l, depth * 8)
        l5 = l
        l = downsample(l, depth * 8)
        l6 = l
        l = downsample(l, depth * 8)
        l = upsample(l, depth * 8, l6)
        l = upsample(l, depth * 8, l5)
        l = upsample(l, depth * 8, l4)
        l = upsample(l, depth * 4, l3)
        l = upsample(l, depth * 2 * 2)
        l = upsample(l, depth * 1 * 2)

        l = Conv2D((1 if self.gray else 3), def_kernel, strides=1, padding='same')(l)
        output = Activation('tanh')(l)

        self.G = Model(inp, output)
        if not silent:
            self.G.summary()
        return self.G



