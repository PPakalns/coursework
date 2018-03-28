
import common
from keras.models import load_model
from datetime import datetime
from skimage.color import rgb2gray
import numpy as np

class NN:

    def __init__(self, gray=False):
        self.gray = gray
        self.sdata = None
        self.ddata = None
        self.G = None

    def load_data(self):
        if self.sdata is None:
            self.sdata, self.ddata = common.get_data()
            self.sdata = np.reshape(self.sdata, (*self.sdata.shape, 1))
            self.sdata = (self.sdata / 127.5) - 1
            self.ddata = (self.ddata / 127.5) - 1
            if self.gray:
                self.ddata = (self.ddata + 1) / 2
                self.ddata = np.asarray(rgb2gray(self.ddata))
                self.ddata = (self.ddata * 2) - 1
                self.ddata = np.reshape(self.ddata, (*self.ddata.shape, 1))
        return self.sdata, self.ddata

    def train(self, epochs=1):
        self.load_data()

        self.G.compile(optimizer='rmsprop',
                       loss='mean_squared_error',
                       metrics=['accuracy'])

        self.G.fit(self.sdata, self.ddata, epochs=epochs)

        timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.G.save(f"../model{timenow}.h5")

    def load(self, h5path):
        self.G = load_model(h5path)
