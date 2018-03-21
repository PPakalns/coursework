
import common
from keras.models import load_model
from datetime import datetime
import numpy as np

class NN:

    def __init__(self):
        self.sdata = None
        self.ddata = None
        self.G = None

    def load_data(self):
        if self.sdata is None:
            self.sdata, self.ddata = common.get_data()
            self.sdata = np.reshape(self.sdata, (*self.sdata.shape, 1))
            self.sdata = (self.sdata / 255) - 0.5
            self.ddata = (self.ddata / 255) - 0.5
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
