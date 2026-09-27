import numpy as np
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model

from ganify.utilities.utils import hidden_width, kernel_initializer


class Generator:
    def __init__(self, data, init=None, random_dim=100, max_units=512, seed=None):
        shape = np.shape(data)
        if len(shape) != 2:
            raise ValueError("Generator expects a 2D feature matrix")
        self.feats = int(shape[1])
        self.random_dim = int(random_dim)
        self.max_units = int(max_units)
        self.seed = seed
        self.init = init

    def _initializer(self, index):
        if self.init is not None:
            return self.init
        return kernel_initializer(self.seed, index)

    def get_generator(self):
        widths = [
            hidden_width(self.feats, 2, self.max_units),
            hidden_width(self.feats, 4, self.max_units),
            hidden_width(self.feats, 8, self.max_units),
        ]
        inputs = Input(shape=(self.random_dim,), name="latent")
        hidden = inputs
        for index, width in enumerate(widths):
            hidden = Dense(width, kernel_initializer=self._initializer(index))(hidden)
            hidden = LeakyReLU(0.2)(hidden)
        outputs = Dense(
            self.feats,
            activation="tanh",
            kernel_initializer=self._initializer(len(widths)),
        )(hidden)
        self.generator = Model(inputs, outputs, name="generator")
        return self.generator
