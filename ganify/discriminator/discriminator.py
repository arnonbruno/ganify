import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Model

from ganify.utilities.utils import Utilities, hidden_width, kernel_initializer


class Discriminator:
    def __init__(self, data, init=None, seed=None, max_units=512):
        shape = np.shape(data)
        if len(shape) != 2:
            raise ValueError("Discriminator expects a 2D feature matrix")
        self.feat = int(shape[1])
        self.init = init
        self.seed = seed
        self.max_units = int(max_units)
        self.utilities = Utilities()
        self.optimizer = self.utilities.get_optimizer_gan()

    def _initializer(self, index):
        if self.init is not None:
            return self.init
        return kernel_initializer(self.seed, index + 20)

    def get_discriminator(self):
        widths = [
            hidden_width(self.feat, 8, self.max_units),
            hidden_width(self.feat, 4, self.max_units),
            hidden_width(self.feat, 2, self.max_units),
        ]
        inputs = Input(shape=(self.feat,), name="features")
        hidden = inputs
        for index, width in enumerate(widths):
            hidden = Dense(width, kernel_initializer=self._initializer(index))(hidden)
            hidden = LeakyReLU(0.2)(hidden)
            dropout_seed = None if self.seed is None else int(self.seed) + 100 + index
            hidden = Dropout(0.5, seed=dropout_seed)(hidden)
        outputs = Dense(
            1,
            activation="sigmoid",
            kernel_initializer=self._initializer(len(widths)),
        )(hidden)
        discriminator = Model(inputs, outputs, name="discriminator")
        discriminator.compile(loss="binary_crossentropy", optimizer=self.optimizer)
        return discriminator
