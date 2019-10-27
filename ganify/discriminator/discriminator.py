from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from ganify.utilities.utils import Utilities


class Discriminator():
    def __init__(self, data, init=RandomNormal(mean=0.0, stddev=0.02)):
        self.init = init
        self.utilities = Utilities()
        self.optimizer = self.utilities.get_optimizer_gan()
        self.feat = data.shape[1]

    def get_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Dense(512, input_dim=self.feat,
                                kernel_initializer=self.init))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(.5))

        discriminator.add(Dense(256))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(.5))

        discriminator.add(Dense(128))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(.2))
        discriminator.add(Dropout(.5))

        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer)
        return discriminator
