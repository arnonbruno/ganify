from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential


class Generator():
    def __init__(self, data, init=RandomNormal(mean=0.0, stddev=0.02)):
        self.random_dim = 100
        self.init = init
        self.feats = data.shape[1]

    def get_generator(self):
        self.generator = Sequential()
        self.generator.add(
            Dense(self.feats*2, input_dim=self.random_dim, kernel_initializer=self.init))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(self.feats*4))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(self.feats*8))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(self.feats, activation='tanh'))
        return self.generator
