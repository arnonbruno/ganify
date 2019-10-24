from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential

class Generator():
    def __init__(self, features, init=RandomNormal(mean=0.0, stddev=0.02)):
        self.random_dim=100
        self.init = init
        self.feat = features

    def get_generator(self):
        self.generator = Sequential()
        self.generator.add(Dense(128, input_dim=self.random_dim, kernel_initializer=self.init))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(256))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(512))
        self.generator.add(BatchNormalization())
        self.generator.add(LeakyReLU(.2))
        self.generator.add(Dropout(.5))

        self.generator.add(Dense(len(self.feat), activation='tanh'))
        return self.generator
