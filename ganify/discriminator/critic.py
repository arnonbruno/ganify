from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from ganify.utilities.utils import ClipConstraint, Utilities


class Critic():
    def __init__(self, data, init=RandomNormal(mean=0.0, stddev=0.02)):
        self.init = init
        self.utilities = Utilities()
        self.const = ClipConstraint(.01)
        self.optimizer = self.utilities.get_optimizer_wgan()
        self.feat = data.shape[1]
        self.loss = self.utilities.get_wasserstein_loss

    def get_critic(self):
        self.critic = Sequential()
        self.critic.add(Dense(self.feat*8, input_dim=self.feat,
                              kernel_initializer=self.init, kernel_constraint=self.const))
        self.critic.add(LeakyReLU(.2))

        self.critic.add(Dense(self.feat*4, kernel_initializer=self.init,
                              kernel_constraint=self.const))
        self.critic.add(LeakyReLU(.2))

        self.critic.add(Dense(self.feat*2, kernel_initializer=self.init,
                              kernel_constraint=self.const))
        self.critic.add(LeakyReLU(.2))

        self.critic.add(Dense(1, activation='linear'))
        self.critic.compile(loss=self.loss, optimizer=self.optimizer)
        return self.critic
