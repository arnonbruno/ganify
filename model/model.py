import warnings
from modules.stem.config import stem_features_small
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.models import Sequential, Model
from discriminator.critic import Critic
from generator.generator import Generator
from utilities.utils import Utilities
from tqdm import tqdm
import numpy as np

class Ganify():
    def __init__(self, x_train, y_train, cols_names=None, type='wgan'):
        self.adversary_one = Generator()
        self.adversary_two = Critic()
        self.utils = Utilities()
        self.random_dim = self.utils.get_random_dim()
        self.data = np.array(x_train)
        if len(self.data.shape) > 2:
            raise RuntimeError('This tool is supposed to work with array data, not tensors')
        if cols_names is not None and len(cols_names) == x_train.shape[1] + 1:
            self.features = cols_names
        else:
            raise RuntimeError('Column names must be an array with the same size as input data')
        self.target = np.array(y_train)
        self.seed = 42
        self.type = type
        self.gen = self.adversary_one.get_generator()
        if self.type == 'wgan':
            self.discriminator = self.adversary_two.get_critic()
            self.loss = self.adversary_two.wasserstein_loss
            self.opt = self.utils.get_optimizer_wgan()
        elif self.type == 'gan':
            print('put otha')
        else:
            raise RuntimeError('Invalid type o GAN, please select a valid option. See documentation for datails.')


    def fit_data(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.data)
        self.x_train = self.scaler.transform(self.data)
        self.y_train = self.data[self.target]
        if len(self.y_train) != len(self.x_train):
            raise RuntimeError('Lenght of data and target are different')
        return (self.x_train, self.y_train)

    def get_gan(self):
        self.discriminator.trainable = False
        self.gan_input = self.Input(shape=(self.random_dim,))
        self.h1 = self.generator(self.gan_input)
        self.output = self.discriminator(self.h1)
        self.gan = self.Model(inputs=self.gan_input, outputs=self.output)
        self.gan.compile(optimizer=self.opt, loss=self.loss)
        return self.gan

    def train_gan(self, epochs=1, batch_size=8):
        self.x_train, self.y_train = self.fit_data()
        self.batch_count = self.x_train.shape[0] // batch_size
        self.gan = self.get_gan()
        self.preds = []
        self.d1_hist, self.d1_tmp, self.d2_hist, self.d2_tmp, self.g_hist, = [[] for _ in range(5)]
        for epoch in range(1, epochs + 1):
            print('Epoch {}'.format(epoch), end='\r')
            for _ in tqdm(range(self.batch_count), leave=False):
                self.discriminator.trainable = True
                for _ in range(5):
                    coin = round(np.random.uniform(0, 1), 2)
                    noise = np.random.normal(0, 1, size=(batch_size, self.random_dim))
                    data_batch = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size)]
                    # gerando sinistros falsos
                    self.generated_data = self.generator.predict(noise)
                    # treino o discriminador com dados reais (seguindo as dicas de Chintala (2016))
                    self.discriminator.trainable = True
                    y_dis = np.ones(batch_size)
                    # y_dis[:batch_size] = round(np.random.uniform(-0.9, -1.1), 2) if coin > .05   else 1.0
                    y_dis[:batch_size] = -1.0 if coin > .05 else 1.0
                    d1 = self.discriminator.train_on_batch(data_batch, y_dis)
                    self.d1_tmp.append(d1)

                    # treino o discriminador com dados falsos (seguindo as dicas de Chintala (2016))
                    y_dis = np.ones(batch_size)
                    y_dis[: batch_size] = 1.0 if coin > .05 else -1.0
                    d2 = self.discriminator.train_on_batch(self.generated_claims, y_dis)
                    self.d2_tmp.append(d2)
                    self.d1_hist.append(np.mean(self.d1_tmp))
                    self.d2_hist.append(np.mean(self.d2_tmp))
                    # treino o gerador
                    noise = np.random.normal(0, 1, size=(batch_size, self.random_dim))
                    y_gen = np.ones(batch_size)*-1
                    self.discriminator.trainable = False
                    g = self.gan.train_on_batch(noise, y_gen)
                    self.g_hist.append(g)
                    #     # Produzo dados aleatórios para criar exemplos falsos e identificar se o modelo  alcançou o equilíbrio de Nash
                    #     sample = np.random.normal(size=random_dim)
                    #     sample = np.expand_dims(sample, 0)
                    #     preds.append(discriminator.predict(generator.predict(sample))[0][0])
                    # plt.plot(preds)
                    # plt.title('Results after {} epochs: (mean = {})'.format(epoch, round(np.mean(preds), 2)))
                    return self.generator, self.discriminator, self.preds, self.d1_hist,self.d2_hist, self.g_hist
