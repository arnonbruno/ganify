from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from ganify.discriminator.critic import Critic
from ganify.discriminator.discriminator import Discriminator
from ganify.generator.generator import Generator
from ganify.utilities.utils import Utilities
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ganify():
    def __init__(self):
        self.utils = Utilities()
        self.random_dim = self.utils.get_random_dim()
        self.seed = 42
        self.f = False
        np.random.seed(self.seed)

    def fit_data(self, x_train, y_train, type='wgan', cols_names=None, batch_size=8, epochs=1):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        if len(self.x_train.shape) > 2:
            raise RuntimeError(
                'This tool is supposed to work with array data, not tensors')
        if len(np.unique(self.y_train)) > 1:
            raise RuntimeError(
                'More than 1 classes to amplify on target. Ideally, you want to create fake examples from one single class')
        if cols_names is not None and len(cols_names) != x_train.shape[1]:
            raise RuntimeError(
                'Column names must be an array with the same size as input data')
        elif cols_names is not None and len(cols_names) == x_train.shape[1]:
            self.cols_names = cols_names
        self.gen = Generator(self.x_train)
        self.adversary_one = self.gen.get_generator()
        self.type = type
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_train, self.scaler = self.utils.transform_data(self.x_train)
        if len(self.y_train) != len(self.x_train):
            raise RuntimeError('Lenght of data and target are different')
        if self.type == 'wgan':
            self.critic = Critic(self.x_train)
            self.adversary_two = self.critic.get_critic()
            self.loss = self.utils.get_wasserstein_loss
            self.opt = self.utils.get_optimizer_wgan()
            self.adversary_one, self.adversary_two, self.d1_hist, self.d2_hist, self.g_hist = self.train_gan(
                self.x_train, self.y_train, self.epochs, self.batch_size)
        elif self.type == 'gan':
            self.discriminator = Discriminator(self.x_train)
            self.adversary_two = self.discriminator.get_discriminator()
            self.loss = self.utils.get_gan_loss()
            self.opt = self.utils.get_optimizer_gan()
            self.adversary_one, self.adversary_two, self.d1_hist, self.d2_hist, self.g_hist = self.train_gan(
                self.x_train, self.y_train, self.epochs, self.batch_size)
        else:
            raise RuntimeError(
                'Invalid type o GAN, please select a valid option. See documentation for datails.')
        return self

    def get_gan(self):
        self.adversary_two.trainable = False
        self.gan_input = Input(shape=(self.random_dim,))
        self.h1 = self.adversary_one(self.gan_input)
        self.output = self.adversary_two(self.h1)
        self.gan = Model(inputs=self.gan_input, outputs=self.output)
        self.gan.compile(optimizer=self.opt, loss=self.loss)
        return self.gan

    def train_gan(self, x_train, y_train, epochs, batch_size):
        self.batch_count = self.x_train.shape[0] // batch_size
        self.gan = self.get_gan()
        self.preds = []
        self.d1_hist, self.d1_tmp, self.d2_hist, self.d2_tmp, self.g_hist, = [
            [] for _ in range(5)]
        for epoch in range(1, epochs + 1):
            print('Epoch {}'.format(epoch), end='\r')
            for _ in tqdm(range(self.batch_count), leave=False):
                self.adversary_two.trainable = True
                for _ in range(5):
                    coin = round(np.random.uniform(0, 1), 2)
                    noise = np.random.normal(
                        0, 1, size=(batch_size, self.random_dim))
                    data_batch = self.x_train[np.random.randint(
                        0, self.x_train.shape[0], size=batch_size)]
                    # gerando sinistros falsos
                    self.generated_data = self.adversary_one.predict(noise)
                    # treino o discriminador com dados reais (seguindo as dicas de Chintala (2016))
                    y_dis = np.ones(batch_size)
                    # y_dis[:batch_size] = round(np.random.uniform(-0.9, -1.1), 2) if coin > .05   else 1.0
                    y_dis[:batch_size] = -1.0 if coin > .05 else 1.0
                    d1 = self.adversary_two.train_on_batch(data_batch, y_dis)
                    self.d1_tmp.append(d1)

                    # treino o discriminador com dados falsos (seguindo as dicas de Chintala (2016))
                    y_dis = np.ones(batch_size)
                    y_dis[: batch_size] = 1.0 if coin > .05 else -1.0
                    d2 = self.adversary_two.train_on_batch(
                        self.generated_data, y_dis)
                    self.d2_tmp.append(d2)
                self.d1_hist.append(np.mean(self.d1_tmp))
                self.d2_hist.append(np.mean(self.d2_tmp))
                # treino o gerador
                noise = np.random.normal(
                    0, 1, size=(batch_size, self.random_dim))
                y_gen = np.ones(batch_size)*-1
                self.adversary_two.trainable = False
                g = self.gan.train_on_batch(noise, y_gen)
                self.g_hist.append(g)
                self.f = True
        return self.adversary_one, self.adversary_two, self.d1_hist, self.d2_hist, self.g_hist

    def create_bulk(self, lenght, output=None):
        if self.f is False:
            raise RuntimeError(
                'Model not fit to data yet. Please train the model with fit_data first')
        g = self.adversary_one
        d = self.adversary_two
        sc = self.scaler
        random_dim = self.utils.get_random_dim()
        self.fake_data = []
        pred_value = []
        latent_space = np.random.normal(size=(lenght, random_dim))
        for i in range(len(latent_space)):
        	x = np.expand_dims(latent_space[i], 0)
        	fake_example = g.predict(x)
        	pred = d.predict(fake_example)[0][0]
        	self.fake_data.append(sc.inverse_transform(fake_example))
        	pred_value.append(pred)
        self.fake_data = np.reshape(np.array(self.fake_data), newshape=(
    		np.array(self.fake_data).shape[0], np.array(self.fake_data).shape[2]))
        if output == 1:
            self.fake_data = pd.DataFrame(
                columns=self.cols_names, data=self.fake_data)
        return self.fake_data

    def plot_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.d1_hist, color='b', label='Real')
        plt.plot(self.d2_hist, color='r', label='Fake')
        plt.plot(self.g_hist, color='g', label='Generator')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
