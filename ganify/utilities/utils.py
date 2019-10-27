from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import RMSprop, Adam


class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return K.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}


class Utilities():
    def __init__(self):
        self.type = type
        return

    def get_optimizer_wgan(self):
        return RMSprop(learning_rate=0.00005)

    def get_optimizer_gan(self):
        return Adam(lr=.0002, beta_1=.5)

    def get_wasserstein_loss(self, y_pred, y_true):
        self.loss = K.mean(y_pred*y_true)
        return self.loss

    def get_gan_loss(self):
        return 'binary_crossentropy'

    def get_random_dim(self):
        return 100

    def transform_data(self, data):
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        self.x_train = self.scaler.transform(data)
        return self.x_train, self.scaler
