import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from . import TemporalConvNet
from tensorflow.keras import regularizers

class TCN(tf.keras.Model):
    def __init__(self, output_size, num_channels, kernel_size, dropout, l2_lambda):
        super(TCN, self).__init__()
        init = tf.keras.initializers.he_normal(seed=1)

        self.temporalCN = TemporalConvNet(num_channels, kernel_size=kernel_size, dropout=dropout, l2_lambda=l2_lambda)
        self.linear = tf.keras.layers.Dense(output_size, kernel_regularizer=regularizers.l2(l2_lambda), kernel_initializer=init)

    def call(self, x, training=True):
        y = self.temporalCN(x, training=training)
        return self.linear(y[:, -1, :])   # use the last element to output the result