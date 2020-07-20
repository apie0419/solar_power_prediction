from matplotlib import pyplot as plt
from tensorflow.contrib.eager.python import tfe
from tcn import TCN
from utils import Dataset, rmse
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

base_path = os.path.dirname(os.path.abspath(__file__))

GPU           = 5
batch_size    = 64
hidden_units  = 12
dropout       = 0.1
epochs        = 500
ksize         = 2
levels        = 5
n_classes     = 1
timesteps     = 3
num_input     = 8
global_step   = tf.Variable(0, trainable=False)
l2_lambda     = 0
starter_learning_rate = 0.002

data_path = os.path.join(base_path, "../data/solar")

dataset = Dataset(data_path, timesteps)
trainset = tf.data.Dataset.from_tensor_slices((dataset.train_data, dataset.train_target))
train_data, train_target = tf.convert_to_tensor(dataset.train_data, dtype=tf.float32), tf.convert_to_tensor(dataset.train_target, dtype=tf.float32)
test_data, test_target = tf.convert_to_tensor(dataset.test_data, dtype=tf.float32), tf.convert_to_tensor(dataset.test_target, dtype=tf.float32)

channel_sizes = [hidden_units] * levels
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.7, staircase=True)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
model = TCN(n_classes, channel_sizes, kernel_size=ksize, dropout=dropout, l2_lambda=l2_lambda)

train_losses = list()
test_losses = list()

def loss_function(batch_x, batch_y):

    logits = model(batch_x, training=True)
    
    lossL2 = tf.add_n(model.losses)

    return rmse(logits, batch_y) + lossL2

with tf.device(f"/gpu:{GPU}"):
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        train = trainset.batch(batch_size, drop_remainder=True)

        for batch_x, batch_y in tfe.Iterator(train):
            batch_x = tf.reshape(batch_x, (batch_size, timesteps, num_input))
            batch_x = tf.dtypes.cast(batch_x, tf.float32)
            batch_y = tf.dtypes.cast(batch_y, tf.float32)
            optimizer.minimize(lambda: loss_function(batch_x, batch_y), global_step=global_step)

        
        logits = model(train_data, training=False)
        train_loss = rmse(logits, train_target)
        train_losses.append(train_loss.numpy())

        logits = model(test_data, training=False)
        test_loss = rmse(logits, test_target)
        target = test_target

        test_losses.append(test_loss.numpy())
        pbar.set_description(f"Epoch {epoch}, Train Loss: {round(float(train_loss.numpy()), 4)}, Test Loss: {round(float(test_loss.numpy()), 4)}")
        
if not os.path.exists(os.path.join(base_path, "Output")):
    os.mkdir(os.path.join(base_path, "Output"))

pd.DataFrame({
    "train": train_losses,
    "test": test_losses
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_loss.png"))

print("Optimization Finished!")

logits = model(test_data, training=False)
target = test_target

pd.DataFrame({
    "predict": logits[:, 0],
    "target": target[:, 0]
}).plot()

plt.savefig(os.path.join(base_path, "Output/tcn_evaluation.png"))
