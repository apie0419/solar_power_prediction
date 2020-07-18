import tensorflow as tf

def rmse(x, y):
    sub = tf.subtract(y, x)
    sq = tf.square(sub)
    rm = tf.reduce_mean(sq)

    return tf.sqrt(rm)

def mape(x, y):
    return tf.keras.losses.mean_absolute_percentage_error(y, x)