import tensorflow as tf
import keras
from customlayer import MyLayerNorm
import numpy as np

def get_keras_vals(tensor, axis=-1):
    '''
    This function calculates the layer normalization outputs
    using Keras's layer
    '''
    layernorm = tf.keras.layers.LayerNormalization(axis=axis)
    return layernorm(tensor)


def get_custom_vals(tensor, axis=-1):
    '''
    This function calculates my custom layer output
    using my custom layer
    '''
    layernorm = MyLayerNorm(axis=axis)
    return layernorm(tensor)

def comp_layer(tensor, batch_size=32, axis=-1):
    rows = tensor.shape[0]
    rounds = rows // batch_size
    L2 = []
    INF = []
    MSE = []
    start = 0
    stop = batch_size
    for batch in range(0, rounds):
        if (batch == (rounds - 1)) and ((rounds % batch_size) != 0):
            sample = tensor[start:]

        else:
            sample = tensor[start:stop]

        keras_vals = get_keras_vals(sample, axis)
        custom_vals = get_custom_vals(sample, axis)

        diff = tf.subtract(keras_vals, custom_vals)

        # getting l infinity norm
        linf = tf.norm(diff, ord=np.inf).numpy()

        # getting l2 norm
        l2norm = tf.norm(diff, ord=2).numpy()
        rel = l2norm / tf.norm(keras_vals, ord=2)

        # getting mean squared error
        mse = tf.reduce_mean(tf.square(custom_vals - keras_vals))

        # putting values in a list
        L2.append(rel)
        INF.append(linf)
        MSE.append(mse)

        start += batch_size
        stop += batch_size
            
    return L2, INF, MSE