'''
Gwen Horzempa
Project 2
MATH 474
Last Updated: 2/25/26

In this file, I will be practicing making a custom model
'''

import tensorflow as tf
import keras

class MyLayerNorm(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		
	def build(self, input_shape):
		'''
		Should define two trainable weight vectors alpha and 
		beta, both of which have shape input_shape[-1:] and data
		type tf.float32
		'''
		self.alpha = self.add_weight(
            name ="alpha", shape=input_shape[-1:], 
			dtype=tf.float32, initializer="ones", trainable=True
        )
		self.beta = self.add_weight(
            name ="beta", shape=input_shape[-1:], 
			dtype=tf.float32, initializer="zeros", trainable=True
        )
			
	def call(self, inputs):
		'''
		should compute the mean and shandard deviation of each
		instance's features
		
		should compute and return that equation
		'''
		mean, var = tf.nn.moments(inputs, axes=-1, keepdims=True)
		std = var**0.5
		eq = (inputs - mean) / (std + 1e-4)
		return (self.alpha * eq) + self.beta
		
	def get_config(self):
		base_config = super().get_config()
		return{**base_config, "alpha": self.alpha, 
						"beta": self.beta}