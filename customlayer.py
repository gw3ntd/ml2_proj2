'''
Gwen Horzempa
Project 2
MATH 474
Last Updated: 3/07/26

In this file, I will be practicing making a custom model
'''

import tensorflow as tf
import keras

class MyLayerNorm(tf.keras.layers.Layer):
	def __init__(self, axis=-1, **kwargs):
		super().__init__(**kwargs)
		self.axis = axis
		
	def build(self, input_shape):
        # making sure axis is a list
		if isinstance(self.axis, int):
			axes = [self.axis]
		else:
			axes = self.axis
		
		param_shape = []
        # figuring out the right shape for alpha and beta
		for a in range(len(input_shape)):
			if a in axes: # if it is an axis we are normalizing over
				param_shape.append(input_shape[a])
			else:
				param_shape.append(1)
		
		self.alpha = self.add_weight(
			name="alpha", shape=param_shape,
			dtype=tf.float32, initializer="ones", trainable=True
		)
		self.beta = self.add_weight(
			name="beta", shape=param_shape,
			dtype=tf.float32, initializer="zeros", trainable=True
		)
			
	def call(self, inputs):
		'''
		should compute the mean and shandard deviation of each
		instance's features
		
		should compute and return that equation
		'''
		mean, var = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
		std = var**0.5
		eq = (inputs - mean) / (std + 1e-3)
		return (self.alpha * eq) + self.beta
	
	def get_config(self):
		base_config = super().get_config()
		return {**base_config, "axis":self.axis}
    
		