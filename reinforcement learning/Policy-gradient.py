import tensorflow as tf
import gym
import numpy as np

#VARIABLES

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

#Creating Network

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer = tf.layers.dense(X, num_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)

probabilties = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial( probabilties, num_samples=1)

# Convert from Tensor to number for network training
y = 1. - tf.to_float(action)

#loss and optimization
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

#gradients
gradients_and_variables = optimizer.compute_gradients(cross_entropy)



gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))


training_op = optimizer.apply_gradients(grads_and_vars_feed)
#initializing variables and saving model
init = tf.global_variables_initializer()
saver = tf.train.Saver()



