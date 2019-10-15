import tensorflow as tf
import numpy as np


class network:
    def __init__(self, name, sess, state_size, action_size, hidden_sizes, size_batch, learning_rate):

        with tf.variable_scope(name):
            self.sess = sess
            self.state_size = state_size
            self.action_size = action_size
            self.size_batch = size_batch
            self.hidden_sizes = hidden_sizes
            self.learning_rate = learning_rate

            self.X = tf.placeholder(tf.float32, shape=[None, self.state_size], name="X")
            self.Y = tf.placeholder(tf.float32, shape=[None, self.action_size], name="Y")
            self.Z = tf.placeholder(tf.float32, shape=[None, self.action_size], name="Z")
            self.q = tf.placeholder(tf.float32, shape=[None, self.action_size], name="q")

            self.parameters = {}
            self.loss = 0
            self.optimizer = 0
            self.losses = []

            self.build_model()
            self.weights = tf.trainable_variables()

    def initialize_forward(self):
        tf.set_random_seed(1)

        W1 = tf.get_variable(
            name="W1",
            shape=[self.state_size, self.hidden_sizes[0]],
            initializer=tf.contrib.layers.xavier_initializer(seed=1),
        )
        b1 = tf.get_variable(name="b1", shape=[1, self.hidden_sizes[0]], initializer=tf.zeros_initializer())

        W2 = tf.get_variable(
            name="W2",
            shape=[self.hidden_sizes[0], self.hidden_sizes[1]],
            initializer=tf.contrib.layers.xavier_initializer(seed=1),
        )
        b2 = tf.get_variable(name="b2", shape=[1, self.hidden_sizes[1]], initializer=tf.zeros_initializer())

        W3 = tf.get_variable(
            name="W3",
            shape=[self.hidden_sizes[1], self.action_size],
            initializer=tf.contrib.layers.xavier_initializer(seed=1),
        )
        b3 = tf.get_variable(name="b3", shape=[1, self.action_size], initializer=tf.zeros_initializer())

        Z1 = tf.add(tf.matmul(self.X, W1), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(A1, W2), b2)
        A2 = tf.nn.relu(Z2)
        self.Z = tf.add(tf.matmul(A2, W3), b3)

    def compute_loss(self):
        # error  ----> in this way the model tries to minimize on both the q_values
        self.loss = tf.squared_difference(self.Z, self.Y)
        self.loss = tf.reduce_mean(self.loss)

    def build_model(self):
        self.initialize_forward()
        self.compute_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, x_batch, y_batch):
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: x_batch, self.Y: y_batch})
        self.losses.append(loss)

    def predict(self, state):
        self.q = self.sess.run(self.Z, feed_dict={self.X: state})
        return self.q

    def get_losses(self):
        losses = self.losses[:]
        self.losses = []
        return losses

    def get_weights(self):
        self.weights = tf.trainable_variables()
        return self.weights
