import tensorflow as tf


class model:

    def __init__(self, sess, state_size, action_size, hidden_sizes, size_batch, learning_rate):

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.size_batch = size_batch
        self.hidden_sizes = hidden_sizes
        self.parameters = {}
        self.learning_rate = learning_rate

        self.X = tf.placeholder(tf.float64, shape=[self.state_size, None], name="X")
        self.Y = tf.placeholder(tf.float64, shape=[self.action_size, None], name="Y")

        self.initialize()

    def initialize(self):
        tf.set_random_seed(1)

        W1 = tf.get_variable(name="W1", shape=[self.hidden_sizes[0], self.state_size],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable(name="b1", shape=[self.hidden_sizes[0], 1],
                             initializer=tf.zeros_initializer())

        self.parameters["W1"] = W1
        self.parameters["b1"] = b1

        W2 = tf.get_variable(name="W2", shape=[self.hidden_sizes[1], self.hidden_sizes[0]],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable(name="b2", shape=[self.hidden_sizes[1], 1],
                             initializer=tf.zeros_initializer())

        self.parameters["W2"] = W2
        self.parameters["b2"] = b2

        W3 = tf.get_variable(name="W3", shape=[self.action_size, self.hidden_sizes[1]],
                             initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable(name="b3", shape=[self.action_size, 1],
                             initializer=tf.zeros_initializer())

        self.parameters["W3"] = W3
        self.parameters["b3"] = b3

    def forward(self):

        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        W3 = self.parameters["W3"]
        b3 = self.parameters["b3"]

        Z1 = tf.add(tf.matmul(W1, self.X), b1)
        A1 = tf.nn.relu(Z1)

        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)

        Z3 = tf.add(tf.matmul(W3, A2), b3)

        return Z3

    def compute_cost(self):

        Z3 = self.forward()

        loss = tf.reduce_mean(tf.squared_difference(Z3, self.Y))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize((loss))

        return loss, optimizer


    def train(self, x_batch, y_batch):

        loss, optimizer = self.compute_cost()

        _, l = self.sess.run([optimizer, loss], feed_dict = {self.X : x_batch, self.Y : y_batch})

        return l

    def predict(self):

        Z3 = self.forward()
        out = tf.nn.sigmoid(Z3)
        return out