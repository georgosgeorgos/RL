import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
import argparse
from matplotlib import pyplot as plt
from model_tf import *


class DQN:

    def __init__(self, sess, env):

        self.best_weights = './best_weights.h5'
        self.sess = sess
        self.env = env

        self.state_size  = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # in memory store the last n action|observation
        # during the training we train on random samples
        self.memory = deque(maxlen=2000)

        # discount factor
        self.gamma = 0.99
        # threshold exploration/exploitation
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.size_batch = 32
        self.hidden_sizes = [32, 32]

        # initialize networks
        self.model = network("model", self.sess, self.state_size, self.action_size, self.hidden_sizes,
                                  self.size_batch, self.learning_rate)
        self.model_target = network("model_target", self.sess, self.state_size, self.action_size, self.hidden_sizes,
                                  self.size_batch, self.learning_rate)

    def action(self, state):
        # explore %epsilon iterations
        if np.random.random() < self.epsilon:
            act = self.env.action_space.sample()
            return act
        # exploit
        q_value_actions = self.model.predict(state)[0]
        act = np.argmax(q_value_actions)
        return act

    def store(self, state, action, reward, new_state, done):

        state = np.reshape(state, newshape=(1, self.state_size))
        new_state = np.reshape(new_state, newshape=(1, self.state_size))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        sample = [state, action, reward, new_state, done]
        self.memory.append(sample)

    def train_model(self):

        '''
        compute the expected reward and train the model
        # we use two networks: the first is the model that we want to train;
        # the second is the model useful to compute the expected reward

        '''

        if len(self.memory) < self.size_batch:
            return None

        state = np.zeros((self.size_batch, self.state_size))
        action = np.zeros((self.size_batch, self.action_size))
        reward = np.zeros((self.size_batch, 1))
        new_state = np.zeros((self.size_batch, self.state_size))
        done = np.zeros((self.size_batch, 1))

        for i in range(self.size_batch):
            sample = random.sample(self.memory, 1)[0]

            state[i, :] = sample[0]
            action[i, sample[1]] = 1
            reward[i, :] = sample[2]
            new_state[i, :] = sample[3]
            done[i, :] = sample[4] * 1

        # we use q_model to train

        q_model = self.model.predict(state)
        # the key point of Double DQN
        # selection of action is from model
        # update is from target model
        # q_model_next = self.model.predict(new_state)
        # a = np.argmax(q_model_next, axis = 1)
        # a = (np.arange(self.output_size) == a[:,None]).astype(np.bool)
        # we use q_model_target to compute the expected reward
        # and update q_model[action]
        q_model_target = self.model_target.predict(new_state)
        # expected reward
        y = reward + self.gamma * np.amax(q_model_target, axis=1, keepdims=True) * (1 - done)

        # action choosen by the model given the state in matrix form
        # action = (np.arange(self.output_size) == action[:,None]).astype(np.bool)
        # update the q-value of our model with the expected reward for the choosen action
        for i in range(self.size_batch):
            q_model[i, np.argmax(action[i])] = y[i]
        #q_model[action] = y

        # we train the model (non target) at every iteration
        # input ---> state
        # output ---> Q-values (or probs not normalized) for actions
        x_batch = state[:]
        y_batch = q_model[:]

        self.model.train(x_batch, y_batch)

    def update_weights(self):

        '''
        upload model_target weights

        '''

        w = self.model.get_weights()

        for i in range(len(w)):
            self.sess.run(self.model_target.weights[i].assign(w[i]))


        #w_target = w * (1 - self.tau) + w_target * (self.tau)