import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
import argparse
from matplotlib import pyplot as plt
from model_tf import *
from dqn_tf import *


parser = argparse.ArgumentParser(description="Train or test")
parser.add_argument("--train", dest="train", action="store_true", default=False)
parser.add_argument("--test", dest="test", action="store_true", default=False)
# parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
parser.add_argument("--model", dest="model", action="store_true", default=False)
parser.add_argument("--token", dest="token", action="store", required=False)
args = parser.parse_args()


def train():

    episodes = []
    scores = []
    losses = []
    old_score = 0
    stop = False

    tf.reset_default_graph()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        env = gym.make("CartPole-v1")
        n = env.observation_space.shape[0]
        agent = DQN(sess, env)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Run the initialization
        sess.run(init)

        for episode in range(1000):

            try:

                obs = env.reset().reshape(1, n)
                done = False
                score = 0
                t = 0

                while not done:

                    t += 1
                    # obs ----> action
                    action = agent.action(obs)
                    # action ---> new_obs and reward
                    obs_new, reward, done, _ = env.step(action)
                    # obs = obs.reshape(1,n)
                    obs_new = obs_new.reshape(1, n)

                    # reward = reward if not done or score == 499 else -10
                    score += reward
                    # add element in memory
                    agent.store(obs, action, reward, obs_new, done)
                    agent.train_model()
                    # attention when you do this part
                    obs = obs_new[:]

                    if done:
                        # score = score if score == 500 else score + 10
                        agent.update_weights()
                        episodes.append(episode)
                        scores.append(score)
                        losses.append(np.sum(agent.model.get_losses()))

                if (episode % 10 == 0) and (episode != 0):
                    print(
                        "episode {} score {} epsilon {} step {} loss {}".format(
                            episode, score, agent.epsilon, t, losses[-1]
                        )
                    )

                    if score > old_score:
                        # agent.save()
                        old_score = score

            except KeyboardInterrupt:

                fig = plt.figure(1)
                plt.plot(episodes, scores, "b")
                fig.savefig("./scores.png")
                plt.close(fig)
                stop = True
                print("Bye")

                # fig = plt.figure(2)
                # plt.plot(agent.history, "b")
                # fig.savefig("./loss.png")
                # plt.close(fig)

            if stop:
                break


def test():
    env = gym.make("CartPole-v1")

    agent = DQN(env)
    scores = []
    old_score = 0
    n = env.observation_space.shape[0]

    for episode in range(1000):

        obs = env.reset().reshape(1, n)

        score = 0

        # env.render()

        for t in range(500):

            # obs ----> action
            action = np.argmax(agent.model.predict(obs)[0])
            # action ---> new_obs and reward
            obs_new, reward, done, info = env.step(action)
            obs = obs.reshape(1, n)
            obs_new = obs_new.reshape(1, n)

            obs = obs_new

            score += reward
            if done:
                scores.append(score)
                break
        print("episode {} score {} epsilon {}".format(episode, score, agent.epsilon))
        print(np.mean(scores))


if __name__ == "__main__":

    if args.train:
        print("train")
        train()
    elif args.test:
        print("test")
        test()
