import gym
import numpy as np
from gym import wrappers
import random
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

numtilings = 8 * 9 * 9
thetas = np.zeros(648 + 1) # one for action
alpha = 0.5/8
gamma = 0.9
epsilon = 0.1

def take_action(observation):
    """
    eplison greedy
    """
    if random.random() > epsilon:
        return np.argmax([qfunction(observation, action) for action in xrange(env.action_space.n)])
    else:
        return random.randint(0,2)

def qfunction(observation, action):
    global thetas
    return np.matmul(features(observation, action), thetas)

def delta(observation, action):
    return features(observation, action)

def features(observation, action):
    global numtilings
    tileIndices = [-1] * 8
    tc(observation[0], observation[1], tileIndices)
    feature = [0] * 648
    for tile_index in tileIndices:
        feature[tile_index] = 1
    return np.append(feature, (np.absolute(np.array([1,0,0]) - action)))

def learn():
    global thetas
    global env
    cost2Go = []
    #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
    for i_episode in range(10000):
        """Initial State"""
        #print("Episode start thetas {}".format(thetas))
        observation = env.reset()
        reward_total = 0
        action = take_action(observation)
        #print("action->{}.format(action))
        for t in range(4000):
            observation_1, reward_1, done, info = env.step(action)
            reward_total += rewards_1
            if done:
                change = aplha * (rewards_1 - qfunction(observation, action))
                thetas += [change * d for d in delta(observation, action)]
                break
            action_1 = take_action(observation_1)
            qdash = qfunction(observation_1, action_1)
            q = qfunction(observation, action)
            change = alpha * ((rewards_1 + gamma * qdash) - q)
            thetas += [change * d for d in delta(observation, action)]
            observation = observation_1
            action = action_1

        cost2Go.append(-reward_total)
        print("Episode# {} finished after {} timesteps with total rewards {}".format(i_episode, t + 1, reward_total))
        #print("Learnt Theta{}".format(thetas))
        plt.plot(cost2Go)
        plt.ylabel('cost to go')
        plt.show()

if __name__ == "__main__":
    learn()

