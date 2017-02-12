import gym
import numpy as np
from gym import wrappers
import random
from utils.TileCoding import *
import matplotlib.pyplot as plt

ENVS = {
    'mountaincar': 'MountainCar-v0',
    'max_episodes': 10000
    # --env mountaincar --gamma 0.99 --eps 0.3  --goal -110  --upload --max_episodes 10000 --eps_schedule 500
}
env = gym.make('MountainCar-v0')
env.max_episode_steps = 10000
env.seed(0)

numtilings = 8
maxtiles = 2048
thetas = np.zeros(maxtiles) # one for action
alpha = 0.01
gamma = 0.98
epsilon = 0.2

hashTable = IHT(maxtiles)


# get indices of active tiles for given state and action
def getActiveTiles(position, velocity, action):
    global hashTable
    global env
    # I think positionScale * (position - position_min) would be a good normalization.
    # However positionScale * position_min is a constant, so it's ok to ignore it.
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    activeTiles = tiles(hashTable, numtilings,
                        [numtilings * position / (max_position - min_position), numtilings * velocity / (max_velocity - min_velocity)],
                        [action])
    return activeTiles

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
    tileIndices = getActiveTiles(observation[0], observation[1], action)
    feature = [0] * maxtiles
    for tile_index in tileIndices:
        feature[tile_index] = 1
    return feature

def learn():

    global thetas
    global env
    cost2Go = []
    env = wrappers.Monitor(env, '/tmp/mountaincar-experiment-1')
    for i_episode in range(5000):
        """Initial State"""
        #print("Episode start thetas {}".format(thetas))
        observation = env.reset()
        reward_total = 0
        action = take_action(observation)
        #print("action->{}.format(action))
        for t in range(200):
            observation_1, reward_1, done, info = env.step(action)
            env.render()
            reward_total += reward_1
            if done:
                change = alpha * (reward_1 - qfunction(observation, action))
                thetas += [change * d for d in delta(observation, action)]
                break
            action_1 = take_action(observation_1)
            qdash = qfunction(observation_1, action_1)
            q = qfunction(observation, action)
            change = alpha * ((reward_1 + gamma * qdash) - q)
            thetas += [change * d for d in delta(observation, action)]
            observation = observation_1
            action = action_1
        if i_episode % 10 == 0:
            cost2Go.append(-reward_total / 10)
            print("Episode# {} finished with avg. rewards {}".format(i_episode, t + 1, reward_total / 10))
            reward_total = 0
        else:
            print("Episode# {} finished after {} timesteps with total rewards {}".format(i_episode, t + 1, reward_total))
        #print("Learnt Theta{}".format(thetas))
    env.close()
    q = raw_input("Want to upload your result to gym [Y/N]: ")
    if q == 'Y':
        gym.upload('/tmp/mountaincar-experiment-1', api_key='sk_nZENylvyQfaNih2pHP2qWA')
    plt.plot(cost2Go)
    plt.ylabel('cost to go')
    plt.show()

if __name__ == "__main__":
    learn()

