import gym
import numpy as np
import pandas as pd
import random

bins_postion = np.array([-4, -2, 0, 2, 4])
bins_velocity = np.array([-4, -2, 0, 2, 4])
bins_theta = np.array([-4, -2, 0, 2, 4])
bins_theta_velocity = np.array([-0.3, -0.1, 0, 0.1, 0.3])


def get_bin(value, bins):
    return int(np.digitize(value, bins))


env = gym.make('CartPole-v0')
observation = env.reset()
gamma = 0.9
epsilon = 0.7
alpha = 0.2
q_table = np.zeros((len(bins_postion)+1, len(bins_velocity)+1,
                    len(bins_theta)+1, len(bins_theta_velocity)+1, 2))
for _ in range(10000):
    observation = env.reset()
    done = False
    score = 0
    while done is False:
        # env.render()
        if random.uniform(0, 1) > epsilon:
            action = env.action_space.sample()  # take a random action
        else:
            action = np.argmax(q_table[get_bin(observation[0], bins_postion),
                                       get_bin(observation[1], bins_velocity),
                                       get_bin(observation[2], bins_theta),
                                       get_bin(observation[3], bins_theta_velocity), :])
        new_observation, reward, done, info = env.step(action)
        score += reward
        q = q_table[get_bin(observation[0], bins_postion),
                    get_bin(observation[1], bins_velocity),
                    get_bin(observation[2], bins_theta),
                    get_bin(observation[3], bins_theta_velocity), (action)]

        q_prim = np.max(q_table[get_bin(new_observation[0], bins_postion),
                                get_bin(new_observation[1], bins_velocity),
                                get_bin(new_observation[2], bins_theta),
                                get_bin(new_observation[3], bins_theta_velocity), :])

        q_table[get_bin(observation[0], bins_postion),
                get_bin(observation[1], bins_velocity),
                get_bin(observation[2], bins_theta),
                get_bin(observation[3], bins_theta_velocity), (action)] = (1-alpha) * q + alpha*(reward + gamma * q_prim)
    print('time we live', score)

print(q_table)
