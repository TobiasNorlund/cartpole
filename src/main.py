import gym
import numpy as np
import pandas as pd
import random

bins_postion = np.array([-4, -2, 0, 2, 4])
bins_velocity = np.linspace(-1.5, 1.5, num=10)
bins_theta = np.linspace(-0.3, 0.3, num=10)
bins_theta_velocity = np.linspace(-2, 2, num=10)


def get_bin(value, bins):
    return int(np.digitize(value, bins))


env = gym.make('CartPole-v0')
observation = env.reset()
gamma = 0.9
epsilon = 0.3
alpha = 0.5
q_table = np.zeros((1, len(bins_velocity)+1,
                    len(bins_theta)+1, len(bins_theta_velocity)+1, 2))
for _ in range(10000):
    observation = env.reset()
    done = False
    score = 0
    while done is False:
        # env.render()
        if random.uniform(0, 1) > epsilon:
            action = env.action_space.sample()  # take a random action
            # print('random action', action)
        else:
            action = np.argmax(q_table[0,
                                       get_bin(observation[1], bins_velocity),
                                       get_bin(observation[2], bins_theta),
                                       get_bin(observation[3], bins_theta_velocity), :])

        print([get_bin(observation[1], bins_velocity),
               get_bin(observation[2], bins_theta),
               get_bin(observation[3], bins_theta_velocity)])
        new_observation, reward, done, info = env.step(action)
        score += reward
        #print('new_observation', new_observation)
        q = q_table[0,
                    get_bin(observation[1], bins_velocity),
                    get_bin(observation[2], bins_theta),
                    get_bin(observation[3], bins_theta_velocity), (action)]

        q_prim = np.max(q_table[0,
                                get_bin(new_observation[1], bins_velocity),
                                get_bin(new_observation[2], bins_theta),
                                get_bin(new_observation[3], bins_theta_velocity), :])

        # print('update', (1-alpha) * q + alpha*(reward + gamma * q_prim))
        q_table[0,
                get_bin(observation[1], bins_velocity),
                get_bin(observation[2], bins_theta),
                get_bin(observation[3], bins_theta_velocity), (action)] = (1-alpha) * q + alpha*(reward + gamma * q_prim-q)
        observation = new_observation
    print('time we live', score)


print(q_table)
