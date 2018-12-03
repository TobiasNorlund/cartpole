import gym
import numpy as np
import pandas as pd
import pandas
import random
import math

# bins_postion = np.array([-4, -2, 0, 2, 4])
# bins_velocity = np.linspace(-1.5, 1.5, num=10)
# bins_theta = np.linspace(-0.3, 0.3, num=10)
# bins_theta_velocity = np.linspace(-2, 2, num=10)
n_bins = 10
bins_postion = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
bins_theta = pandas.cut([-2, 2], bins=6, retbins=True)[1][1:-1]
bins_velocity = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
bins_theta_velocity = pandas.cut(
    [-3.5, 3.5], bins=12, retbins=True)[1][1:-1]


def get_bin(value, bins):
    return int(np.digitize(value, bins))


env = gym.make('CartPole-v0')
observation = env.reset()
gamma = 1.0
# epsilon = 0.1
# alpha = 0.5
q_table = np.zeros((1, len(bins_velocity)+1,
                    len(bins_theta)+1, len(bins_theta_velocity)+1, 2))
for t in range(500):
    print('new')
    observation = env.reset()
    done = False
    score = 0
    while done is False:
        env.render()
        epsilon = max(
            0.1, min(1, 1.0 - math.log10((t + 1) / 25)))
        alpha = max(
            0.1, min(1, 1.0 - math.log10((t + 1) / 25)))
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # take a random action
            # print('random action', action)
        else:
            action = np.argmax(q_table[0,
                                       0,
                                       get_bin(observation[2], bins_theta),
                                       get_bin(observation[3], bins_theta_velocity), :])

        # print([get_bin(observation[1], bins_velocity),
        #       get_bin(observation[2], bins_theta),
        #       get_bin(observation[3], bins_theta_velocity)])
        new_observation, reward, done, info = env.step(action)
        print('reward', reward)
        score += reward
        # print('new_observation', new_observation)
        q = q_table[0,
                    0,
                    get_bin(observation[2], bins_theta),
                    get_bin(observation[3], bins_theta_velocity), action]

        q_prim = np.max(q_table[0,
                                0,
                                get_bin(new_observation[2], bins_theta),
                                get_bin(new_observation[3], bins_theta_velocity), :])

        # self.Q[state_old][action] += alpha * \
        #            (reward + self.gamma *
        #             np.max(self.Q[state_new]) - self.Q[state_old][action])

        # print('update', (1-alpha) * q + alpha*(reward + gamma * q_prim))
        if q == 0:
            q_table[0,
                    0,
                    get_bin(observation[2], bins_theta),
                    get_bin(observation[3], bins_theta_velocity), action] = reward

        else:
            q_table[0,
                    0,
                    get_bin(observation[2], bins_theta),
                    get_bin(observation[3], bins_theta_velocity), action] += alpha * (reward + gamma * q_prim-q)

        print('self.Q[state_old][action]', q_table[0,
                                                   0,
                                                   get_bin(
                                                       observation[2], bins_theta),
                                                   get_bin(observation[3], bins_theta_velocity), (action)])

        observation = new_observation
    print('time we live', score)


print(q_table)
