"""
Author: @BestQuark
Summary: This code was written for the final project of the 2021-10 stochastic optimization course at uniandes.
"""

import drone
import qlearning as ql
import numpy as np
import gym
import matplotlib.pyplot as plt

#Choose environment
env = drone.droneEnv(size= 15, reward_loc=[[2,2]], start_seed=[2,2], randomize=True, fuel=30)


#CAMBIAR No INTENTO CADA RUN 
intento = 1
episodios = 5000000
epsilon = 0.2

#Implement epsilon greedy
qtable, scores = ql.epsilon_greedy(env, epsilon=epsilon, ntry=intento, save_every=episodios, episodes = episodios, render=False)

#Implements decaying epsilon greedy
#qtable, scores = ql.decaying_epsilon_greedy(env, max_epsilon=0.6, min_epsilon=epsilon, ntry=intento, step_last_dacay=episodios//2, save_every=episodios, episodes = episodios, render=False)

np.save(f"scores_try_{intento}.npy",scores)