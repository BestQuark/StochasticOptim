"""
Author: @BestQuark
Summary: This code was written for the final project of the 2021-10 stochastic optimization course at uniandes.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

def epsilon_greedy(env, algorithm = 0, ntry=0, epsilon=0.2, gamma=0.9, episodes = 20000, lr = 0.2, save_every= 0, q_table = None, q_table_low = 1, q_table_high = 2, render = False, extra={}):
    """
    Implements epsilon_greedy qlearning algorithm

    algorithm=0 is q-learning
    algorithm=1 is SARSA
    """
    assert algorithm==1 or algorithm==0, "Invalid algorithm"

    if q_table is not None:
        q_table = q_table
    else:
        q_table = np.random.uniform(low =q_table_low, high=q_table_high, size = (env.size, env.size, env.action_space.n) )

    scores = []

    for episode in range(episodes):
        
        if save_every==0:
            pass
        elif episode%save_every==0 or episode==episodes-1:
            try:
                print("Episode: ",episode+1)
                np.save(f"qtables/qtable_try_{ntry}_ep_{episode+1}.npy", q_table)
            except:
                print("Episode: ",episode+1)
                print("Could not save qtable")

        state = env.reset()
        done = False
        score = 0
        
        while not done:
            #epsilon-greedy
            x,y = state
            if np.random.random() > epsilon:
                action = np.argmax(q_table[x,y,:])
            else:
                action = np.random.randint(0, env.action_space.n)
                    
            new_state, reward, done, info = env.step(action)
            score += reward
            if render:
                if episodes==1 and extra.keys():
                    print("Episode: ", extra["n_ep"]+1, "\nScore: ", score, "\nEpsilon: ", np.around(epsilon, decimals=2))
                else:
                    print("Episode: ", episode+1 , "\nScore: ", score)
                env.render()

            nx, ny = new_state
            if not done:
                #updates using q-learning
                if algorithm==0:
                    next_q = np.max(q_table[nx,ny,:])

                ##updates using SARSA
                elif algorithm==1:
                    if np.random.random() > epsilon:
                        action2 = np.argmax(q_table[nx,ny,:])
                    else:
                        action2 = np.random.randint(0, env.action_space.n)
                    next_q = q_table[nx,ny, action2]

                q_table[x,y, action] = (1-lr)*q_table[x,y,action] + lr*(reward + gamma*next_q)
            
            state = new_state        
        
        scores.append(score)
        
    env.close()

    return q_table, scores


def decaying_epsilon_greedy(env, algorithm = 0, ntry=0, max_epsilon=0.5, min_epsilon=0.1, step_last_dacay = 10000, gamma=0.9, episodes = 20000, lr = 0.3, save_every= 2000, q_table = None, q_table_low = 1, q_table_high = 2, render = False):
    """
    Implements epsilon_greedy qlearning algorithm

    algorithm=0 is q-learning
    algorithm=1 is SARSA
    """
    assert step_last_dacay<=episodes, "step_last_decay > episodes"
    if q_table is not None:
        q_table = q_table
    else:
        q_table = np.random.uniform(low =q_table_low, high=q_table_high, size = (env.size, env.size, env.action_space.n) )

    scores = []

    for episode in range(episodes):
        if save_every==0:
            pass
        elif episode%save_every==0 or episode==episodes-1:
            try:
                print("Episode: ", episode+1)
                np.save(f"qtables/qtable_ep_{episode+1}_try{ntry}.npy", q_table)
            except:
                print("Episode: ", episode+1)
                print("Could not save qtable")

        p = min(1, episode/step_last_dacay)
        epsilon = max_epsilon*(1-p) + min_epsilon*p
        q_table, score = epsilon_greedy(env, algorithm = algorithm, epsilon=epsilon, gamma=gamma, episodes = 1, lr = lr, save_every= 0, q_table = q_table, render = render, extra={"n_ep": episode})
        assert len(score)==1, "Error"
        scores.append(score[0])

    return q_table, scores


        


        