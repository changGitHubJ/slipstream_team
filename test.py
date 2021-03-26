from __future__ import division

import argparse
import numpy as np
import os
import random
import shutil
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from slipstream import Slipstream
from dqn_agent import DQNAgent

def main(trial, epoch):
    directory = "./gif" + str(trial)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)        

    # environmet, agent
    env = Slipstream(plot=False)
    agent1 = DQNAgent(np.linspace(0, 24, 25, dtype=np.int32), [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCavendish%d_%d"%(trial, epoch))
    agent1.load_model()

    count_game = 0
    REWARD = []
    counter = 0
    while True:
        state, reward, terminal = env.observe(show=True)
        env.store_images()

        if terminal:
            REWARD.append([reward[0], counter])
            msg = "REWARD: {:.3f}".format(reward[0])
            print(msg)
            if reward[0] >= 0.8:
                env.output_images(counter, directory)
            env.release_images()
            env.reset()
            count_game += 1
            if count_game >= 300:
                break
        else:
            action_t_1 = agent1.select_action(state.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), 0.0) #agent1.exploration)
            env.step([action_t_1])
        counter += 1
    
    with open(directory + "/score.txt", "w") as f:
        tmp = np.array(REWARD)
        REWARD = tmp[:, 0]
        INDEX = tmp[:, 1]
        best_score = REWARD.max()
        f.write("max = %.2f\n"%best_score)
        f.write("average = %.2f\n"%np.average(REWARD))
        f.write("18 step\n")
        heads = np.where(REWARD == 0.8)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")
        f.write("17 step\n")
        heads = np.where(REWARD == 0.9)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")
        f.write("16 step\n")
        heads = np.where(REWARD == 1.0)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")
        f.write("15 step\n")
        heads = np.where(REWARD == 1.1)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")
        f.write("14 step\n")
        heads = np.where(REWARD == 1.2)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")
        f.write("13 step\n")
        heads = np.where(REWARD == 1.3)
        for head in heads:
            f.write(str(INDEX[head]) + "\n")

if __name__ == "__main__":
    args = sys.argv
    if len(args) >= 2:
        trial = int(args[1])
    else:
        trial = 0

    main(trial, 99999)

