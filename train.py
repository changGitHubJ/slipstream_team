import numpy as np
import sys
import random

from collections import deque

from slipstream import Slipstream
from dqn_agent import DQNAgent

def main(trial):
    # parameters
    n_epochs = 100000

    # environment, agent
    env = Slipstream(plot=False)
    agent1 = DQNAgent(np.linspace(0, 24, 25, dtype=np.int32), [env.screen_n_cols, env.screen_n_rows, env.max_time], env.name, "modelCavendish" + str(trial))
    agent1.compile()

    training_log = []
    buf1 = deque()
    for e in range(n_epochs):
        # reset
        env.reset()
        show = False
        #if e%200 == 0:
        show = True
        state_t_1, reward_t, terminal = env.observe(show=show)

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t_1 = agent1.select_action(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), agent1.exploration)
            env.step([action_t_1])

            # observe environment
            state_t_1, reward_t, terminal = env.observe(show=show)

            # store experience
            buf1.append((state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time), action_t_1, state_t_1.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time)))

            if terminal:
                reward_t_1 = reward_t[0]
                agent1.store_experience(buf1, reward_t_1)
                agent1.experience_replay()
                buf1.clear()

            # for log
            loss1 = agent1.current_loss
            Q_max1 = np.max(agent1.Q_values(state_t.reshape(env.screen_n_cols*env.screen_n_rows*env.max_time)))
            
        msg = "EPOCH: {:03d}/{:03d} | REWARD: {:.3f} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(e, n_epochs - 1, reward_t_1, loss1, Q_max1)
        print(msg)
        training_log.append(msg + "\n")

        # save model
        if (e + 1)%5000== 0 and e != 0:
            agent1.save_model(e)

    # save log
    with open("./training_log" + str(trial) + ".txt", "w") as f:
        for log in training_log:
            f.write(log)

def test():
    trial = 0
    main(trial)

if __name__ == "__main__":
    args = sys.argv
    trial = int(args[1])
    main(trial)

    # test()
    