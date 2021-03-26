import matplotlib.pyplot as plt
import numpy as np
import os
import random

from math import pi

class Slipstream:
    def __init__(self, plot=False):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.field_n_rows = 32
        self.field_n_cols = 12
        self.screen_n_rows = self.field_n_rows
        self.n_players = 4
        self.screen_n_cols = self.field_n_cols + self.n_players
        self.max_time = 48
        self.n_actions = 5

        # variables
        self.reset()

        # animation
        self.plot = plot
        if self.plot:
            plt.ion()
            self.fig = plt.figure(figsize=[3, 7])
        self.img_cnt = 0

    def play_action(self, id_player, chara, action):
        if action == 0: # 1 dash
            if self.player_energy[id_player] > 0:
                self.player_row[id_player] += 2
                self.player_energy[id_player] -= 1
            else: # no leg
                self.player_row[id_player] += 1
                self.player_col[id_player] = self.player_col[id_player]
        elif action == 1: # move left
            self.player_row[id_player] += 1
            self.player_col[id_player] -= 2
            if self.player_col[id_player] < 0:
                self.player_col[id_player] += self.field_n_cols
        elif action == 2: # go straight
            self.player_row[id_player] += 1
            self.player_col[id_player] = self.player_col[id_player]
        elif action == 3: # move right
            self.player_row[id_player] += 1
            self.player_col[id_player] += 2
            if self.player_col[id_player] > self.field_n_cols - 1:
                self.player_col[id_player] -= self.field_n_cols
        elif action == 4: # super dash
            if self.player_energy[id_player] >= chara[2]:
                self.player_row[id_player] += chara[2] + 1
                self.player_energy[id_player] -= chara[2]
            elif self.player_energy[id_player] > 0:
                self.player_row[id_player] += self.player_energy[id_player] + 1
                self.player_energy[id_player] = 0
            else: # no leg
                self.player_row[id_player] += 1
                self.player_col[id_player] = self.player_col[id_player]
        else:
            pass

    def update(self, action):
        chara0 = [1, 1, 3] # Cavendish
        chara1 = [1, 1, 2] # Petacchi
        chara2 = [1, 1, 1] # Renshow
        chara3 = [1, 1, 1] # Zabel

        # Cavendish(energy=5)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   |   |   |   |
        # |   |   | 4 |   |   |
        # Renshow(energy=5)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   | 4 |   |   |
        # |   |   |   |   |   |
        if action[0]//self.n_actions == 0: # 1 dash
            self.play_action(0, chara0, 0)
            self.play_action(2, chara2, action[0]%self.n_actions)
        elif action[0]//self.n_actions == 1: # move left
            self.play_action(0, chara0, 1)
            self.play_action(2, chara2, action[0]%self.n_actions)
        elif action[0]//self.n_actions == 2: # go straight
            self.play_action(0, chara0, 2)
            self.play_action(2, chara2, action[0]%self.n_actions)
        elif action[0]//self.n_actions == 3: # move right
            self.play_action(0, chara0, 3)
            self.play_action(2, chara2, action[0]%self.n_actions)
        elif action[0]//self.n_actions == 4: # super dash
            self.play_action(0, chara0, 4)
            self.play_action(2, chara2, action[0]%self.n_actions)
        else:
            # do nothing
            pass

        # Petacchi(energy=6)
        # |   |   | P |   |   |
        # |   | 1 | 2 | 3 |   |
        # |   |   | 0 |   |   |
        # |   |   | 4 |   |   |
        # |   |   |   |   |   |
        act_rand = random.randint(0, 25)
        if act_rand//self.n_actions == 0: # 1 dash
            self.play_action(1, chara1, 0)
            self.play_action(3, chara3, act_rand%self.n_actions)
        elif act_rand//self.n_actions == 1: # move left
            self.play_action(1, chara1, 1)
            self.play_action(3, chara3, act_rand%self.n_actions)
        elif act_rand//self.n_actions == 2: # go straight
            self.play_action(1, chara1, 2)
            self.play_action(3, chara3, act_rand%self.n_actions)
        elif act_rand//self.n_actions == 3: # move right
            self.play_action(1, chara1, 2)
            self.play_action(3, chara3, act_rand%self.n_actions)
        elif act_rand//self.n_actions == 4: # super dash
            self.play_action(1, chara1, 3)
            self.play_action(3, chara3, act_rand%self.n_actions)
        else:
            # do nothing
            pass

        # slip stream
        for i in range(self.n_players):
            recover1 = False
            recover2 = False
            for j in range(self.n_players):
                if i != j:
                    if self.player_row[i] == self.player_row[j] - 1 and abs(self.player_col[i] - self.player_col[j]) <= 1:
                        recover2 = True
                        break
                    elif self.player_row[i] == self.player_row[j] - 2 and abs(self.player_col[i] - self.player_col[j]) <= 1:
                        recover1 = True
            if recover2:
                self.player_energy[i] += 2
            elif recover1:
                self.player_energy[i] += 1

        # collision detection
        self.terminal = False
        for i in range(self.n_players):
            if self.player_row[i] >= self.field_n_rows - 1 and self.player_team[i] == 0:
                self.terminal = True
                self.reward[0] = ((self.field_n_rows - 7) - self.time)*0.1
                self.reward[2] = ((self.field_n_rows - 7) - self.time)*0.1
                break

        # ace        
        # if self.player_row[0] >= self.field_n_rows - 1:
        #     self.terminal = True
        #     self.reward[0] = ((self.field_n_rows - 7) - self.time)*0.1
        #     self.reward[2] = ((self.field_n_rows - 7) - self.time)*0.1

        # update time
        if self.time < self.max_time - 1:
            self.time += 1

    def draw(self):
        color = [1.0, 0.5, 0.75, 0.25]

        # try:
        for i in range(self.n_players):
            # draw player
            if self.player_row[i] > self.field_n_rows - 1:
                self.player_row[i] = self.field_n_rows - 1
            self.screen[self.player_row[i], self.player_col[i], self.time] = color[i]

            # draw energy consumption
            if self.player_energy[i] > 0:
                val = min(self.player_energy[i], self.screen_n_rows - 1)
                for j in range(val):
                    self.screen[j, self.screen_n_cols - self.n_players + i, self.time] = color[i]
        # except:
        #     print("exception: %d, %d, %d, %d"%(self.player_col[0], self.player_col[1], self.player_col[2], self.player_col[3]))

    def observe(self, show=False):
        self.draw()
        if self.plot and show:
            self.update_plot()
        # self.save_images()
        return self.screen, self.reward, self.terminal

    def step(self, action):
        self.update(action)
        return self.screen, self.reward, self.terminal

    def rand_ints_nodup(self, a, b, k):
        ns = []
        while len(ns) < k:
            n = random.randint(a, b)
            if not n in ns:
                ns.append(n)
        return ns

    def reset(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols, self.max_time))
        self.time = 0

        # reset player position
        self.player_row = np.zeros(self.n_players, dtype=np.int8)
        self.player_col = self.rand_ints_nodup(0, self.field_n_cols - 1, self.n_players)
        
        # reset other variables
        self.reward = np.ones(self.n_players)*-1
        self.terminal = False
        self.player_energy = np.zeros(self.n_players, dtype=np.int8)
        self.player_energy[0] = 5
        self.player_energy[1] = 0
        self.player_energy[2] = 5
        self.player_energy[3] = 0
        self.player_team = np.zeros(self.n_players, dtype=np.int8)
        self.player_team[0] = 0
        self.player_team[1] = 1
        self.player_team[2] = 0
        self.player_team[3] = 1

        self.image_buf = []

    def update_plot(self):
        self.fig.clear()
        plt.imshow(self.screen[:, :, self.time], cmap="jet", vmin=0.0, vmax=1.0)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.pause(0.1)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def save_images(self):
        plt.imshow(self.screen[:, :, self.time], cmap="jet", vmin=0.0, vmax=1.0)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        #plt.show()
        plt.savefig("./gif11/%03d.png"%self.img_cnt)
        self.img_cnt += 1
    
    def store_images(self):
        self.image_buf.append(self.screen[:, :, self.time])

    def release_images(self):
        self.image_buf.clear()
    
    def output_images(self, index, directory):
        for i, image in enumerate(self.image_buf):
            plt.imshow(image, cmap="jet", vmin=0.0, vmax=1.0)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            #plt.show()
            plt.savefig(directory + "/%d_%02d.png"%(index, i))
            self.img_cnt += 1
