from collections import deque
import os

import numpy as np
import tensorflow as tf

from keras.layers import Dense, Flatten, InputLayer, Input, Convolution2D
from keras.models import Sequential

def build_model(input_shape, nb_output):
    model = Sequential()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,input_shape[0]*input_shape[1]], name="input")
    model.add(InputLayer(input_shape=(input_shape[0]*input_shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(nb_output))
    outputs = model(inputs)
    return inputs, outputs, model

# def build_model_cnn(input_shape, nb_output):
#     model = Sequential()
#     inputs = tf.placeholder(dtype=tf.float32, shape=[None,input_shape[0]*input_shape[1]*input_shape[2]], name="input")
#     model.add(InputLayer(input_shape=(input_shape[0], input_shape[1], input_shape[2])))
#     model.add(Convolution2D(16, 4, 4, border_mode='same', activation='relu', subsample=(2, 2)))
#     model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
#     model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
#     model.add(Dense(nb_output, activation='linear', kernel_initializer='he_normal', bias_initializer='zeros'))
#     outputs = model(tf.reshape(inputs, shape=[-1, input_shape[0], input_shape[1], input_shape[2]]))
#     return inputs, outputs, model

def build_model_cnn(input_shape, nb_output):
    model = Sequential()
    inputs = tf.placeholder(dtype=tf.float32, shape=[None,input_shape[0]*input_shape[1]*input_shape[2]], name="input")
    model.add(InputLayer(input_shape=(input_shape[0], input_shape[1], input_shape[2])))
    model.add(Convolution2D(16, 4, 4, border_mode='same', activation='relu', subsample=(2, 2), kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1), kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu', subsample=(1, 1), kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Dense(nb_output, activation='linear', kernel_initializer='he_normal', bias_initializer='zeros'))
    outputs = model(tf.reshape(inputs, shape=[-1, input_shape[0], input_shape[1], input_shape[2]]))
    return inputs, outputs, model

class DQNAgent:
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, input_shape, environment_name, model_name):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.gamma = 0.99
        self.minibatch_size = 16
        self.replay_memory_size = self.minibatch_size
        self.learning_rate = 0.00025
        self.discount_factor = 0.9
        self.exploration = 0.3
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
        self.model_name = "{}.ckpt".format(self.environment_name)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.sess = tf.InteractiveSession()
        self.model_inputs, self.model_outputs, self.model = build_model_cnn(input_shape, len(self.enable_actions))        
        self.target_model_inputs, self.target_model_outputs, self.target_model = build_model_cnn(input_shape, len(self.enable_actions))
        target_model_weights = self.target_model.trainable_weights
        model_weights = self.model.trainable_weights
        self.update_target_model = [target_model_weights[i].assign(.999*target_model_weights[i]+.001*model_weights[i]) for i in range(len(target_model_weights))]

        # saver
        self.saver = tf.train.Saver(max_to_keep=None)

        # variables
        self.current_loss = 0.0

    def compile(self):
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="target_q")
        self.inputs= tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        actions_one_hot = tf.one_hot(indices=self.inputs, depth=len(self.enable_actions), on_value=1.0, off_value=0.0, name="action_one_hot")

        pred_q = tf.multiply(self.model_outputs, actions_one_hot)
        error = self.targets - pred_q
        square_error = .5 * tf.square(error)
        self.loss = tf.reduce_mean(square_error, axis=0, name="loss")
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train = optimizer.minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())

    def Q_values(self, state):
        # Q(state, action) of all actions
        return self.sess.run(self.target_model_outputs, feed_dict={self.target_model_inputs: [state]})[0]

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # random
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            return self.enable_actions[np.argmax(self.Q_values(state))]

    def store_experience(self, experience, reward):
        self.D.append((experience, reward))

    def experience_replay(self):
        state_minibatch = []
        reward_minibatch = []
        action_minibatch = []
        state1_minibatch = []
        # terminal_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)

        for j in range(minibatch_size): # minibatch_indexes:
            game_length = len(self.D[j][0])
            reward_j = self.D[j][1]
            for i in range(game_length):
                state_j, action_j, state_j_1 = self.D[j][0][i]
                state_minibatch.append(state_j)
                reward_minibatch.append(reward_j * i/(game_length - 1))
                action_minibatch.append(action_j)
                state1_minibatch.append(state_j_1)
                #terminal_minibatch.append(0. if terminal els 1.)

        target_minibatch = [] #np.zeros((self.minibatch_size, self.n_actions))
        reward_minibatch = np.array(reward_minibatch)
        target_q_values = np.array(self.compute_target_q_value(state1_minibatch))   # compute maxQ'(s')
        discounted_reward_batch = (self.gamma * target_q_values)
        # discounted_reward_batch *= terminal_minibatch
        targets = reward_minibatch + discounted_reward_batch    # target = r + γ maxQ'(s')

        for (action, target) in (zip(action_minibatch, targets)):
            tmp_minibatch = np.zeros(self.n_actions)
            tmp_minibatch[action] = target
            target_minibatch.append(tmp_minibatch)

        # training
        state_minibatch = np.array(state_minibatch)
        action_minibatch = np.array(action_minibatch)
        target_minibatch = np.array(target_minibatch)
        self.train_on_batch(state_minibatch, action_minibatch, target_minibatch)
        self.sess.run(self.update_target_model)

    def clear_experience(self):
        self.D.clear()

    def train_on_batch(self, state_batch, action_batch, targets):
        self.sess.run(self.train, feed_dict={self.model_inputs: state_batch, self.inputs: action_batch, self.targets: targets})

        # for log
        loss = self.sess.run(self.loss, feed_dict={self.model_inputs: state_batch, self.inputs: action_batch, self.targets: targets})
        self.current_loss = np.max(loss)

    def compute_target_q_value(self, state1_batch):
        q_values = self.sess.run(self.target_model_outputs, feed_dict={self.target_model_inputs: state1_batch})
        q_values = np.max(q_values, axis=1)
        return q_values

    def compute_q_values(self, state):
        q_values = self.sess.run(self.target_model_outputs, feed_dict={self.target_model_inputs: [state]})
        return q_values[0]

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self, epoch):
        self.saver.save(self.sess, os.path.join(self.model_dir + "_%d"%epoch, self.model_name))
