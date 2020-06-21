# -*- coding: utf-8 -*-

# Copyright 2020 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Distributed Experience Replay with Ray
#
# Distributed Q Learning with Actor / Learner architecture.
#
# https://arxiv.org/abs/1803.00933
# https://github.com/werner-duvaud/muzero-general


import time
import copy
import numpy as np
import gym
import ray
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class Network:
    def __init__(self, obs_size=4, action_size=2, lr=0.001): # TODO
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=obs_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss=Huber(), optimizer=Adam(lr=lr))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict(self, obs):
        return self.model.predict(obs)

    def fit(self, inputs, targets, batch_size=64, epochs=10, verbose=0): # TODO
        return self.model.fit(x=inputs, y=targets, batch_size=batch_size, epochs=epochs, verbose=verbose)


@ray.remote
class ReplayMemory:
    def __init__(self):
        self.memory = {}
        self.episode_count = 0
        self.transition_count = 0
        self.window_size = 10000 # TODO
        self.batch_size = 64 # TODO

    def get_episode_count(self):
        return self.episode_count # number of episodes

    def get_transition_count(self):
        return self.transition_count # number of transitions

    def save_episode(self, transitions):
        self.memory[self.episode_count] = transitions
        self.episode_count += 1
        self.transition_count += len(transitions)
        # delete old data with FIFO
        if self.window_size < len(self.memory):
            del_id = self.episode_count - len(self.memory)
            del self.memory[del_id]

    def get_batch(self):
        batch = []
        for _ in range(self.batch_size):
            episode_id, transitions = self.sample_episode()
            position = self.sample_position(transitions)
            batch.append(transitions[position]) # TODO
        return batch

    def sample_episode(self):
        index = np.random.choice(len(self.memory)) # TODO
        episode_id = self.episode_count - len(self.memory) + index
        return episode_id, self.memory[episode_id]

    def sample_position(self, transitions):
        position_index = np.random.choice(len(transitions)) # TODO
        return position_index


@ray.remote
class SharedStorage:
    def __init__(self, weights):
        self.weights = weights
        self.infos = {
            'learning_step': 0,
            'episode_length': 0,
            'loss': 0
        }

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value


@ray.remote
class Actor:
    def __init__(self, weights, env, epsilon, gamma):
        self.model = Network()
        self.model.set_weights(weights)
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma

    def play_loop(self, storage, replay_memory):
        while True:
            weights = ray.get(storage.get_weights.remote())
            self.model.set_weights(copy.deepcopy(weights))
            transitions = self.play_episode()
            episode_length = len(transitions)
            storage.set_infos.remote('episode_length', episode_length)
            replay_memory.save_episode.remote(transitions)

    def play_episode(self):
        transitions = []
        obs = self.env.reset()
        done = False
        t = 0
        while not done:
            action = self.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            t += 1
            if done:
                next_obs = None
            reward = 1 if done and t > 190 else 0 # TODO: CartPole specific
            gamma = 0 if done else self.gamma
            transitions.append([obs, action, reward, gamma, next_obs])
            obs = next_obs
        return transitions

    def select_action(self, obs):
        if self.epsilon > np.random.rand():
            action = self.env.action_space.sample()
        else:
            obs = np.reshape(obs, (1, -1))
            action = np.argmax(self.model.predict(obs)[0])
        return action


@ray.remote
class Learner:
    def __init__(self, weights):
        self.model = Network()
        self.model.set_weights(weights)
        self.learning_step = 0

    def learn_loop(self, storage, replay_memory):
        while ray.get(replay_memory.get_episode_count.remote()) < 1: # TODO
            time.sleep(0.1)
        while True:
            batch = ray.get(replay_memory.get_batch.remote())
            history = self.update_weights(batch)
            weights = self.model.get_weights()
            storage.set_weights.remote(weights)
            storage.set_infos.remote('learning_step', self.learning_step)
            storage.set_infos.remote('loss', history.history['loss'][-1])
            # time.sleep(1.0)

    def update_weights(self, batch):
        batch_size = len(batch)
        inputs = np.zeros((batch_size, 4)) # TODO
        targets = np.zeros((batch_size, 2)) # TODO
        for i, (obs, action, reward, gamma, next_obs) in enumerate(batch):
            obs = np.reshape(obs, (1, -1)) # batch with length 1
            inputs[i] = obs
            targets[i] = self.model.predict(obs)
            # target = reward + gamma * argmax_a(q(s', a))

            if gamma > 0.0: # if gamma == 0, it's end of episode, then q(s', a) == 0
                next_obs = np.reshape(next_obs, (1, -1)) # batch with length 1
                next_q = self.model.predict(next_obs)[0]
                target = reward + gamma * np.amax(next_q)
            else: # gamma == 0
                target = reward
            targets[i][action] = target
        history = self.model.fit(inputs, targets)
        self.learning_step += 1
        return history


def epsilon_i(n, i, epsilon=0.4, alpha=7): # TODO
    return epsilon ** (1 + i / (n - 1) * alpha) # see Ape-X paper, page 6.


def main():
    # ray.init(address='auto') # for cluster
    ray.init() # for local machine

    weights = Network().get_weights()

    replay_memory = ReplayMemory.remote()
    storage = SharedStorage.remote(copy.deepcopy(weights))

    learner_worker = Learner.remote(copy.deepcopy(weights))
    learner_worker.learn_loop.remote(storage, replay_memory) # main loop for learning

    n = 4 # TODO
    gamma = 0.99 # TODO
    actor_workers = [Actor.remote(copy.deepcopy(weights), gym.make('CartPole-v0'), epsilon_i(n, i), gamma) for i in range(n)]
    [actor_worker.play_loop.remote(storage, replay_memory) for actor_worker in actor_workers] # main loop for self play

    try:
        start_time = time.time()
        while True:
            infos = ray.get(storage.get_infos.remote())
            transition_count = ray.get(replay_memory.get_transition_count.remote())
            duration = time.time() - start_time
            print('{:.2f}, {}, {}, {}, {:.7f}, {:.5f}, {:.5f}'.format(duration, infos['learning_step'], transition_count, infos['episode_length'], infos['loss'], infos['learning_step']/duration, transition_count/duration))
            time.sleep(1.0)
    except KeyboardInterrupt as err:
        pass

    ray.shutdown()

if __name__ == '__main__':
    main()
