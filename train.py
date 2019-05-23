from itertools import cycle
import random
import sys
import numpy as np
import math
import time
from game import Game
import pygame
from pygame.locals import *

import tensorflow as tf 
from collections import deque


learning_rate = 0.001
discount_rate = 0.95
memory_size = 1000000
batch_size = 20
exploration_max = 1.0
exploration_min = 0.01
exploration_decay = 0.995


class Agent:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = exploration_max
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen = memory_size)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape = (observation_space,), activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(24,activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(self.action_space, activation = "linear"))
        self.model.compile(loss= "mse", optimizer = tf.keras.optimizers.Adam(lr = learning_rate))
        
        json_file = open("models/dqn_with_er.json",'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights("models/dqn_with_er.h5")
        self.model.compile(loss= "mse", optimizer = tf.keras.optimizers.Adam(lr = learning_rate))

    def save_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def experience_replay(self):
        if len(self.memory)< batch_size:
            return
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            q_update = reward  
            if not done:
                q_update = reward + discount_rate*np.amax(self.model.predict(next_state))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose = 0)
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(exploration_min, self.exploration_rate)

    def save(self):
        model_json = self.model.to_json()
        with open("models/dqn_with_er.json","w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("models/dqn_with_er.h5")
        print("***************************** Model Saved ******************************")


if __name__ == "__main__":
    game = Game()

    observation_space = 11
    action_space = 2 
    episodes = 2500
    render = False

    agent = Agent(observation_space, action_space)
    
    for episode in range(episodes):
        readings, rewards, done = game.init_elements()
        done = False
        total_reward = 0
        rewards = 0
        quit = False
        while not (done):
            readings = np.reshape(readings,(1,observation_space))
            total_reward += rewards
            action = agent.get_action(readings)
            next_readings, rewards, done  = game.frame_step(action)
            next_readings = np.reshape(next_readings, (1,observation_space))
            agent.save_to_memory(readings, action, rewards, next_readings, done)
            readings = next_readings
            agent.experience_replay()
            if done: 
                print("Episode : ",episode, " Total reward : ",total_reward, 'of current reward :',rewards, " with exploration rate : ",agent.exploration_rate)
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUITING THE GAME")
                    quit = True
                    done = True

        if (episode % 50 == 0):
            agent.save()
        if (quit):
            break
    pygame.quit()
