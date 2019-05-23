import random 
import numpy as np 
import time 
import pygame 
from pygame.locals import *
from itertools import cycle

from game import Game 

import tensorflow as tf 


class Agent:

    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(24, input_shape = (observation_space,), activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(24,activation = 'relu'))
        self.model.add(tf.keras.layers.Dense(self.action_space, activation = "linear"))
        
        json_file = open("models/dqn_with_er.json",'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights("models/dqn_with_er.h5")

    def get_action(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])


if __name__ == "__main__":
    game = Game()

    observation_space = 11
    action_space = 2 
    render = False

    agent = Agent(observation_space, action_space)
    
    for episode in range(2):
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
            readings = next_readings
            if done: 
                print("Episode : ",episode, " Total reward : ",total_reward, 'of current reward :',rewards)
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUITING THE GAME")
                    quit = True
                    done = True
        if (quit):
            break
    pygame.quit()
