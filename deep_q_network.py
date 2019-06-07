import tensorflow as tf 
from collections import deque
import math
import time
import numpy as np 
import random
import json 

learning_rate = 0.001
discount_rate = 0.95
memory_size = 1000000
batch_size = 20
exploration_max = 0.01
exploration_min = 0.0001
exploration_decay = 0.995


class Agent:

    def __init__(self, observation_space, action_space):

        self.exploration_rate = exploration_max
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen = memory_size)
        self.q_values_collection = []

        try:             
            json_file = open("models/dqn_with_er.json",'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = tf.keras.models.model_from_json(loaded_model_json)
            self.model.load_weights("models/dqn_with_er.h5")
            self.model.compile(loss= "mse", optimizer = tf.keras.optimizers.Adam(lr = learning_rate))
            print("Retriving Old Model")
        except:
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Dense(24, input_shape = (observation_space,), activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(24,activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(self.action_space, activation = "linear"))
            self.model.compile(loss= "mse", optimizer = tf.keras.optimizers.Adam(lr = learning_rate))
            print("Creating New Model")

    def save_to_memory(self, state, action, reward, next_state, done, episode_number):
        self.memory.append((state, action, reward, next_state, done, episode_number))
    

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    

    def experience_replay(self):
        if len(self.memory)< batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        x = 0
        y = int(random.randrange(0,batch_size))
        for state, action, reward, next_state, done, episode_number in batch:
            q_update = reward  
            if not done:
                q_update = reward + discount_rate*np.amax(self.model.predict(next_state))
            q_values = self.model.predict(state)
            old_q = q_values
            q_values[0][action] = q_update
            if x == y:
                self.save_q_values(old_q.tolist(), q_values.tolist(), episode_number)
            self.model.fit(state, q_values, verbose = 0)
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(exploration_min, self.exploration_rate)


    def save(self):
        model_json = self.model.to_json()
        with open("models/dqn_with_er.json","w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("models/dqn_with_er.h5")
        print("***************************** Model Saved ******************************")

    def save_collected_data(self,episodes,total_rewards):
        data_json = {'episodes': int(episodes), 'rewards_per_episode': total_rewards, 'q_values':self.q_values_collection}
        # try:
        print("here")
        
        with open('data/total_rewards.json','w') as json_file:
            json.dump(data_json, json_file)
        print("All data saved.")
        # except:
        #     print("Error Saving Rewards.")
    

    def save_q_values(self,old_q,new_q,episode_number):
        # print("old value : ",old_q, " new value : ", new_q)
        try:
            self.q_values_collection[int(episode_number)]['old_q'].append(old_q)
            self.q_values_collection[int(episode_number)]['new_q'].append(new_q)
        except:
            temp = {'old_q':[], 'new_q': []}
            self.q_values_collection.append(temp)
        