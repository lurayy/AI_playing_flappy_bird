import numpy as np 
import pygame 

from game import Game 
from deep_q_network import Agent

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
                print("Episode : ",episode, " Total reward : ",total_reward)
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("QUITING THE GAME")
                    quit = True
                    done = True
        if (quit):
            break
    pygame.quit()
