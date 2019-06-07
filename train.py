import numpy as np
from game import Game
import pygame
from deep_q_network import Agent

if __name__ == "__main__":
    game = Game()

    observation_space = 11
    action_space = 2 
    episodes = 300
    render = False
    
    total_rewards = []
    agent = Agent(observation_space, action_space)
    
    for episode in range(episodes):
        readings, rewards, done = game.init_elements()
        done = False
        total_reward = 0
        quit = False
        while not (done):
            readings = np.reshape(readings,(1,observation_space))
            total_reward += rewards
            action = agent.get_action(readings)
            next_readings, rewards, done  = game.frame_step(action)
            next_readings = np.reshape(next_readings, (1,observation_space))
            agent.save_to_memory(readings, action, rewards, next_readings, done, episode)
            readings = next_readings
            agent.experience_replay()
            if done: 
                print("Episode : ",episode, " Total reward : ",total_reward, " with exploration rate : ",agent.exploration_rate)
                total_rewards.append(total_reward)
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
    agent.save_collected_data(episodes, total_rewards)
    pygame.quit()
