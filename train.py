from itertools import cycle
import random
import sys
import numpy as np
import math
import time
from game import Game
import pygame
from pygame.locals import *


if __name__ == '__main__':
    game = Game()
    run = True
    while run:
        action = 0
        game.init_elements()
        while run:    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_SPACE:
                        game.frame_step(1)
            state, reward = game.frame_step(0)
            print(state, " with reward = ",reward)
    pygame.quit()
