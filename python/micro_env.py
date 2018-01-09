import os
import gym

import pygame as pg
import numpy as np

from gym import error, spaces
from gym import utils
from gym.utils import seeding
from copy import deepcopy

from game_utils import *
from math import *
from random import *

class GameEnv(gym.Env):
    def __init__(self, visualization=False):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/60.0

        self.world_shape = (640, 480)
        self.params = (max_dw, max_ds,friction_k, dt)

        max_speed = sqrt(max_ds / friction_k)
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))

        ob_low  = [0.0,  0.0,\
                  -max_speed,-max_speed,\
                  -max_ds, -max_ds,\
                  -2.01*pi , -max_dw,\
                   0.0,  0.0,\
                   0.0, 0.0 ]

        ob_high = [self.world_shape[0], self.world_shape[1],\
                   max_speed, max_speed, max_ds, max_ds,\
                   2.01*pi, max_dw, \
                   self.world_shape[0], self.world_shape[1], \
                   self.world_shape[0], self.world_shape[1]]
        
        self.observation_space = spaces.Box(low=np.array(ob_low), high=np.array(ob_high))

        self._spec = EnvSpec(timestep_limit = 3000, id=1)
        self._reset()

        if visualization == True:
            pg.init()
            self.surface = pg.display.set_mode((self.world_shape[0], self.world_shape[1]), 16)
            self.player_sprite = pg.image.load("Arrow.png").convert()
            self.target_sprite = pg.image.load("Circle.png").convert()
            self.font = pg.font.SysFont("Times New Roman",12)
            self.metadata["render.modes"].append("human")

    def _step(self, action): 
        alpha = action[0]
        beta = action[1]
        reward = 0.0
        
        k = alpha*self.params[1]
        dt = self.params[3]

        dd1x = self.player[0] - self.target_1[0]
        dd1y = self.player[1] - self.target_1[1]
        dd2x = self.player[0] - self.target_2[0]
        dd2y = self.player[1] - self.target_2[1]

        d1 = sqrt(dd1x*dd1x + dd1y*dd1y)
        d2 = sqrt(dd2x*dd2x + dd2y*dd2y)

        #Check targets
        if d1 <= 40.0:
            self.target_1[0] = randint(0, self.world_shape[0])
            self.target_1[1] = randint(0, self.world_shape[1])
            reward += 1.0
    
        if d2 <= 40.0:
            self.target_2[0] = randint(0, self.world_shape[0])
            self.target_2[1] = randint(0, self.world_shape[1])
            reward += 1.0

        reward -= 1.0/1000
        self.score += reward

        a = [0.0, 0.0]
        speed_abs = (self.player[2]*self.player[2]+ self.player[3]*self.player[3])
        a[0] = k*cos(self.player[6]) - self.params[2]*speed_abs*self.player[2]
        a[1] = k*sin(self.player[6]) - self.params[2]*speed_abs*self.player[3]

        self.player[6] += self.player[7]*dt
        self.player[0] += self.player[2]*dt
        self.player[1] += self.player[3]*dt

        self.player[2] += self.player[4]*dt
        self.player[3] += self.player[5]*dt

        self.player[4] = a[0]
        self.player[5] = a[1]

        self.player[7] = beta * self.params[0]

        # World reaction
        
        if self.player[0] < 0.0: self.player[0] = 0.0
        if self.player[1] < 0.0: self.player[1] = 0.0
        if self.player[0] > self.world_shape[0]: self.player[0] = self.world_shape[0]
        if self.player[1] > self.world_shape[1]: self.player[1] = self.world_shape[1]

        if self.player[6] >=  2.0*pi: self.player[6] -= 2.0*pi
        if self.player[6] <= -2.0*pi: self.player[6] += 2.0*pi

        done = False
        if self.score < -10.0: done = True
        ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              self.target_1[0], self.target_1[1],\
              self.target_2[0], self.target_2[1]]

        info = dict()
        return np.array(ob), reward, done, info 

        
    def _reset(self): 
        self.player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.target_1 = [0.0, 0.0]
        self.target_2 = [0.0, 0,0]

        self.player[0] = randint(0, self.world_shape[0]) 
        self.player[1] = randint(0, self.world_shape[1]) 
        self.player[6] = uniform(0.0, 2.0*pi)  

        self.target_1[0] = randint(0, self.world_shape[0])
        self.target_1[1] = randint(0, self.world_shape[1])

        self.target_2[0] = randint(0, self.world_shape[0])
        self.target_2[1] = randint(0, self.world_shape[1])

        self.score = 0
        ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              self.target_1[0], self.target_1[1],\
              self.target_2[0], self.target_2[1]]\

        return np.array(ob)

    def _render(self, mode='human', close=False):
        if mode == 'human':
            self.surface.fill((255,255,255))
            phi = -(180.0/pi)*self.player[6] - 90.0
            im =  pg.transform.rotate(self.player_sprite, phi)
            h = im.get_height()/2.0
            w = im.get_width()/2.0
            im.set_colorkey((0,128,0))
            self.surface.blit(im, (self.player[0]-w,self.player[1]-h))

            im = self.target_sprite
            h = im.get_height()/2.0
            w = im.get_width()/2.0
            im.set_colorkey((0,128,0))

            self.surface.blit(im, (self.target_1[0]-w, self.target_1[1]-h))
            self.surface.blit(im, (self.target_1[0]-w, self.target_1[1]-h))
            pg.display.flip()

    def _seed(self, seed=None): 
        return []

    def _close(self):
        pass
