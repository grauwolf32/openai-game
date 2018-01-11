import os
import gym
import time

import pygame as pg
import numpy as np

from gym import error, spaces
from gym.utils import seeding
from gym import utils

from math import *
from random import *

def getAngle(x1,y1,x2,y2):
    s = x1*x2 + y1*y2
    v = x1*y2 - y1*x2

    l1 = sqrt(x1*x1 + y1*y1)
    l2 = sqrt(x2*x2 + y2*y2)
    pr = l1*l2

    if l1 < 10e-6 or l2 < 10e-6 or pr < 10e-5:
        return 0.0

    s /= pr
    alpha = acos(min(1.0, max(-1.0,s)))

    if v < 0.0:
        alpha *= -1.0
        
    return alpha 

class EnvSpec(object):
    def __init__(self, timestep_limit, id):
        self.timestep_limit = timestep_limit
        self.id = id

class GameEnv(gym.Env):
    def __init__(self, visualization=False):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0

        self.world_shape = (480, 360)
        self.params = (max_dw, max_ds, friction_k, dt)

        max_speed = sqrt(max_ds / friction_k)
        max_dist = sqrt(2.0) * max(self.world_shape[0], self.world_shape[1])

        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))

        ob_low  = [0.0,  0.0,\
                  -max_speed,-max_speed,\
                  -max_ds, -max_ds,\
                  -1.1*pi , -max_dw,\
                   -max_dist, -1.1*pi,\
                   -max_dist, -1.1*pi ]

        ob_high = [self.world_shape[0], self.world_shape[1],\
                   max_speed, max_speed, max_ds, max_ds,\
                   1.1*pi, max_dw, \
                   max_dist, 1.1*pi, \
                   max_dist, 1.1*pi]
        
        self.observation_space = spaces.Box(low=np.array(ob_low), high=np.array(ob_high))

        self._spec = EnvSpec(timestep_limit = 1500, id=1)
        self._seed()
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

        self.score += reward

        reward -= 1.0/1000
        old_phi = self.player[6]

        a = [0.0, 0.0]
        cpl = cos(self.player[6])
        spl = sin(self.player[6])

        speed_abs = sqrt(self.player[2]*self.player[2]+ self.player[3]*self.player[3])
        a[0] = k*cpl - self.params[2]*speed_abs*self.player[2]
        a[1] = k*spl - self.params[2]*speed_abs*self.player[3]

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

        reward -= abs(self.player[6] - old_phi)/ 100.0

        if self.player[6] >=  pi: self.player[6] -= 2.0*pi
        if self.player[6] <= -pi: self.player[6] += 2.0*pi

        done = False

        if self.score < -10.0: done = True

        t1a = getAngle(cpl, spl, -dd1x, -dd1y)
        t2a = getAngle(cpl, spl, -dd2x, -dd2y)

        if d1 <= d2:
            ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              d1, t1a,\
              d2, t2a]

        else:
            ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              d2, t2a,\
              d1, t1a]

        info = dict()
        return np.array(ob), reward, done, info 

        
    def _reset(self): 
        self.player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.target_1 = [0.0, 0.0]
        self.target_2 = [0.0, 0,0]

        rx = self.np_random.random_integers(0, self.world_shape[0],3)
        ry = self.np_random.random_integers(0, self.world_shape[1],3)

        self.player[0] = rx[0]
        self.player[1] = ry[0] 
        self.player[6] = self.np_random.uniform(0.0, 2.0*pi)  

        self.target_1[0] = rx[1]
        self.target_1[1] = ry[1]

        self.target_2[0] = rx[2]
        self.target_2[1] = ry[2]

        self.score = 0

        dx1 = (self.player[0] - self.target_1[0])
        dy1 = (self.player[1] - self.target_1[1])

        dx2 = (self.player[0] - self.target_2[0])
        dy2 = (self.player[1] - self.target_2[1])

        d1  = sqrt(dx1*dx1 + dy1*dy1)
        d2  = sqrt(dx2*dx2 + dy2*dy2)

        cpl = cos(self.player[6])
        spl = sin(self.player[6])

        t1a = getAngle(cpl, spl, -dx1, -dy1)
        t2a = getAngle(cpl, spl, -dx2, -dy2)

        if d1 <= d2:
            ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              d1, t1a,\
              d2, t2a]
        else:
            ob = [self.player[0],self.player[1],\
              self.player[2],self.player[3],\
              self.player[4],self.player[5],\
              self.player[6],self.player[7],\
              d2, t2a,\
              d1, t1a]

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
            self.surface.blit(im, (self.target_2[0]-w, self.target_2[1]-h))
            pg.display.flip()
            time.sleep(0.01)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _close(self):
        pass
