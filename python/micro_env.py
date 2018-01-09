import os
import gym

import pygame as pg
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from copy import deepcopy

from game_utils import *
from math import *


class GameEnv(gym.Env):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/10

        c = 2.0*sqrt(max_dw)
        rs = 20.0 + 20.0 

        self.world_shape = (640, 480)
        self.params = (max_dw, max_ds,friction_k, dt, c, rs)

        max_speed = np.sqrt(self.max_ds / self.friction_k)

        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))

        ob_low  = [0.0, 0.0, -max_speed,-max_speed, -max_ds,-max_ds, ,-max_dw, ]
        ob_high = [self.world.shape[0], self.world.shape[1], 1,1,max_speed, max_speed]*n_players + [self.world.shape[0], self.world.shape[1]]*n_targets

        self.observation_space = spaces.Box(low=np.array(ob_low), high=np.array(ob_high))

        self._spec = EnvSpec(timestep_limit = 2000, id=1)
        self.total_reward = 0.0

    def _step(self, action): 
        alpha = action[0]
        beta = action[1]
        
        k = alpha*max_ds
        speed_abs = (player[2]*player[2]+ player[3]*player[3])

        a[0] = k*cpl - friction_k*speed_abs*player[2]
        a[1] = k*spl - friction_k*speed_abs*player[3]

        player[6] += player[7]*dt
        player[0] += player[2]*dt
        player[1] += player[3]*dt

        player[2] += player[4]*dt
        player[3] += player[5]*dt

        player[4] = a[0]
        player[5] = a[1]
        player[7] = beta * max_dw

    # World reaction
    if player[0] < 0.0: player[0] = 0.0
    if player[1] < 0.0: player[1] = 0.0
    if player[0] > world_shape[0]: player[0] = world_shape[0]
    if player[1] > world_shape[1]: player[1] = world_shape[1]

        
        return observation, reward, done, info 

        
    def _reset(self): 
        self.player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
        self.target_1 = [0.0, 0.0]
        self.target_2 = [0.0, 0,0]

        self.player[0] = randint(0, world_shape[0]) 
        self.player[1] = randint(0, world_shape[1]) 
        self.player[6] = uniform(0.0, 2.0*pi)  

        self.target_1[0] = randint(0, world_shape[0])
        self.target_1[1] = randint(0, world_shape[1])
        self.target_2[0] = randint(0, world_shape[0])
        self.target_2[1] = randint(0, world_shape[1])

        self.score = 0
        ob = self.player + self.target_1 + self.target_2

        return ob

    def _render(self, mode='human', close=False):
        if mode == 'human':
            pass

    def _seed(self, seed=None): 
        return []

    def _close(self):
        pass
