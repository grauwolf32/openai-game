import numpy as np
import os
import gym

import pygame as pg
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from copy import deepcopy

from game_utils import *


class GameEnv(gym.Env):
    def __init__(self):
        self.shape = (640, 480)
    
        self.max_dw = 2.0
        self.max_ds = 6.0
        self.friction_k = 0.013
        self.dt = 1.0/10

        self.c = 2.0*sqrt(max_dw)
        self.rs = 20.0 + 20.0 # sum of radious

        ## Init Visualization
        self.visualization = None

        if visualization == True:
            pg.init()
            surface = pg.display.set_mode((self.world.shape[0], self.world.shape[1]), 16)
            resources = dict()
            resources["image"] = dict()

            for player in world.players:
                resources["image"][player.id] = pg.image.load("Arrow.png").convert()

            for target in world.targets:
                resources["image"][target.id] = pg.image.load("Circle.png").convert()
 
            resources["font"] = pg.font.SysFont("Times New Roman",12)
            
            self.surface = surface
            self.visualization = Visualization(self.world, resources, self.surface)
            self.metadata["render.modes"].append("human")

        ## Other
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))

        max_speed = np.sqrt(self.actor.max_ds / self.actor.friction_k)
        n_players = len(self.world.players)
        n_targets = len(self.world.targets)

        ob_low  = [0, 0, -1,-1,-max_speed,-max_speed]*n_players + [0,0]*n_targets
        ob_high = [self.world.shape[0], self.world.shape[1], 1,1,max_speed, max_speed]*n_players + [self.world.shape[0], self.world.shape[1]]*n_targets

        self.observation_space = spaces.Box(low=np.array(ob_low), high=np.array(ob_high))

        self._spec = EnvSpec(timestep_limit = 2000, id=1)
        self.total_reward = 0.0

    def _step(self, action): 
        alpha = action[0]
        beta = action[1]

        #print("{} {}".format(alpha, beta))
        #old_state = deepcopy(self.world.state)

        self.actor.controller.strategy.setControl(alpha, beta)
        self.world.updateState(dt=1.0/4.0)
        
        state = self.world.getState()
        #actor_score = (state["player_{}".format(self.actor.id)] - old_state["player_{}".format(self.actor.id)])
        
        #opponent_score = 0.0
        #for player in self.world.players:
        #    if player.id == self.actor.id:
        #        continue
        #    opponent_score += (state["player_{}".format(player.id)] - old_state["player_{}".format(player.id)])

        #n_opponents = len(self.world.players) - 1.0

        #if n_opponents > 0.0:
        #    opponent_score /= n_opponents

        #reward = actor_score #- 0.5*opponent_score
        #print("Reward: {}".format(reward))
        #reward -= 0.001 # Step penalty
        
        distances = []

        for player in self.world.players:
            if player.id == self.actor.id:
                continue
            distances.append((player.position - self.actor.position).abs())

        reward = np.min(distances) / 1000.0   

        observation = get_observables(world=self.world, actor_id=self.actor.id)
        done = state["done"]
        if done:
            reward -= 100.0 # Catch penalty
        self.total_reward += reward
        
        if self.total_reward >= 400:#<= -2.0:
            done = True

        info = dict()
        return observation, reward, done, info 

        
    def _reset(self): 
        self.world.resetState()
        self.total_reward = 0.0
        observation = get_observables(world=self.world, actor_id=self.actor.id)
        return observation

    def _render(self, mode='human', close=False):
        if mode == 'human':
            if self.visualization != None:
                self.visualization.render()

    def _seed(self, seed=None): 
        return []

    def _close(self):
        #if self.visualization != None:
        #    pg.quit()
        pass
