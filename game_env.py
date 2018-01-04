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
    def __init__(self, visualization=False):
        ## Init World
        world_shape = (480, 360)
        controller_1 = StrategyController()
        controller_2 = StrategyController()

        p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
        p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

        player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
        player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

        player_1.controller.bindStrategy(TargerFollowStrategy())
        player_2.controller.bindStrategy(AIControlledStrategy())

        target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
        target_2 = Target(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

        world = World(players=[player_1, player_2], targets=[target_1, target_2], shape=world_shape)
        self.world = world
        self.actor = player_2

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

        self._spec = EnvSpec(timestep_limit = 1000, id=1)
        self.total_reward = 0.0

    def _step(self, action): 
        alpha = action[0]
        beta = action[1]

        #print("{} {}".format(alpha, beta))
        old_state = deepcopy(self.world.state)

        self.actor.controller.strategy.setControl(alpha, beta)
        self.world.updateState(dt=0.75)
        
        state = self.world.getState()
        actor_score = (state["player_{}".format(self.actor.id)] - old_state["player_{}".format(self.actor.id)])
        
        opponent_score = 0.0
        for player in self.world.players:
            if player.id == self.actor.id:
                continue
            opponent_score += (state["player_{}".format(player.id)] - old_state["player_{}".format(player.id)])

        n_opponents = len(self.world.players) - 1.0

        if n_opponents > 0.0:
            opponent_score /= n_opponents

        reward = actor_score - 0.5*opponent_score
        reward -= 0.001 #Step penalty
        #print("Reward: {}".format(reward))

        observation = get_observables(world=self.world, actor_id=self.actor.id)
        done = False
        self.total_reward += reward
        
        if self.total_reward <= -10.0:
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
