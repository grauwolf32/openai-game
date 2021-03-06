import tensorflow as tf
import numpy as np

from gym import error, spaces, utils
from math import *

class EnvSpec(object):
    def __init__(self, timestep_limit, id):
        self.timestep_limit = timestep_limit
        self.id = id

class GatheringConstantsClass(object):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0

        world_shape = (480, 360)
        self.world_shape = world_shape

        timestep_limit = 1500
        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        step_penalty_reward = -2.0 / timestep_limit
        dw_penalty_reward   = -1.8 / timestep_limit
        high_speed_reward   = 1.0 / timestep_limit
        target_reward       = 3.0

        self.rewards = (target_reward, step_penalty_reward, dw_penalty_reward, high_speed_reward)
        max_speed = np.sqrt(max_ds / friction_k)
        max_dist  = np.sqrt(world_shape[0]*world_shape[0] + world_shape[1]*world_shape[1])
        self.params = (max_dw, max_ds, friction_k, dt, max_speed)

        ob_low  = [0.0,  0.0,\
            -max_speed,-max_speed,\
            -max_ds, -max_ds,\
            -1.1*pi , -max_dw,\
            -max_dist, -1.1*pi,\
            -max_dist, -1.1*pi ]

        ob_high = [world_shape[0], world_shape[1],\
            max_speed, max_speed, max_ds, max_ds,\
            1.1*pi, max_dw, \
            max_dist, 1.1*pi, \
            max_dist, 1.1*pi]

        ob_low = np.asarray(ob_low, dtype=np.float64)
        ob_high = np.asarray(ob_high, dtype=np.float64)
        
        self.ac_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.ob_space = spaces.Box(low=ob_low, high=ob_high)

GatheringConstants = GatheringConstantsClass()


class PursuitConstantsClass(object):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0
        c = np.sqrt(max_dw)

        world_shape = (480, 360)
        self.world_shape = world_shape

        timestep_limit = 1500
        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        step_reward = 2.0 / timestep_limit
        high_speed_reward   = 1.0 / timestep_limit
        target_reward       = -100.0
        distance_reward = 5.0 / timestep_limit

        self.rewards = (target_reward, step_reward, high_speed_reward, distance_reward)
        max_speed = np.sqrt(max_ds / friction_k)
        max_dist  = np.sqrt(world_shape[0]*world_shape[0] + world_shape[1]*world_shape[1])
        self.params = (max_dw, max_ds, friction_k, dt, c, max_speed)

        ob_low  = [0.0,  0.0,\
            -max_speed,-max_speed,\
            -max_ds, -max_ds,\
            -1.1*pi , -max_dw,
             0.0,  0.0,\
            -max_speed,-max_speed,\
            -1.1*pi, 0.0]

        ob_high = [world_shape[0], world_shape[1],\
            max_speed, max_speed,\
            max_ds, max_ds,\
            1.1*pi, max_dw,\
            world_shape[0], world_shape[1],\
            max_speed, max_speed,\
            1.1*pi, max_dist]

        ob_low = np.array(ob_low)
        ob_high = np.array(ob_high)
        
        self.ac_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.ob_space = spaces.Box(low=ob_low, high=ob_high)

PursuitConstants = PursuitConstantsClass()
