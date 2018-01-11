import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U
from gym import error, spaces, utils
from math import *

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

class AIRunner(object):
    def __init__(self, ob_space, ac_space ,fname):
        self.ob_shape = ob_space.ob_shape
        self.ac_space = ac_space
        self.ob = np.float32(np.zeros(ob_shape))
        self.prev_ob = np.float32(np.zeros(ob_shape))

    def step(self, ob):
        state = np.concatenate([ob, self.prev_ob], -1)
        ac, ac_dist, logp = policy.act(state)
        self.prev_ob = np.copy(ob)

        scaled_ac = self.ac_space.low + (ac + 1.) * 0.5 * (self.ac_space.high - self.ac_space.low)
        scaled_ac = np.clip(scaled_ac, self.ac_space.low, self.ac_space.high)
        return scaled_ac

# Some globals


class GatheringConstantsClass(object):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0

        world_shape = (120, 1000)
        self.world_shape = world_shape

        max_speed = sqrt(max_ds / friction_k)
        max_dist  = sqrt(self.world_shape[0]**2 + self.world_shape[1]**2)

        self.params = (max_dw, max_ds, friction_k, dt)
        self.spec = EnvSpec(timestep_limit = 1500, id=1)

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

        ob_low = np.array(ob_low)
        ob_high = np.array(ob_high)
        
        self.ac_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.ob_space = spaces.Box(low=ob_low, high=ob_high)

GatheringConstants = GatheringConstantsClass()
    




