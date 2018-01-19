import tensorflow as tf
import numpy as np

from baselines.acktr.policies import GaussianMlpPolicy

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

# Some globals


class GatheringConstantsClass(object):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0

        world_shape = (480, 360)
        self.world_shape = world_shape

        self.params = (max_dw, max_ds, friction_k, dt)

        timestep_limit = 1500
        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        step_penalty_reward = -2.0 / timestep_limit
        dw_penalty_reward   = -1.8 / timestep_limit
        high_speed_reward   = 1.0 / timestep_limit
        target_reward       = 3.0

        self.rewards = (target_reward, step_penalty_reward, dw_penalty_reward, high_speed_reward)
        max_speed = sqrt(max_ds / friction_k)
        max_dist  = sqrt(world_shape[0]*world_shape[0] + world_shape[1]*world_shape[1])

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


class DummyPursuitConstantsClass(object):
    def __init__(self):
        max_dw = 2.0
        max_ds = 6.0
        friction_k = 0.013
        dt = 1.0/2.0
        c = sqrt(max_dw)

        world_shape = (480, 360)
        self.world_shape = world_shape

        self.params = (max_dw, max_ds, friction_k, dt, c)

        timestep_limit = 1500
        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        step_reward = 2.0 / timestep_limit
        high_speed_reward   = 1.0 / timestep_limit
        target_reward       = -100.0
        distance_reward = 5.0 / timestep_limit

        self.rewards = (target_reward, step_reward, high_speed_reward, distance_reward)
        max_speed = sqrt(max_ds / friction_k)
        max_dist  = sqrt(world_shape[0]*world_shape[0] + world_shape[1]*world_shape[1])

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

DummyPursuitConstants = DummyPursuitConstantsClass()

class AIRunner(object):
    def __init__(self, ob_space, ac_space):
        self.ob_space = ob_space
        self.ac_space = ac_space
        
        self.ob = np.float32(np.zeros(self.ob_shape))
        self.prev_ob = np.float32(np.zeros(self.ob_shape))
        self.obfilter = ZFilter(env.observation_space.shape)

        self.sess = tf.Session(config=tf.ConfigProto())

        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.shape[0]

        with tf.variable_scope("pi"):
            self.policy = GaussianMlpPolicy(ob_dim, ac_dim)

    def step(self, ob):
        with tf.device('/cpu:0'):
            state = np.concatenate([ob, self.prev_ob], -1)
            ac, ac_dist, logp = self.policy.act(state)
            self.ob = self.obfilter(self.ob)
            self.prev_ob = np.copy(ob)

            scaled_ac = self.ac_space.low + (ac + 1.) * 0.5 * (self.ac_space.high - self.ac_space.low)
            scaled_ac = np.clip(scaled_ac, self.ac_space.low, self.ac_space.high)

            return scaled_ac

    




