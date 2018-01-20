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


from envs.gathering_env import *
from envs.pursuit_env import *

def make_env(env_name, visualization=False):
    if env_name == "pursuit":
        return PursuitGameEnv(visualization=visualization)
    if env_name == "gathering":
        return GatheringGameEnv(visualization=visualization)
    return None

    




