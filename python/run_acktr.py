import argparse
import logging
import os
import tensorflow as tf
import gym

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common import tf_util as U
from baselines import bench

from acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction

from micro_env import *

visualization = True

def train(num_timesteps, seed, fname):
    env=GameEnv(visualization=visualization)
    env = bench.Monitor(env, logger.get_dir(),  allow_early_resets=True)
    set_global_seeds(seed)
    env.seed(seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        try:
            learn(env, policy=policy, vf=vf,
                gamma=0.99, lam=0.97, timesteps_per_batch=4500,
                desired_kl=0.002,
                num_timesteps=num_timesteps, animate=visualization, fname=fname)
        except KeyboardInterrupt:
            if fname != None:
                U.save_state(fname)
                logger.log("Model saved to file {}".format(fname))

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--fname', type=str, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e9))
    args = parser.parse_args()
    logger.configure()
    train(num_timesteps=args.num_timesteps, seed=args.seed, fname=args.fname)
