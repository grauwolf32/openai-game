import argparse

from game_utils import *
from micro_env import *
from baselines.common import tf_util as U

visualization = True

def run_game(fname):
    with tf.Session(config=tf.ConfigProto()):
        GatheringAIRunner = AIRunner(ob_space=GatheringConstants.ob_space, ac_space = GatheringConstants.ac_space, fname=fname)
        if fname != None and tf.train.checkpoint_exists(fname):
            result = U.load_state(fname)

        env = GatheringGameEnv(visualization=visualization)
        ob = env._reset()

        while True:
            ac = GatheringAIRunner.step(ob)
            ob, rew, done, _ = env.step(ac)
            if visualization == True:
                env.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run gathering game with AI')
    parser.add_argument('--fname', type=str, default=None)
    args = parser.parse_args()
    run_game(fname=args.fname)