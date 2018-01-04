import numpy as np
import sys

from random import randint
from vectors import *
from game_utils import *

def main():
    world_shape = (480, 360)

    controller_1 = StrategyController()
    #controller_2 = StrategyController()

    p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
    #p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

    player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
    #player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

    player_1.controller.bindStrategy(TargerFollowStrategy())
    #player_2.controller.bindStrategy(TargerFollowStrategy())

    target_1 = Unit(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
    target_2 = Unit(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

    world = World(players=[player_1], targets=[target_1, target_2], shape=world_shape)
    n_iter = 0
    
    score = []
    cntr = 0.0
    n = 200
    t1 = time.time()
    for i in xrange(0, n):
        while n_iter < 2000:
            world.updateState(dt=1.0/2.0)
            n_iter += 1
        
        cntr += n_iter
        n_iter = 0
        state = world.getState()
        player_id = "player_{}".format("1")
        score.append(float(state[player_id]))
        world.resetState()

    print "Mean score: {}\nMax score: {} \nMin score {}".format(np.mean(score),np.max(score),np.min(score))
    



    t2 = time.time()
    print 

    print "Processing speed: {} iter/sec".format(float(cntr)/(t2-t1))
if __name__=="__main__":
    main()