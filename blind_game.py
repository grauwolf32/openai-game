import numpy as np
import sys

from random import randint
from vectors import *
from game_utils import *

def main():
    world_shape = (480, 360)

    controller_1 = StrategyController()
    controller_2 = StrategyController()

    p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
    p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

    player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
    player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

    player_1.controller.bindStrategy(TargerFollowStrategy())
    player_2.controller.bindStrategy(TargerFollowStrategy())

    target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
    target_2 = Target(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

    world = World(players=[player_1, player_2], targets=[target_1, target_2], shape=world_shape)
    n_iter = 0

    t1 = time.time()
    while True:
        world.updateState(dt=1.0/10.0)
        n_iter += 1
        
        if n_iter % 250 == 0:
            state = world.getState()
            print "iteration: {}".format(n_iter)
            for player in world.players:
                player_id = "player_{}".format(player.id)
                print "{} : {}".format(player_id, state[player_id])
            print "-"*20
            print ""

        if n_iter >= 3000:
            break
    t2 = time.time()

    print "Processing speed: {} iter/sec".format(float(n_iter)/(t2-t1))
if __name__=="__main__":
    main()