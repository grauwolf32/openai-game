import pygame as pg
import numpy as np
import sys

from random import randint
from pygame.locals import *
from vectors import *
from game_utils import *

def main():
    #t1 = time.time()
    pg.init()
    #t2 = time.time()
    #print "PyGame init time: {}".format(t2-t1) # ~0.04
    
    world_shape = (480, 360)
    surface = pg.display.set_mode((world_shape[0], world_shape[1]), 16)
    pg.display.set_caption("OpenAI test game")

    controller_1 = StrategyController()
    controller_2 = StrategyController()

    player_1 = ControlledUnit(id=1, position=Vec(0.0,0.0),direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
    player_2 = ControlledUnit(id=2, position=Vec(0.0,0.0),direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

    player_1.controller.bindStrategy(UserInputStrategy())
    player_2.controller.bindStrategy(TargerFollowStrategy())

    target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
    target_2 = Target(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

    world = World(players=[player_1, player_2], targets=[target_1, target_2], shape=world_shape)

    resources = dict()
    resources["image"] = dict()

    for player in world.players:
        resources["image"][player.id] = pg.image.load("Arrow.png").convert()

    for target in world.targets:
        resources["image"][target.id] = pg.image.load("Circle.png").convert()

    resources["font"] = pg.font.SysFont("Times New Roman",12)
    clock = pg.time.Clock()

    
    visualization = Visualization(world, resources, surface)

    #n_iter = 0

    while True:
        keystate = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == QUIT: pg.quit();sys.exit()
        if keystate[K_ESCAPE]: pg.quit();sys.exit()
        if keystate[K_r]: world.resetState();time.sleep(1.0)

        world.updateState(dt=1.0/10.0)
        visualization.render()
        time.sleep(0.01)
        clock.tick()
        #n_iter += 1

        #info = resources["font"].render("iteration: {}".format(n_iter), True, (0,0,0))
        #size = resources["font"].size("iteration: {}".format(n_iter))
        #surface.blit(info, (20,size[1] + 10))
        pg.display.flip()

        # print get_observables(world, 2)

if __name__=="__main__":
    main()