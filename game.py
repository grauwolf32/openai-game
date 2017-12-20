import pygame as pg
import numpy as np
import sys

from random import randint
from pygame.locals import *
from vectors import *
from game_utils import *

def main():
    pg.init()
    world_shape = (1152, 784)
    surface = pg.display.set_mode((world_shape[0], world_shape[1]), 16)
    pg.display.set_caption("OpenAI test game")

    controller_1 = StrategyController()
    controller_2 = StrategyController()

    p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
    p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

    player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_1)
    player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=20.0, max_ds=6.0, max_dw=2, friction_k=0.013, controller=controller_2)

    player_1.controller.bindStrategy(UserInputStrategy())
    player_2.controller.bindStrategy(TargerFollowStrategy())

    target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)
    target_2 = Target(id=4, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

    resources = dict()
    resources["image"] = dict()

    for player in [player_1, player_2]:
        resources["image"][player.id] = pg.image.load("Arrow.png").convert()

    for target in [target_1, target_2]:
        resources["image"][target.id] = pg.image.load("Circle.png").convert()

    resources["font"] = pg.font.SysFont("Times New Roman",12)
    clock = pg.time.Clock()

    world = World(players=[player_1, player_2], targets=[target_1, target_2], shape=world_shape)
    visualization = Visualization(world, resources, surface)

    n_iter = 0

    while True:
        keystate = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == QUIT: pg.quit();sys.exit()
        if keystate[K_ESCAPE]: pg.quit();sys.exit()

        world.updateState(dt=1.0/20.0)
        visualization.render()
        clock.tick()
        n_iter += 1

        info = resources["font"].render("iteration: {}".format(n_iter), True, (0,0,0))
        size = resources["font"].size("iteration: {}".format(n_iter))
        surface.blit(info, (20,size[1] + 10))
        pg.display.flip()

if __name__=="__main__":
    main()