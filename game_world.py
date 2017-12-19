import  numpy as np
import pygame as pg
import sys

from random import randint
from pygame.locals import *
from vectors import *

class Unit(object):
    def __init__(self, id, position, direction, radious):
        self.direction = direction
        self.position = position
        self.radious = radious
        self.id = id
        
class Target(Unit):
    def __init__(self, id, position, direction, radious):
        super(Target, self).__init__(id, position, direction, radious)
    
    def update(self, position):
        self.position = position


class MovableUnit(Unit):
    def __init__(self, id, position, direction, radious, max_ds, max_dw, friction_k):
        super(MovableUnit, self).__init__(id, position, direction, radious)
        self.max_ds = max_ds
        self.max_dw = max_dw
        self.friction_k = friction_k

        self.speed = Vec(0.0,0.0)
        self.acceleration = Vec(0.0,0.0)
        self.dw = 0.0

class ControlledUnit(MovableUnit):
    def __init__(self, id, position, direction, radious, max_ds, max_dw, friction_k, controller):
        super(ControlledUnit, self).__init__(id, position, direction, radious,max_ds, max_dw, friction_k)
        self.controller = controller
        self.controller.bindUnit(self)

class Controller(object):
    def __init__(self):
        pass
    
    def bindUnit(self, munit):
        self._munit = munit

    def updateState(self, alpha, beta, dt):
        a = alpha*self._munit.max_ds*self._munit.direction - self._munit.friction_k*self._munit.speed.abs()*self._munit.speed
        self._munit.direction.rot(self._munit.dw*dt)
        self._munit.position = self._munit.position + self._munit.speed*dt
        self._munit.speed = self._munit.speed + dt*self._munit.acceleration
        self._munit.acceleration = a
        self._munit.dw = beta*self._munit.max_dw

class Strategy(object):
    def __init__(self):
        pass
    def implement(self, state):
        pass

    def bindController(self, controller):
        self.controller = controller

class StrategyController(Controller):
    def __init__(self):
        super(StrategyController, self).__init__()
        self.strategy = None

    def bindStrategy(self, strategy):
        self.strategy = strategy
        self.strategy.bindController(self)

    def updateState(self, state):
        dt = state["time_delta"]
        if self.strategy != None:
            alpha, beta = self.strategy.implement(state)
            super(StrategyController, self).updateState(alpha=alpha, beta=beta, dt=dt)

class TargerFollowStrategy(Strategy):
    def __init__(self):
        super(TargerFollowStrategy, self).__init__()
        
    def implement(self, state):
        target = state["targets"][0]
        target_distance = (self.controller._munit.position - target).abs()
        target_angle    = getAngle(self.controller._munit.direction, target - self.controller._munit.position)

        alpha = 0.0
        beta  = 0.0

        if target_distance > 2.0*self.controller._munit.radious:
            beta = min(1.0, (target_angle / 2.0*np.sqrt(self.controller._munit.max_dw)))
            alpha = 1.0
        
        return alpha, beta

class UserInputStrategy(Strategy):
    def __init__(self):
        super(UserInputStrategy, self).__init__()

    def implement(self, state):
        alpha = 0.0
        beta = 0.0

        keystate = pg.key.get_pressed()
        if keystate[K_UP]: alpha += 1.0
        if keystate[K_DOWN]: alpha += -1.0
        if keystate[K_LEFT]: beta += -0.2
        if keystate[K_RIGHT]: beta += 0.2

        return alpha, beta

class AIControlledStrategy(Strategy):
    pass
        
class World(object):
    def __init__(self, players, targets, shape):
        self.state = dict()
        self.players = players
        self.shape = shape
        self.targets = targets

        for target in self.targets:
            pos_x = float(randint(0, self.shape[0]))
            pos_y = float(randint(0, self.shape[1]))
            position = Vec(pos_x, pos_y)
            target.update(position)


        for i in xrange(0, len(self.players)):
            self.state["player_{}".format(self.players[i].id)] = 0.0

        self.state["targets"] = [target.position for target in self.targets]
        self.state["players"] = [player.position for player in self.players]
        self.state["time_delta"] = 0.0

    def getState(self):
        return self.state
    
    def updateState(self, dt):
        for player in self.players:
            player.controller.updateState(self.state)

        for player in self.players:
            if player.position.x < 0.0: player.position.x = 0.0
            if player.position.y < 0.0: player.position.y = 0.0
            if player.position.x > self.shape[0]: player.position.x = self.shape[0]
            if player.position.y > self.shape[1]: player.position.y = self.shape[1]

        for target in self.targets:
            for player in self.players:
                if (player.position - target.position).abs() <= (player.radious + target.radious):
                    self.state["player_{}".format(player.id)] += 1.0
                    
                    pos_x = float(randint(0, self.shape[0]))
                    pos_y = float(randint(0, self.shape[1]))
                    
                    position = Vec(pos_x, pos_y)
                    target.update(position)

        self.state["targets"] = [target.position for target in self.targets]
        self.state["players"] = [player.position for player in self.players]
        self.state["time_delta"] = dt

class Visualization(object):
    def __init__(self, world, resources, surface):
        self.world = world
        self.resources = resources
        self.surface = surface

    def render(self, stat=True):
        self.surface.fill((255,255,255))
        units = self.world.players + self.world.targets 
        for unit in units:
            unit_image = self.resources["image"][unit.id]
            phi = 180.0*getAngle(unit.direction,Vec(1.0,0.0))/np.pi
            image = pg.transform.rotate(unit_image, phi) 
            height = image.get_height()/2.0
            width  = image.get_width()/2.0

            image.set_colorkey((0,128,0))
            self.surface.blit(image, (unit.position.x-width,unit.position.y-height))
          
        if stat == True:  
            player_id = ["player_{}".format(player.id) for player in self.world.players]
            
            font = self.resources["font"]
            for i in xrange(0,len(player_id)):
                info = font.render("{} : {}".format(player_id[i], self.world.state[player_id[i]]), True, (0,0,0))
                self.surface.blit(info, (self.world.shape[0]-240,i*12 + 10))

        pg.display.flip()

def main():
    pg.init()
    world_shape = (1152, 784)
    surface = pg.display.set_mode((world_shape[0], world_shape[1]), 16)
    pg.display.set_caption("OpenAI test game")

    controller_1 = StrategyController()
    controller_2 = StrategyController()

    p1_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))
    p2_pos = Vec(randint(0, world_shape[0]), randint(0, world_shape[1]))

    player_1 = ControlledUnit(id=1, position=p1_pos,direction=Vec(1.0,0.0), radious=40.0, max_ds=6.0, max_dw=5, friction_k=0.013, controller=controller_1)
    player_2 = ControlledUnit(id=2, position=p2_pos,direction=Vec(1.0,0.0), radious=40.0, max_ds=6.0, max_dw=5, friction_k=0.013, controller=controller_2)

    player_1.controller.bindStrategy(UserInputStrategy())
    player_2.controller.bindStrategy(TargerFollowStrategy())

    target_1 = Target(id=3, position=Vec(0.0,0.0), direction=Vec(1.0,0.0), radious=20)

    resources = dict()
    resources["image"] = dict()

    for player in [player_1, player_2]:
        resources["image"][player.id] = pg.image.load("Arrow.png").convert()

    for target in [target_1]:
        resources["image"][target.id] = pg.image.load("Circle.png").convert()

    resources["font"] = pg.font.SysFont("Times New Roman",12)
    clock = pg.time.Clock()

    world = World(players=[player_1, player_2], targets=[target_1],shape=world_shape)
    visualization = Visualization(world, resources, surface)

    while True:
        keystate = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == QUIT: pg.quit();sys.exit()
        if keystate[K_ESCAPE]: pg.quit();sys.exit()

        world.updateState(dt=1.0/60.0)
        visualization.render()
        clock.tick()

if __name__=="__main__":
    main()
