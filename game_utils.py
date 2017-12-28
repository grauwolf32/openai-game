import  numpy as np
import pygame as pg
import time
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
        unit = self.controller._munit
        distances = list(enumerate([(unit.position - target).abs() for target in state["targets"]]))
        nearest_target = min(distances, key=lambda x:x[1]) 
        target = state["targets"][nearest_target[0]]
        target_angle = getAngle(unit.direction, target - unit.position)

        alpha = 0.0
        beta  = 0.0

        if nearest_target[1] > 2.0*unit.radious:
            beta = min(1.0, (target_angle / 2.0*np.sqrt(unit.max_dw)))
            alpha = 1.0
        
        return alpha, beta

class UserInputStrategy(Strategy):
    def __init__(self):
        super(UserInputStrategy, self).__init__()

    def implement(self, state):
        self.alpha = 0.0
        self.beta = 0.0

        keystate = pg.key.get_pressed()
        if keystate[K_UP]: self.alpha += 1.0
        if keystate[K_DOWN]: self.alpha += -1.0
        if keystate[K_LEFT]: self.beta += -0.3
        if keystate[K_RIGHT]: self.beta += 0.3

        return self.alpha, self.beta

class AIControlledStrategy(Strategy):
    def __init__(self):
        super(AIControlledStrategy, self).__init__()
        self.alpha = 0.0
        self.beta  = 0.0

    def setControl(self, alpha, beta):
        #print "Control set to {} {}".format(alpha, beta)
        self.alpha = alpha
        self.beta = beta

    def implement(self, state):
        alpha = self.alpha
        beta  = self.beta

        self.alpha = 0.0
        self.beta  = 0.0

        return alpha, beta
        
class World(object):
    def __init__(self, players, targets, shape):
        self.shape = shape
        self.players = players
        self.targets = targets
        self.resetState()
        
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

    def resetState(self):
        self.state = dict()

        for player in self.players:
            pos_x = float(randint(0, self.shape[0]))
            pos_y = float(randint(0, self.shape[1]))
            position = Vec(pos_x, pos_y)
            player.position = position

        for target in self.targets:
            pos_x = float(randint(0, self.shape[0]))
            pos_y = float(randint(0, self.shape[1]))
            position = Vec(pos_x, pos_y)
            target.update(position)


        for i in range(0, len(self.players)):
            self.state["player_{}".format(self.players[i].id)] = 0.0

        self.state["targets"] = [target.position for target in self.targets]
        self.state["players"] = [player.position for player in self.players]
        self.state["time_delta"] = 0.0

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
            phi = 180.0*getAngle(unit.direction,Vec(0.0,-1.0))/np.pi
            image = pg.transform.rotate(unit_image, phi) 
            height = image.get_height()/2.0
            width  = image.get_width()/2.0

            image.set_colorkey((0,128,0))
            self.surface.blit(image, (unit.position.x-width,unit.position.y-height))
          
        if stat == True:  
            player_id = ["player_{}".format(player.id) for player in self.world.players]
            
            font = self.resources["font"]
            for i in iter(range(0,len(player_id))):
                text = "{} : {}".format(player_id[i], self.world.state[player_id[i]])
                info = font.render(text, True, (0,0,0))
                size = font.size(text)
                self.surface.blit(info, (self.world.shape[0]-size[0]-20,i*(size[1]+3) + 10))

        pg.display.flip()

def get_observables(world, actor_id):
    # I choose these observables:
    # position, direction, speed of the actor
    # position, direction, speed of other players, sorted in ascented order of distance to the actor
    # position of the targets sorted in ascented order of distance to the actor

    observables = list()
    actor = None

    for player in world.players:
        if player.id == actor_id:
            actor = player
            break
    if actor == None:
        raise Exception("No such player with the actor id in this world")

    sorted_players = sorted(world.players, key=lambda x: (x.position - actor.position).abs())
    sorted_targets = sorted(world.targets, key=lambda x: (x.position - actor.position).abs())

    observables.append(actor.position.x)
    observables.append(actor.position.y)
    observables.append(actor.direction.x)
    observables.append(actor.direction.y)
    observables.append(actor.speed.x)
    observables.append(actor.speed.y)


    for player in sorted_players:
        if player.id != actor.id:
            observables.append(player.position.x)
            observables.append(player.position.y)
            observables.append(player.direction.x)
            observables.append(player.direction.y)
            observables.append(player.speed.x)
            observables.append(player.speed.y)
            
    for target in sorted_targets:
        observables.append(target.position.x)
        observables.append(target.position.y)

    return np.array(observables)

class EnvSpec(object):
    def __init__(self, timestep_limit,id):
        self.timestep_limit = timestep_limit
        self.id = id
