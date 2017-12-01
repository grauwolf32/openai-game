import pygame as pg
import numpy as np
import sys

from pygame.locals import *

class Vec(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __add__(self, a):
        return Vec(self.x + a.x, self.y + a.y)
    
    def __sub__(self, a):
        return Vec(self.x - a.x, self.y - a.y)
    
    def __mul__(self, c):
        return Vec(self.x * c, self.y * c)
    
    __rmul__ = __mul__
    
    def tuple(self):
        return (self.x, self.y)
    
    def rot(self, alpha):
        x = self.x
        y = self.y
        
        self.x = x*np.cos(alpha) - y*np.sin(alpha)
        self.y = x*np.sin(alpha) + y*np.cos(alpha)
    
    def abs(self):
        return np.sqrt(self.x**2 + self.y**2)
    
        
def getAngle(v1,v2):
    scalar = v1.x*v2.x + v1.y*v2.y
    vector = v1.x*v2.y - v1.y*v2.x
    
    scalar /= (v1.abs() * v2.abs())
    vector /= (v1.abs() * v2.abs())
    alpha = np.arccos(scalar)
    beta  = np.arcsin(vector)
    
    if beta < 0.0:
        alpha *= -1.0
        
    return alpha 

def scalar(v1, v2):
    return v1.x*v2.x + v1.y*v2.y

        
class Unit(object):
    def __init__(self, max_ds, max_dw, sprite, position, direction,k):
        self.direction = direction
        self.position = position
        
        self.max_ds = max_ds
        self.max_dw = max_dw
        self.k = k
        
        self.sprite = sprite
        
        self.speed = Vec(0.0,0.0)
        self.acceleration = Vec(0.0,0.0)
        self.dw = 0.0
        self.phi = 0.0
        
    
    def updateState(self, alpha, beta, dt):
        a = alpha*self.max_ds*self.direction - self.k*self.speed.abs()*self.speed
        self.direction.rot(-self.dw*dt)
        self.position = self.position + self.speed*dt
        self.speed = self.speed + dt*self.acceleration
        self.acceleration = a
        self.dw = beta*self.max_dw
        self.phi += self.dw*dt
        
class Controller(object):
    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        
    def setParams(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta 
        
    def getParams(self):
        alpha = self.alpha
        beta = self.beta
        self.flush()
        return alpha, beta
    
    def flush(self):
        self.alpha = 0.0
        self.beta = 0.0        
    

class World(object):
    def __init__(self, units, controllers, surface, screen):
        self.units = units
        self.controllers = controllers
        self.surface = surface
        self.screen = screen
    
    def tick(self, dt):
        for i in xrange(0,len(self.units)):
            alpha, beta = self.controllers[i].getParams()
            self.units[i].updateState(alpha, beta, dt)
        
            if self.units[i].position.x < 0.0: self.units[i].position.x = 0.0
            if self.units[i].position.y < 0.0: self.units[i].position.y = 0.0
            if self.units[i].position.x > self.screen[0]: self.units[i].position.x = self.screen[0]
            if self.units[i].position.y > self.screen[1]: self.units[i].position.y = self.screen[1]
        
            
    def getState(self):
        unit1 = self.units[0]
        unit2 = self.units[1]
        
        distance = (unit1.position - unit2.position).abs()
        angle = getAngle(unit1.direction, unit2.direction)

        state1 = np.sign(scalar((unit2.position-unit1.position),unit1.direction))
        state2 = np.sign(scalar((unit1.position-unit2.position),unit2.direction))

        state = (state1, state2)
        
        return distance, angle, state
    
    def draw(self, font):
        self.surface.fill((255,255,255))
        for unit in self.units:
            image = pg.transform.rotate(unit.sprite,180.0*unit.phi/np.pi) 
            height = image.get_height()/2.0
            image.set_colorkey((0,128,0))
            self.surface.blit(image, (unit.position.x-height,unit.position.y-height))
                               
        distance, angle, state = self.getState()
        distance = int(distance*10)/10.0
        angle = int(10*(180*angle/np.pi))/10.0
        
        info = font.render("Distance: {} Angle: {} State: {} {}".format(distance,angle, state[0], state[1]), True, (0,0,0))
        self.surface.blit(info, (self.screen[0]-240,10))

        stat = font.render("U1 Speed: {} U2 Speed: {}".format(int(100*self.units[0].speed.abs())/100.0,int(100*self.units[0].speed.abs())/100.0), True, (0,0,0))
        self.surface.blit(stat, (self.screen[0]-240,30))
        pg.display.flip()

class Strategy(object):
     pass

class SimpleStrategy(Strategy):
    def __init__(self, unit):
        self.alpha = 0.0
        self.beta = 0.0
        self.unit = unit

    def implementStrategy(self, state):
        distance = state[0]
        angle = state[1]
        status = state[2]
        
        self.alpha = -1.0
        self.beta = 0.02

        if distance < 250:
            if status[0] == 1.0 and status[1]==-1.0:
                self.beta = -0.4 * np.sign(angle)
                self.alpha = -1.0
                return self.alpha, self.beta

            if status[0] == -1.0 and status[1]==-1.0:
                self.beta = 0.4 * np.sign(angle)
                self.alpha = -0.5
                return self.alpha, self.beta
            
            if status[0] == 1.0 and status[1]==1.0:
                self.beta = -0.3*np.sign(angle)
                self.alpha = -1.0
                return self.alpha, self.beta

            if status[0] == -1.0 and status[1]==1.0:
                self.beta = 0.8 * (angle / self.unit.max_dw)
                self.alpha = -1.0
                return self.alpha, self.beta
            
            return self.alpha, self.beta

        return self.alpha, self.beta


def main():
    pg.init()
    screen = [1152, 864]
    surface = pg.display.set_mode((1152, 864), 16)
    pg.display.set_caption("Predator Prey Game")
    font = pg.font.SysFont("Times New Roman",12)

    sprite = pg.image.load("Arrow.png").convert()
    
    start_position = Vec(50,50)
    controller1 = Controller()
    unit1 = Unit(max_ds=6.0, max_dw=5.0, sprite=sprite, position=start_position, direction=Vec(0.0,1.0),k=0.013)
    
    start_position = Vec(screen[0]/2.0,screen[1]/2.0)
    controller2 = Controller()
    unit2 = Unit(max_ds=5.0, max_dw=4.0, sprite=sprite, position=start_position, direction=Vec(0.0,1.0),k=0.018)
    
    world = World([unit1,unit2],[controller1,controller2], surface, screen)
    clock = pg.time.Clock()

    s = SimpleStrategy(unit2)
    
    while True:
        keystate = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == QUIT: pg.quit();sys.exit()
                    
        if keystate[K_ESCAPE]: pg.quit();sys.exit()
        if keystate[K_UP]: controller1.alpha = -1.0
        if keystate[K_DOWN]: controller1.alpha = 1.0
        if keystate[K_LEFT]: controller1.beta = 0.05
        if keystate[K_RIGHT]: controller1.beta = -0.05

        alpha, beta = s.implementStrategy(world.getState())
        controller2.alpha = alpha
        controller2.beta = beta

        if keystate[K_w]: controller2.alpha = -1.0
        if keystate[K_s]: controller2.alpha = 1.0
        if keystate[K_a]: controller2.beta = 0.05
        if keystate[K_d]: controller2.beta = -0.05
                
        world.tick(1.0/60.0)
        world.draw(font)
        clock.tick()
       
if __name__ == "__main__":
    main()

