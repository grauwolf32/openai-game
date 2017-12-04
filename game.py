import pygame as pg
import numpy as np
import sys

from random import randint
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
    
    def norm(self):
        abs_ = self.abs()

        if abs > 1e-6:
            return Vec(self.x / abs_, self.y / abs_)

        else:
            return Vec(0.0,0.0)            
            
    
        
def getAngle(v1,v2):
    scalar = v1.x*v2.x + v1.y*v2.y
    vector = v1.x*v2.y - v1.y*v2.x

    if np.fabs(v1.abs() * v2.abs()) < 10e-6:
        return 0.0
    
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
    def __init__(self, max_ds, max_dw, id, position, direction, k, radious):
        self.direction = direction
        self.position = position
        self.radious = radious
        self.id = id
        
        self.max_ds = max_ds
        self.max_dw = max_dw
        self.k = k
        
        self.speed = Vec(0.0,0.0)
        self.acceleration = Vec(0.0,0.0)
        self.dw = 0.0
        self.phi = 0.0
        
    
    def updateState(self, alpha, beta, dt):
        a = alpha*self.max_ds*self.direction - self.k*self.speed.abs()*self.speed
        self.direction.rot(self.dw*dt)
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

class Target(object):
    global world
    def __init__(self, location, radious):        
        self.location = location
        self.radious = radious

    def generate(self):
        self.location = Vec(randint(0.0,world.screen[0]),randint(0.0,world.screen[1]))
        
    def draw(self):
        c = pg.draw.circle(world.surface, (128,0,0), [self.location.x, self.location.y], 20)
        pg.display.update(c)

    def detect_collision(self):
        for unit in world.units:
            #print "Unit {} dist: {}".format(unit.id, (unit.position - target.location).abs() )
            if (unit.position - self.location).abs() <= self.radious + unit.radious:
                self.generate()
                self.draw()
                world.units_data[unit.id]["score"] += 1.0


class World(object):
    def __init__(self, units, units_data, target ,controllers, surface, screen):
        self.units = units
        self.controllers = controllers
        self.surface = surface
        self.screen = screen
        self.units_data = units_data
        self.target = target
    
    def tick(self, dt):
        for i in xrange(0,len(self.units)):
            alpha, beta = self.controllers[i].getParams()
            self.units[i].updateState(alpha, beta, dt)
        
            if self.units[i].position.x < 0.0: self.units[i].position.x = 0.0
            if self.units[i].position.y < 0.0: self.units[i].position.y = 0.0
            if self.units[i].position.x > self.screen[0]: self.units[i].position.x = self.screen[0]
            if self.units[i].position.y > self.screen[1]: self.units[i].position.y = self.screen[1]

        self.target.detect_collision()
            
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
            image = pg.transform.rotate(self.units_data[unit.id]["sprite"],-180.0*unit.phi/np.pi) 
            height = image.get_height()/2.0
            image.set_colorkey((0,128,0))
            self.surface.blit(image, (unit.position.x-height,unit.position.y-height))
            
#            dir_st = unit.position
#            dir_end = unit.position + 50*(unit.direction)
#            dir_ = pg.draw.line(self.surface, (123.0,128.0,0.0), [dir_st.x,dir_st.y], [dir_end.x,dir_end.y], 5)
            

        distance, angle, state = self.getState()
        distance = int(distance*10)/10.0
        angle = int(10*(180*angle/np.pi))/10.0
        
        info = font.render("Distance: {} Angle: {} State: {} {}".format(distance,angle, state[0], state[1]), True, (0,0,0))
        self.surface.blit(info, (self.screen[0]-240,10))

        stat = font.render("U1 Speed: {} U2 Speed: {}".format(int(100*self.units[0].speed.abs())/100.0,int(100*self.units[1].speed.abs())/100.0), True, (0,0,0))
        self.surface.blit(stat, (self.screen[0]-240,30))

        s = ""
        for unit_id in self.units_data:
            s += "User {} score {} |".format(unit_id, self.units_data[unit_id]["score"])

        stat = font.render(s, True, (0,0,0))
        self.surface.blit(stat, (self.screen[0]-240,50))

        self.target.draw()

        pg.display.flip()

class Strategy(object):
     pass

class RandomWalkStrategy(Strategy):
    global world
    def __init__(self, unit, x_bound, y_bound):
        self.unit = unit
        self.target = Vec(randint(x_bound[0],x_bound[1]), randint(y_bound[0],y_bound[1]))
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.alpha = 0.0
        self.beta  = 0.0
        self.beta_ = 0.0

        
    def implementStrategy(self, state, target):
        distance = state[0]
        angle = state[1]
        status = state[2]

        target_angle = getAngle(self.unit.direction, target.location - self.unit.position)
        speed_angle = getAngle(self.unit.speed, target.location - self.unit.position)

        target_distance = (target.location - self.unit.position).abs()

        if target_distance > target.radious + self.unit.radious:
            if target_angle * speed_angle >= 0.0:
                self.beta = min(1.0, 0.8 * (target_angle / self.unit.max_dw))
            else: 
                self.beta = 0.0

            self.alpha = 1.0*min(1.0, target_distance/self.unit.max_ds)

        else:
            self.alpha = 0.0
            self.beta  = 0.0
        


#        font = pg.font.SysFont("Times New Roman",12)
#        stat = font.render("TA: {} SA Speed: {}".format(int(100*target_angle)/100.0,int(100*speed_angle )/100.0), True, (0,0,0))
#        d = world.surface.blit(stat, (world.screen[0]-240,50))
#        pg.display.update(d)

#        print "Beta: {} TA: {} SA: {}".format(self.beta,target_angle,speed_angle) 

        return self.alpha, self.beta
        
world = None

def main():
    global world

    pg.init()
    screen = [1152, 864]
    surface = pg.display.set_mode((1152, 864), 16)
    pg.display.set_caption("Predator Prey Game")
    font = pg.font.SysFont("Times New Roman",12)

    sprite1 = pg.image.load("Arrow.png").convert()
    sprite2 = pg.image.load("Arrow.png").convert()

    units_data = dict()
    
    start_position = Vec(50,50)
    controller1 = Controller()
    unit1 = Unit(max_ds=6.0, max_dw=5.0, id=1, position=start_position, direction=Vec(0.0,-1.0).norm(),k=0.013,radious=30)
    
    start_position = Vec(screen[0]/2.0,screen[1]/2.0)
    controller2 = Controller()
    unit2 = Unit(max_ds=6.0, max_dw=4.0, id=2, position=start_position, direction=Vec(0.0,-1.0).norm(),k=0.013,radious=30)
    
    units_data[1] = dict()
    units_data[2] = dict()

    units_data[1]["sprite"] = sprite1 
    units_data[2]["sprite"] = sprite2 
    units_data[1]["score"] = 0
    units_data[2]["score"] = 0

    target = Target(Vec(0,0), 20)
    world = World([unit1,unit2],units_data,target, [controller1,controller2], surface, screen)
    target.generate()

    clock = pg.time.Clock()

    s = RandomWalkStrategy(unit2, (0,screen[0]),(0, screen[1]))
    
    while True:
        keystate = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == QUIT: pg.quit();sys.exit()
                    
        if keystate[K_ESCAPE]: pg.quit();sys.exit()
        if keystate[K_UP]: controller1.alpha = 1.0
        if keystate[K_DOWN]: controller1.alpha = -1.0
        if keystate[K_LEFT]: controller1.beta = -0.05
        if keystate[K_RIGHT]: controller1.beta = 0.05

        alpha, beta = s.implementStrategy(world.getState(),world.target)
        controller2.alpha = alpha
        controller2.beta = beta
                
        world.tick(1.0/60.0)
        world.draw(font)
        clock.tick()
       
if __name__ == "__main__":
    main()

