import time
import pygame as pg

from random import *
from math import *

def getAngle(x1,y1,x2,y2): # return angle between two vectors (x1,y1) and (x2, y2)
    s = x1*x2 + y1*y2
    v = x1*y2 - y1*x2

    l1 = sqrt(x1*x1 + y1*y1)
    l2 = sqrt(x2*x2 + y2*y2)
    pr = l1*l2

    if l1 < 10e-6 or l2 < 10e-6 or pr < 10e-5:
        return 0.0

    s /= pr
    alpha = acos(min(1.0, max(-1.0,s)))

    if v < 0.0:
        alpha *= -1.0
        
    return alpha 

visualization = True
world_shape = (480, 360)

# visualization init
if visualization:
    pg.init()
    surface = pg.display.set_mode((world_shape[0], world_shape[1]), 16)
    player_sprite = pg.image.load("Arrow.png").convert()
    target_sprite = pg.image.load("Circle.png").convert()
    font = pg.font.SysFont("Times New Roman",12)

max_dw = 2.0 # maximum of the angular velocity
max_ds = 6.0 # maximum of the velocity abs 

friction_k = 0.013
dt = 1.0/3.0

c  = sqrt(max_dw)
rs = 20 + 20 # sum of radious

player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0] # x, y, v_x, v_y, a_x, a_y, phi, omega
target_1 = [0.0, 0.0] #(x, y) - player position (v_x, v_y) - velocity (a_x, a_y) - acceleration
target_2 = [0.0, 0,0] # phi - player direction angle (from Ox), omega - angular velocity (d(phi)/ dt) 

player[0] = randint(0, world_shape[0]) 
player[1] = randint(0, world_shape[1]) 

player[6] = uniform(0.0, 2.0*pi)  
target_1[0] = randint(0, world_shape[0])
target_1[1] = randint(0, world_shape[1])

target_2[0] = randint(0, world_shape[0])
target_2[1] = randint(0, world_shape[1])

a = [0.0, 0.0]
score = 0

for _ in iter(range(0,1500*100)):
    dd1x = player[0] - target_1[0]
    dd1y = player[1] - target_1[1]
    dd2x = player[0] - target_2[0]
    dd2y = player[1] - target_2[1]

    d1 = sqrt(dd1x*dd1x + dd1y*dd1y) # distance to target_1
    d2 = sqrt(dd2x*dd2x + dd2y*dd2y) # distance to target_2

    #Check targets
    if d1 <= rs:
        score += 1.0
        target_1[0] = randint(0, world_shape[0])
        target_1[1] = randint(0, world_shape[1])
        continue
    
    if d2 <= rs:
        score += 1.0
        target_2[0] = randint(0, world_shape[0])
        target_2[1] = randint(0, world_shape[1])
        continue

    spl = sin(player[6])
    cpl = cos(player[6])

    # Dummy strategy
    if d1 <= d2:
        target_angle = getAngle(cpl,spl,-dd1x,-dd1y)
    else:
        target_angle = getAngle(cpl,spl,-dd2x,-dd2y)

    # physical stuff
    alpha = 1.0
    bval = target_angle / c
    beta = min(1.0, max(bval,-1.0))

    k = alpha*max_ds
    speed_abs = sqrt((player[2]*player[2] + player[3]*player[3]))

    a[0] = k*cpl - friction_k*speed_abs*player[2]
    a[1] = k*spl - friction_k*speed_abs*player[3]

    player[6] += player[7]*dt
    player[0] += player[2]*dt
    player[1] += player[3]*dt

    player[2] += player[4]*dt
    player[3] += player[5]*dt

    player[4] = a[0]
    player[5] = a[1]
    player[7] = beta * max_dw

    # World reaction
    if player[0] < 0.0: player[0] = 0.0
    if player[1] < 0.0: player[1] = 0.0
    if player[0] > world_shape[0]: player[0] = world_shape[0]
    if player[1] > world_shape[1]: player[1] = world_shape[1]

    # visualization block
    if visualization:
        surface.fill((255,255,255))
        phi = -(180.0/pi)*player[6] - 90.0
        im =  pg.transform.rotate(player_sprite, phi)
        h = im.get_height()/2.0
        w = im.get_width()/2.0
        im.set_colorkey((0,128,0))
        surface.blit(im, (player[0]-w, player[1]-h))

        im = target_sprite
        h = im.get_height()/2.0
        w = im.get_width()/2.0
        im.set_colorkey((0,128,0))

        surface.blit(im, (target_1[0]-w, target_1[1]-h))
        surface.blit(im, (target_2[0]-w, target_2[1]-h))
            
        text = "score : {}".format(score)
        info = font.render(text, True, (0,0,0))
        size = font.size(text)
        surface.blit(info, (world_shape[0]-size[0]-20, 10))

        text = "speed : {}".format(speed_abs)
        info = font.render(text, True, (0,0,0))
        surface.blit(info, (world_shape[0]-size[0] -20, (size[1]+3) + 10))

        pg.display.flip()
        time.sleep(0.01)




