import time
from random import *
from math import *


def getAngle(x1,y1,x2,y2):
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


world_shape = (640, 480)
# position, speed, acceleration, phi, dw
# phi - angle beween (0,1) and direction vector
max_dw = 2.0
max_ds = 6.0
friction_k = 0.013
dt = 1.0/10

c = 2.0*sqrt(max_dw)
rs = 20 + 20 # sum of radious

player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
target_1 = [0.0, 0.0]
target_2 = [0.0, 0,0]

player[0] = randint(0, world_shape[0]) 
player[1] = randint(0, world_shape[1]) 

player[6] = uniform(0.0, 2.0*pi)  
target_1[0] = randint(0, world_shape[0])
target_1[1] = randint(0, world_shape[1])

target_2[0] = randint(0, world_shape[0])
target_2[1] = randint(0, world_shape[1])

a = [0.0, 0.0]
score = 0
n = 20000

t1 = time.time()
for i in xrange(0,n):
    dd1x = player[0] - target_1[0]
    dd1y = player[1] - target_1[1]
    dd2x = player[0] - target_2[0]
    dd2y = player[1] - target_2[1]

    d1 = sqrt(dd1x*dd1x + dd1y*dd1y)
    d2 = sqrt(dd2x*dd2x + dd2y*dd2y)

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

    alpha = 1.0
    bval = target_angle / c
    beta = min(1.0, max(bval,-1.0))

    #print "alpha: {} beta: {}".format(alpha, beta)
    #print "d1: {} d2: {}".format(d1, d2)
    #print  player
    #print "target_1 ({},{})  target_2({},{})".format(target_1[0],target_1[1],target_2[0],target_2[1]) 
    #print "-"*20
    #print "\n"
    
    # Update player state based on alpha, beta
    k = alpha*max_ds
    speed_abs = (player[2]*player[2]+ player[3]*player[3])

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

t2 = time.time()
print "Elapsed time: {} score: {} iter/sec: {}".format(t2-t1, score, float(n)/(t2-t1))
    



