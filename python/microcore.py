import numpy as np
import time
from random import *


def getAngle(x1,y1,x2,y2):
    s = x1*x2 + y1*y2
    v = x1*y2 - y1*x2

    l1 = np.sqrt(x1*x1 + y1*y1)
    l2 = np.sqrt(x2*x2 + y2*y2)
    pr = l1*l2

    if l1 < 10e-6 or l2 < 10e-6 or pr < 10e-5:
        return 0.0

    s /= pr

    alpha = np.arccos(s)

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

c = 2.0*np.sqrt(max_dw)
rs = 20 + 20 # sum of radious

player = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
target_1 = [0.0, 0.0]
target_2 = [0.0, 0,0]

player[0] = randint(0, world_shape[0]) 
player[1] = randint(0, world_shape[1]) 

player[6] = uniform(0.0, 2.0*np.pi)  
target_1[0] = randint(0, world_shape[0])
target_1[1] = randint(0, world_shape[1])

target_2[0] = randint(0, world_shape[0])
target_2[1] = randint(0, world_shape[1])

a = [0.0, 0.0]
score = 0
n = 40000

t1 = time.time()
for i in xrange(0,n):
    dd1x = player[0] - target_1[0]
    dd1y = player[1] - target_1[1]
    dd2x = player[0] - target_2[0]
    dd2y = player[1] - target_2[1]

    d1 = np.sqrt((dd1x)**2 + (dd1y)**2)
    d2 = np.sqrt((dd2x)**2 + (dd2y)**2)

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

    alpha = 0.0
    beta = 0.0

    # Dummy strategy
    if d1 <= d2:
        target_angle = getAngle(player[0],player[1],-dd1x,-dd1y)
        if d1 > rs:
            beta = min(1.0, (target_angle / c))
            alpha = 1.0
    else:
        target_angle = getAngle(player[0],player[1],-dd2x,-dd2y)
        if d2 > rs:
            beta = min(1.0, (target_angle / c))
            alpha = 1.0
    
    # Update player state based on alpha, beta
    k = alpha*max_ds
    speed_abs = (player[2]**2 + player[3]**2)

    a[0] = k*np.cos(player[6]) - friction_k*speed_abs*player[2]
    a[1] = k*np.sin(player[6]) - friction_k*speed_abs*player[3]

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

    



