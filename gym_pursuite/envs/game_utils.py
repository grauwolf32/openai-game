import tensorflow as tf
import numpy as np

from baselines.common import tf_util as U
from gym import error, spaces, utils
from math import *

def getAngle(x1,y1,x2,y2):
    s = x1*x2 + y1*y2
    v = x1*y2 - y1*x2

    l1 = np.sqrt(x1*x1 + y1*y1)
    l2 = np.sqrt(x2*x2 + y2*y2)
    pr = l1*l2

    if l1 < 10e-4 or l2 < 10e-4 or pr < 10e-3:
        return 0.0

    s = s / pr
    if s < -1.0:
        s = -1.0
    if s > 1.0:
        s = 1.0

    alpha = np.arccos(s)

    if v < 0.0:
        alpha *= -1.0
        
    return alpha 
    




