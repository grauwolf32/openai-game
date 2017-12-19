import numpy as np

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