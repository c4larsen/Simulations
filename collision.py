# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 08:46:25 2019

@author: clars
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dt = 1/60

def collision1d(body1, body2):
    m1 = body1.mass
    v1 = body1.vel
    m2 = body2.mass
    v2 = body2.vel
    v1_final = (1.0/(m1 + m2)) * (m2*(v2 - v1) + m1*v1 + m2*v2)
    v2_final = (1.0/(m1 + m2)) * (-m1 * (v2 - v1) + m1*v1 + m2*v2)
    
    return v1_final, v2_final

def collision2d(body1, body2):
    ds = (body1.pos - body2.pos) - 
    
    
    

class body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        
body1 = body(1000, 0, 0)
body2 = body(1, 3, -1)

def init():
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    return ln,

fig, ax = plt.subplots()
ln, = ax.plot([], [], 'ro')
ln1, = ax.plot([], [], 'bo')

def update(frame):
    ds = 0
    deltat = 0
    
    if body2.pos <= body1.pos:
        ds = np.abs(body2.pos - body1.pos)
        dv = np.abs(body2.vel - body1.vel)
        deltat = -ds/dv
        body1.pos += body1.vel*deltat
        body2.pos += body2.vel*deltat
        
        body1.vel, body2.vel = collision1d(body1, body2)
        
        body1.pos += body1.vel*(dt - deltat)
        body2.pos += body2.vel*(dt - deltat)
    else:
        body1.pos += body1.vel*dt
        body2.pos += body2.vel*dt
        
    ln.set_data(body1.pos, 0)
    ln1.set_data(body2.pos, 0)
    print(body1.vel, body2.vel)
        
    
    return ln, ln1,

ani = FuncAnimation(fig, update, frames=1000, interval=1000*dt, init_func=init,
                    blit=True)

plt.show()
    

    
