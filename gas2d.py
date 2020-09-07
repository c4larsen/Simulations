# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 08:46:25 2019

@author: clars
"""
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 30                                 # Number of balls.
BALL_RADIUS = 0.1                      # Radius of balls.
RWALL = 10                             # Position of rightmost wall.
TWALL = 6                              # Position of topmost wall.
SPEEDF = 5                             # Scaling factor for initial velocities.
g = 0 * (9.8 / 10) * np.array([0, -1]) # Acceleration due to gravity.

FF = 1                # Fast forward factor
FPS = 80              # Frames per second of animation.
FRAME_DELTA = 1/(FPS) # Time between frames.

# Matrix for rotating 2d vector by 90 degrees.
rot90 = np.zeros((2,2))
rot90[0, 1] = -1
rot90[1, 0] = 1

# Reflect about x-axis.
x_refl = np.zeros((2,2))
x_refl[0, 0] = -1
x_refl[1, 1] = 1

# Reflect about y-axis
y_refl = -1.0 * x_refl

def collision1d(m1, v1, m2, v2):
    """ Returns final velocities in a 1d elastic collision.
    
    """
    v1_final = (1.0/(m1 + m2)) * (m2*(v2 - v1) + m1*v1 + m2*v2)
    v2_final = (1.0/(m1 + m2)) * (-m1 * (v2 - v1) + m1*v1 + m2*v2)
    return v1_final, v2_final

def ball_collision(b1, b2, dt):
    """ Corrects positions and velocities of overlapping balls for collisions.
    
    """
    m1 = b1.mass
    m2 = b2.mass
    r1 = b1.radius
    r2 = b2.radius
    x1 = b1.prevpos
    x2 = b2.prevpos
    v1 = b1.vel
    v2 = b2.vel
    
    s = x1 - x2
    sdot = v1 - v2
    
    # Set up quadratic eqn to find time to collision from previous frame when 
    # balls were not overlapping.
    a = np.dot(sdot, sdot)
    b = 2 * np.dot(s, sdot)
    c = np.dot(s, s) - (r1 + r2)**2
    
    delta1 = (- b - np.sqrt(b**2 - 4*a*c))/(2*a)
    delta2 = (- b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
    # Collision happens at first instance of balls touching, so take minimum
    # of solutions.
    deltat = np.minimum(delta1, delta2)
    
    # Positions at collision.
    x1c = x1 + v1*deltat
    x2c = x2 + v2*deltat
    
    sc = x1c - x2c
    
    # Normal direction unit vector at collision.
    norm = (1/(np.linalg.norm(sc))) * sc
    
    # Normal component of velocities.
    v1n = np.dot(v1, norm)
    v2n = np.dot(v2, norm)
    
    # Calculate new normal velocities. Since velocities in tangential direction
    # remain unchanged, 2d collision is reduced to a 1d collision in the 
    # normal components of velocities.
    v1nf, v2nf = collision1d(m1, v1n, m2, v2n)
    
    # Outgoing velocities.
    v1f =  v1 -v1n*norm + v1nf*norm
    v2f  = v2 -v2n*norm + v2nf*norm
    
    # Update positions and velocities for current frame.
    b1.pos[:] = x1 + (deltat)*v1 + (dt - deltat) * v1f
    b2.pos[:] = x2 + (deltat)*v2 + (dt - deltat) * v2f
    b1.vel[:] = v1f
    b2.vel[:] = v2f

def left_wall_collision(ball, wall_coord, dt):
    """ Corrects position and velocity of ball after detection of collision 
    with leftmost wall.
    
    """
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # Time since collision from previous frame.
    delta_x_coord = wall_coord + r - x[0]
    deltat = delta_x_coord / v[0] 
    
    # Reflect velocity about the x-axis.
    vf = v[0] * x_refl[:, 0] + v[1] * x_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf

def right_wall_collision(ball, wall_coord, dt):
    """ Corrects position and velocity of ball after detection of collision 
    with rightmost wall.
    
    """
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # Time since collision from prev frame.
    delta_x_coord = wall_coord - r - x[0]
    deltat = delta_x_coord / v[0] 
    
    # Reflect velocity about x-axis.
    vf = v[0] * x_refl[:, 0] + v[1] * x_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf

def top_wall_collision(ball, wall_coord, dt):
    """ Corrects position and velocity of ball after detection of collision 
    with topmost wall.
    
    """
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # Time since collision.
    delta_y_coord = wall_coord - r - x[1]
    deltat = delta_y_coord / v[1] 
    
    # Reflect velocity about y-axis.
    vf = v[0] * y_refl[:, 0] + v[1] * y_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf
    
def bot_wall_collision(ball, wall_coord, dt):
    """ Corrects position and velocity of ball after detection of collision 
    with lowest wall.
    
    """
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # Time since collision.
    delta_y_coord = wall_coord + r - x[1]
    deltat = delta_y_coord / v[1] 
    
    # Reflect velocity about y-axis.
    vf = v[0] * y_refl[:, 0] + v[1] * y_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf
    

class body:
    def __init__(self, mass, pos, vel, radius):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.radius = radius
    
    def physics(self, delta):
        self.prevpos = self.pos.copy()
        self.vel += g*delta
        self.pos += self.vel * delta
    
    def drawball(self, ax):
        # self.image = plt.Circle(self.pos, BALL_RADIUS, color='b', fill=False)
        self.image = plt.Circle(self.pos, self.radius, color='#555555', fill=False)
        ax.add_artist(self.image)
        return self.image

# Generate balls.
balls = []
for i in range(N):
    balls.append(body(0.1, np.array([random.uniform(BALL_RADIUS, RWALL - BALL_RADIUS),
                                     random.uniform(BALL_RADIUS, TWALL - BALL_RADIUS)]), 
                                    SPEEDF * np.array([random.uniform(-1, 1),
                                                       random.uniform(-1, 1)]),
                                    BALL_RADIUS))      
fig, ax = plt.subplots()

# Draw walls.
ax.axvline(x = 0)
ax.axvline(x = RWALL)
ax.axhline(y = 0)
ax.axhline(y = TWALL)

# Time elapsed in simulation.
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
time_elapsed = 0

images = [] # Stores artists to be updated during animation

def init():
    global prevtime
    ax.set_ylim(-0.5, TWALL + 0.5)
    ax.set_xlim(-0.5, RWALL + 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    time_text.set_text('')
    
    for ball in balls:
        ball.drawball(ax)
        images.append(ball.image)
    
    images.append(time_text)
    prevtime = time.time()
    return images

def update(frame):
    global prevtime
    global time_elapsed
    
    currtime = time.time()
    dt = FF * (currtime - prevtime)
    prevtime = time.time()
    time_elapsed += dt
    time_text.set_text('Sim time: {:.1f}s'.format(time_elapsed))
    
    # Update state of system ignoring collisions.
    for ball in balls:
        ball.physics(dt)
    
    # Account for collisions.
    for i in range(len(balls)):
        b = balls[i]
        if (b.pos[0] < b.radius):
            left_wall_collision(b, 0, dt)
            continue
        elif (b.pos[0] > RWALL - b.radius):
            right_wall_collision(b, RWALL, dt)
            continue
        elif (b.pos[1] > TWALL - b.radius):
            top_wall_collision(b, TWALL, dt)
            continue
        elif (b.pos[1] < b.radius):
            bot_wall_collision(b, 0, dt)
            continue
        
        for j in range(i + 1, len(balls)):
            b2 = balls[j]
            
            if (np.dot(b.pos - b2.pos, b.pos - b2.pos) < (b.radius + b2.radius)**2):
                ball_collision(b, b2, dt)
    return images

ani = FuncAnimation(fig, update, frames=1000, interval=1000 * FRAME_DELTA , init_func=init,
                    blit=True)

plt.show()
