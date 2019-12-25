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

dt = 1/60
g = (9.8 / 10) * np.array([0, -1])

rot90 = np.zeros((2,2))
rot90[0, 1] = -1
rot90[1, 0] = 1

x_refl = np.zeros((2,2))
x_refl[0, 0] = -1
x_refl[1, 1] = 1

y_refl = -1.0 * x_refl

rwall = 10
topwall = 6

SPEEDF = 5


def collision1d(m1, v1, m2, v2):
    v1_final = (1.0/(m1 + m2)) * (m2*(v2 - v1) + m1*v1 + m2*v2)
    v2_final = (1.0/(m1 + m2)) * (-m1 * (v2 - v1) + m1*v1 + m2*v2)
    return v1_final, v2_final

def ball_collision(b1, b2, dt):
    # get velocities of bodies
    print("entered")
    m1 = b1.mass
    m2 = b2.mass
    x1 = b1.prevpos
    x2 = b2.prevpos
    v1 = b1.vel
    v2 = b2.vel
    # calculate time since collision
    
    s = x1 - x2
    sdot = v1 - v2
    
    # set up quadratic eqn to find time since collision
    r1 = b1.radius
    r2 = b2.radius
    a = np.dot(sdot, sdot)
    b = 2 * np.dot(s, sdot)
    c = np.dot(s, s) - (r1 + r2)**2
    
    # solve quadratic
    
    delta1 = (- b - np.sqrt(b**2 - 4*a*c))/(2*a)
    delta2 = (- b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
    deltat = np.minimum(delta1, delta2)
    # positions at collision
    x1c = x1 + v1*deltat
    x2c = x2 + v2*deltat
    sc = x1c - x2c
    
    #tangential and normal direction unit vectors
    norm = (1/(np.linalg.norm(sc))) * sc
    # wrong: norm = tang[0]*rot90[:, 0] + tang[1]*rot90[:, 1]
    
    # normal component of velocities
    
    v1n = np.dot(v1, norm)
    v2n = np.dot(v2, norm)
    
    # calculate new normal velocities
    v1nf, v2nf = collision1d(m1, v1n, m2, v2n)
    
    # get outgoing velocities
    
    v1f =  v1 -v1n*norm + v1nf*norm
    v2f  = v2 -v2n*norm + v2nf*norm
    
    # update positions for current frame 
    b1.pos[:] = x1 + (deltat)*v1 + (dt - deltat) * v1f
    b1.vel = v1f
    b2.pos[:] = x2 + (deltat)*v2 + (dt - deltat) * v2f
    b2.vel = v2f
    print("done")
    
def left_wall_collision(ball, wall_coord):
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # time since collision
    delta_x_coord = wall_coord + r - x[0]
    deltat = delta_x_coord / v[0] 
    
    vf = v[0] * x_refl[:, 0] + v[1] * x_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf

def right_wall_collision(ball, wall_coord):
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # time since collision
    delta_x_coord = wall_coord - r - x[0]
    deltat = delta_x_coord / v[0] 
    
    vf = v[0] * x_refl[:, 0] + v[1] * x_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf

def top_wall_collision(ball, wall_coord):
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # time since collision
    delta_y_coord = wall_coord - r - x[1]
    deltat = delta_y_coord / v[1] 
    
    vf = v[0] * y_refl[:, 0] + v[1] * y_refl[:, 1]
    
    ball.vel[:] = vf
    ball.pos[:] = x + (deltat) * v + (dt - deltat) * vf
    
def bot_wall_collision(ball, wall_coord):
    x = ball.prevpos
    v = ball.vel
    r = ball.radius
    
    # time since collision
    delta_y_coord = wall_coord + r - x[1]
    deltat = delta_y_coord / v[1] 
    
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
    
    def updateball(self):
        # self.image.center = self.pos
        return self.image
N = 5
balls = []
BALL_RADIUS = 0.1
for i in range(N):
    balls.append(body(0.1, np.array([random.uniform(BALL_RADIUS, rwall - BALL_RADIUS), random.uniform(BALL_RADIUS, topwall - BALL_RADIUS)]), 
                      np.array([SPEEDF * random.uniform(-1, 1), SPEEDF * random.uniform(-1, 1)]), BALL_RADIUS))      

# =============================================================================
# body1 = body(1.0 , np.array([1.0, 0.5]), np.array([5.0, 0.0]), 0.2)
# body2 = body(1.0 , np.array([0.5, 1.0]), 3*np.array([-1, -0.5]), 0.2)
# =============================================================================

fig, ax = plt.subplots()
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
mean_y = 0.5
ax.axvline(x = 0)
ax.axvline(x = rwall)
ax.axhline(y = 0)
ax.axhline(y = topwall)


    
# balls = [body1, body2]
images = []
time_elapsed = 0

def init():
    global prevtime
    ax.set_ylim(-0.5, topwall + 0.5)
    ax.set_xlim(-0.5, rwall + 0.5)
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
    global mean_y
    # display time elapsed in sim
    
# =============================================================================
#     body1.updateball()
#     body2.updateball()
# =============================================================================
    currtime = time.time()
    dt = currtime - prevtime
    time_elapsed += dt
    time_text.set_text('Sim time: {:.1f}s, mean y: {:.2f}'.format(time_elapsed, mean_y))
    prevtime = time.time()
    
    # update positions of balls
    for ball in balls:
        ball.physics(dt)
    
    # account for collisions
  
    for i in range(len(balls)):
        b = balls[i]
        if (b.pos[0] < b.radius):
            left_wall_collision(b, 0)
            continue
        elif (b.pos[0] > rwall - b.radius):
            right_wall_collision(b, rwall)
            continue
        elif (b.pos[1] > topwall - b.radius):
            top_wall_collision(b, topwall)
            continue
        elif (b.pos[1] < b.radius):
            bot_wall_collision(b, 0)
        
        for j in range(i + 1, len(balls)):
            b2 = balls[j]
            
            if (np.linalg.norm(b.pos - b2.pos) < b.radius + b2.radius):
                ball_collision(b, b2, dt)
# =============================================================================
#                 b.pos = np.array([0, 0])
#                 b2.pos = np.array([1, 0])
#                 b.vel = np.array([0, 0])
#                 b2.vel = np.array([0, 0])
# =============================================================================
    toty = 0
    for ball in balls:
        toty += ball.pos[1]
        # ball.updateball()
        # print(ball.pos)
    mean_y = toty / N
    return images

ani = FuncAnimation(fig, update, frames=1000, interval=10, init_func=init,
                    blit=True)

plt.show()
