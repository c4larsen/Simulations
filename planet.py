from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

G = 6.6E-11
MASS = 6E24
RADIUS = 6E6
dt = 1/60
ff = 1000

class body:
    def __init__(self, id, mass, xpos, ypos, xvel, yvel, xacc, yacc):
        self.id = id
        self.mass = mass
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.xacc = xacc
        self.yacc = yacc
        
def grav_acc(x,y):
    k = - G*MASS/((x**2 + y**2)**(3/2))
    a_x =  k*x
    a_y = k*y
    return a_x, a_y

body1 = body(1, 0.1, (1.5)*RADIUS, 0.0, 0.0, 7500.0, 0, 0)

# Animation
fig, ax = plt.subplots()
ax.set_xlim(-8*RADIUS, 8*RADIUS)
ax.set_ylim(-8*RADIUS, 8*RADIUS)
plt.gca().set_aspect('equal', adjustable='box')
xtrail, ytrail = [], []
ln, = plt.plot([], [], 'ro')
trail, = plt.plot(xtrail, ytrail, 'g-')

def init():
    circle = plt.Circle((0,0), RADIUS, color='b')
    ax.add_artist(circle)
    return ln, trail,

def update(frame):

    delta = dt * ff
    if (body1.xpos**2 + body1.ypos**2 >= RADIUS**2):
        new_xvel = body1.xvel + body1.xacc * delta 
        new_yvel = body1.yvel + body1.yacc * delta 
        body1.xvel = new_xvel
        body1.yvel = new_yvel
        
        new_xpos = body1.xpos + body1.xvel* delta
        new_ypos = body1.ypos + body1.yvel* delta

        body1.xpos = new_xpos
        body1.ypos = new_ypos
        ln.set_data(body1.xpos, body1.ypos)
        
        body1.xacc = grav_acc(body1.xpos, body1.ypos)[0]
        body1.yacc = grav_acc(body1.xpos, body1.ypos)[1]

        xtrail.append(body1.xpos)
        ytrail.append(body1.ypos)
        trail.set_data(xtrail, ytrail)
        return ln, trail,
    
ani = FuncAnimation(fig, update, frames=1000, interval=1000*dt,

                    init_func=init, blit=True)

print('start the show')
plt.show()
