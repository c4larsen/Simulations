import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Press up and down arrows to speed up / slow down the sim, and press t to turn trail on/off

G = 6.6E-11
MASS = 6E24
RADIUS = 6E6
dt = 1/60
ff = 1

R0 = RADIUS + 6e5, np.pi/2
X0, Y0 = R0[0]*np.cos(R0[1]), R0[0]*np.sin(R0[1])

VCIRC = np.sqrt(G*MASS/R0[0])
SPEEDFRAC = 1
V0 = SPEEDFRAC*VCIRC, (R0[1] + np.pi/2)
V0X, V0Y = V0[0]*np.cos(V0[1]), V0[0]*np.sin(V0[1])

class body:
    def __init__(self, id, mass, xpos, ypos, xvel, yvel, xacc, yacc, time_elapsed):
        self.id = id
        self.mass = mass
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.xacc = xacc
        self.yacc = yacc
        self.time_elapsed = time_elapsed
        
def grav_acc(x,y):
    k = - G*MASS/((x**2 + y**2)**(3/2))
    a_x =  k*x
    a_y = k*y
    return a_x, a_y

body1 = body(1, 0.1, X0, Y0, V0X, V0Y, 0, 0, 0)

fig, ax = plt.subplots()
ax.set_xlim(-5*R0[0], 5*R0[0])
ax.set_ylim(-5*R0[0], 5*R0[0])
plt.gca().set_aspect('equal', adjustable='box')
xtrail, ytrail = [], []
ln, = plt.plot([], [], 'ro')
trail, = plt.plot(xtrail, ytrail, 'g-')
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

draw_trails = True

def key_presses(event):
    global draw_trails
    global xtrail
    global ytrail
    global ff
    global pause
    
    if event.key == 't':
        if (draw_trails):
            draw_trails = False
        else:
            xtrail = []
            ytrail = []
            draw_trails = True  
    elif event.key == 'h':
        visible = trail.get_visible()
        trail.set_visible(not visible)
        fig.canvas.draw()
    elif event.key == 'up':
        ff *= 10
    elif event.key == 'down':
        ff = ff/10
    
def init():
    circle = plt.Circle((0,0), RADIUS, color='b')
    ax.add_artist(circle)
    time_text.set_text('')
    fig.canvas.mpl_connect('key_press_event', key_presses)
    return ln, trail, time_text

def update(frame):
    delta = dt * ff
    if (body1.xpos**2 + body1.ypos**2 >= RADIUS**2):
        new_xvel = body1.xvel + body1.xacc * delta 
        new_yvel = body1.yvel + body1.yacc * delta 
        body1.xvel = new_xvel
        body1.yvel = new_yvel
        
        body1.time_elapsed += ff*dt
        
        new_xpos = body1.xpos + body1.xvel* delta
        new_ypos = body1.ypos + body1.yvel* delta

        body1.xpos = new_xpos
        body1.ypos = new_ypos
        ln.set_data(body1.xpos, body1.ypos)
        
        body1.xacc = grav_acc(body1.xpos, body1.ypos)[0]
        body1.yacc = grav_acc(body1.xpos, body1.ypos)[1]
        if (draw_trails):
            xtrail.append(body1.xpos)
            ytrail.append(body1.ypos)
            trail.set_data(xtrail, ytrail)
            
        time_text.set_text('Sim time: {:.1f} hrs (x{:.0f})'.format(body1.time_elapsed/3600, ff))
        
        return ln, trail, time_text
    
ani = FuncAnimation(fig, update, frames=1000, interval=1000*dt,
                    init_func=init, blit=True)

print('start the show')
plt.show()
