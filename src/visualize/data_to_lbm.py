import matplotlib as mpl
import sys
sys.path.append("../LBM")
from LBM import LBM
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append("../data")
import numpy as np
from extract import fetchdata

fig = plt.figure()

DATA = 2020100200
dx = 5600
dt = 5
c = dx / dt
v_real = 0.000015
viscosity = v_real * dt / dx**2
print(viscosity)

pressure, wind_speed  = fetchdata(DATA)

lbm = LBM(pressure.shape, dt=dt)
lbm.rho = pressure
lbm.u = wind_speed / c
im = plt.imshow(np.sqrt(wind_speed[0]**2 + wind_speed[1]**2), cmap='jet',\
                     interpolation='nearest', animated=True)

def update(*args):
    lbm.forward_a_step()
    wind_speed = np.sqrt((lbm.u[0]*c)**2 + (lbm.u[1]*c)**2)

    im.set_array(wind_speed)
    return im,

ani = animation.FuncAnimation(fig, update, interval=12, frames=1000)

# ani.save("lbm_windspeed_with_coriolis.mp4", writer="ffmpeg", fps=30, bitrate=1000)
plt.show()

