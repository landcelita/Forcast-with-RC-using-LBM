from time import sleep
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
dx = 1000
dt = 1
c = dx / dt
v_real = 0.0000015
viscosity = v_real * dt / dx**2
print(viscosity)

pressure, wind_speed  = fetchdata(DATA)

lbm = LBM(pressure.shape, dt=dt)
lbm.rho = pressure / 100000
lbm.u = wind_speed / c
im = plt.imshow(lbm.u[0], cmap='jet',\
                     interpolation='nearest', animated=True)

def update(frame):
    lbm._stream_field()
    lbm._collide_field()
    lbm._coriolis()
    show_value = lbm.u[0]

    im.set_array(show_value)
    return im,

ani = animation.FuncAnimation(fig, update, interval=1, frames=1000)

# ani.save("lbm_windspeed_with_coriolis.mp4", writer="ffmpeg", fps=30, bitrate=1000)
plt.show()

