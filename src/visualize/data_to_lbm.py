import matplotlib as mpl
import sys
sys.path.append("../LBM")
from LBM import LBM
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.append("../data")
from extract import fetchdata

fig = plt.figure()

DATA = 2020100200
dx = 5600
dt = 10
c = dx / dt
v_real = 0.000015
viscosity = v_real * dt / dx**2
print(viscosity)

pressure, wind_speed  = fetchdata(DATA)

lbm = LBM(pressure.shape, viscosity=viscosity)
lbm.rho = pressure
lbm.u = wind_speed / c
im = plt.imshow(lbm.rho, cmap='jet', interpolation='nearest', animated=True)

def update(*args):
    lbm.forward_a_step()
    pressure = lbm.rho

    im.set_array(pressure)
    return im,

ani = animation.FuncAnimation(fig, update, interval=12, frames=1000)

ani.save("data_to_lbm.mp4", writer="ffmpeg", fps=30, bitrate=1000)
# plt.show()

