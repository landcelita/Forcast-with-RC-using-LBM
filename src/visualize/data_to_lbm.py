from datetime import date, timedelta
import matplotlib as mpl
import sys
sys.path.append("../LBM")
from LBM import LBM
mpl.use('TkAgg')
import pygrib
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

load_dotenv(override=True)

fig = plt.figure()

YEAR = 2020
DATA = 2020010100
DATA_DIR = os.getenv("DATA_DIR")

grbs = pygrib.open(DATA_DIR + str(YEAR) + "/" + str(DATA) + ".grib2")
grb = grbs.select()[0]
grb1, grb2 = grbs.select()[1:3]

pressure = grb.values
ugrd = grb1.values
vgrd = grb2.values
wind_speed = np.stack([ugrd, vgrd])

lbm = LBM(pressure.shape)
lbm.rho = pressure
lbm.u = wind_speed / 500
im = plt.imshow(lbm.rho, cmap='jet', interpolation='nearest', animated=True)

def update(*args):
    lbm.forward_a_step()
    pressure = lbm.rho

    im.set_array(pressure)
    return im,

ani = animation.FuncAnimation(fig, update, interval=20, frames=1000)
# ani.save("data_to_lbm.mp4", writer="ffmpeg", fps=30, bitrate=1000)
plt.show()