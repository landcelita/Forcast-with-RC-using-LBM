from datetime import date, timedelta
import matplotlib as mpl
mpl.use('TkAgg')
import pygrib
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
sys.path.append("../data")
from extract import fetchdata

def gen_timestr_by_3h(start, stop):
    """This generates a time string in the form of "yyyymmddhh" at 3-hour intervals.
    
    Args:
        start (date): The date which it generates from.
        stop (date): The date which it generates until.

    Yields:
        str: A string in the form of "yyyymmddhh".

    Examples:
        >>> START = date(2020, 1, 1)
        >>> STOP = date(2020, 1, 3)
        >>> print([i for i in generate_timestr_by_3h(START, STOP)])
        ['2020010100', '2020010103', '2020010106', ..., '2020010221']
    """
    cur_day = start
    while cur_day < stop:
        for hour in range(0, 24, 3):
            cur_time = f"{cur_day:%Y%m%d}" + str(hour).zfill(2)
            yield cur_time
        cur_day += timedelta(days=1)


fig = plt.figure()

YEAR = 2020
START = date(YEAR, 1, 1)
STOP = date(YEAR+1, 1, 1)

timestr_gen = gen_timestr_by_3h(START, STOP)

_, wind = fetchdata(next(timestr_gen))
windspeed = np.sqrt(wind[0]**2 + wind[1]**2)

im = plt.imshow(windspeed, cmap='jet', interpolation='nearest', animated=True)

def update(*args):
    timestr = next(timestr_gen)

    _, wind = fetchdata(timestr)
    windspeed = np.sqrt(wind[0]**2 + wind[1]**2)

    im.set_array(windspeed)
    plt.title(timestr[0:4] + "/" + timestr[4:6] + "/" + timestr[6:8])
    return im,

ani = animation.FuncAnimation(fig, update, interval=20, frames=365*8)
# ani.save("windspeed_plt.mp4", writer="ffmpeg", fps=30, bitrate=1000)
plt.show()