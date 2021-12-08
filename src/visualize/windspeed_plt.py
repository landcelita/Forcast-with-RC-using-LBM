from datetime import date, timedelta
import matplotlib as mpl
mpl.use('TkAgg')
import pygrib
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

load_dotenv(override=True)


def gen_timestr_by_3h(start, stop):
    """
    This generates a time string in the form of "yyyymmddhh" at 3-hour intervals.
    
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
DATA_DIR = os.getenv("DATA_DIR")

timestr_gen = gen_timestr_by_3h(START, STOP)

grbs = pygrib.open(DATA_DIR + str(YEAR) + "/" + next(timestr_gen) + ".grib2")
grb1, grb2 = grbs.select()[1:3]

ugrd = grb1.values
vgrd = grb2.values
windspeed = ugrd ** 2 + vgrd ** 2

im = plt.imshow(windspeed, cmap='jet', interpolation='nearest', animated=True)

def update(*args):
    timestr = next(timestr_gen)

    grbs = pygrib.open(DATA_DIR + str(YEAR) + "/" + timestr + ".grib2")
    grb1, grb2 = grbs.select()[1:3]

    ugrd = grb1.values
    vgrd = grb2.values

    windspeed = np.sqrt(ugrd ** 2 + vgrd ** 2) * 10 # 10 times to enhance the contrast

    im.set_array(windspeed)
    plt.title(timestr[0:4] + "/" + timestr[4:6] + "/" + timestr[6:8])
    return im,

ani = animation.FuncAnimation(fig, update, interval=20)
plt.show()