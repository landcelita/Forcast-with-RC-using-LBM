import os
import pygrib
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)
DATA_DIR = os.getenv("DATA_DIR")

def fetchdata(time):
    str_year = str_time = None
    if isinstance(time, int):
        str_year = str(time)[0:4]
        str_time = str(time)
    elif isinstance(time, str):
        str_year = str(time)[0:4]
        str_time = str(time)
    
    grbs = pygrib.open(DATA_DIR + str_year + "/" + str_time + ".grib2")
    pressure = grbs.select()[0].values
    ugrd = grbs.select()[1].values
    vgrd = grbs.select()[2].values
    wind_speed = np.stack([ugrd, vgrd])
    
    return pressure, wind_speed
