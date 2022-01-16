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

class RC:
    def __init__(self):
        self._lbm = None
        self._initLBM()

    def _initLBM(self):
        self._lbm = LBM()