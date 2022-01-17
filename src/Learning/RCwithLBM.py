import sys
sys.path.append("../LBM")
from LBM import LBM
import numpy as np
from Ridge import Ridge
sys.path.append("../data")
from extract import fetchdata

class RCwithLBM:
    '''The class of reservoir computing with LBM.

    Attributes:
        train: the array of a set of training data's time.
            eg. [2020010100, 2020010200, ...]
        test: the array of a set of test data's time.
            eg. [2021010100, 2021010200, ...]
        delta: How far ahead in hours you want to be correct.
            If you have test data of 2021010100 and delta = 3,
            the desired answer will be 2021010103.
    '''
    def __init__(self, train, test, delta=3):
        self._train = train
        self._test = test
        self._delta = 3

    def data_into_LBM_and_set_result(self, step):
        for tr in self._train:
            lbm = self._data_into_LBM(tr)
            for i in range(step):
                lbm.forward_a_step()
            self._set_result(lbm)

    def _data_into_LBM(self, train):
        
        lbm = LBM()

    def _set_result(lbm):
        pass


