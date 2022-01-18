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
        train: the array of a set of training data's time (int).
            eg. [2020010100, 2020010200, ...]
        test: the array of a set of test data's time (int).
            eg. [2021010100, 2021010200, ...]
        delta: How far ahead in hours you want to be correct.
            If you have test data of 2021010100 and delta = 3,
            the desired answer will be 2021010103.
    '''
    def __init__(self, train, test, delta=3, v_real=0.000015, dx=5600,\
                dt=10, Nedge=205, Sedge=305, Wedge=195, Eedge=295, interval=10):
        self._viscosity = v_real * dt / dx**2
        self._dx = dx
        self._dt = dt
        self._c = dx / dt
        self._train = train
        self._test = test
        self._delta = delta
        self._Nedge = Nedge
        self._Sedge = Sedge
        self._Wedge = Wedge
        self._Eedge = Eedge
        self._interval = interval
        self._N_xd = ((Sedge-1-Nedge)//10 + 1) * ((Eedge-1-Wedge)//10 + 1)
                        # the number of nodes in the reservoir
        self._X = np.empty((self._N_xd, 0))
        self._D = np.empty((self._N_xd, 0))
        self._Wout = None

    def data_into_LBM_and_set_result(self, step):
        for tr in self._train:
            lbm_X, lbm_D = self._data_into_LBM(tr)
            for i in range(step):
                lbm_X.forward_a_step()
            self._set_result(lbm_X, lbm_D)

    def learn(self, beta=0.001):
        ridge = Ridge(self._X.shape[0], self._D.shape[0], beta)
        ridge.set_DX(self._D, self._X)
        self._Wout = ridge.get_Wout_opt()
        # print(np.dot(self._Wout, self._X) - self._D) # print error
        return self._Wout

    def _data_into_LBM(self, train):
        train_ans = train + self._delta
        pres, wind = fetchdata(train)
        pres_ans, wind_ans = fetchdata(train_ans)
        lbm_X = LBM(pres.shape, viscosity=self._viscosity)
        lbm_D = LBM(pres.shape, viscosity=self._viscosity)
        lbm_X.rho = pres
        lbm_X.u = wind / self._c
        lbm_D.rho = pres_ans
        lbm_D.u = wind_ans / self._c

        return lbm_X, lbm_D

    def _set_result(self, lbm_X, lbm_D):
        x = lbm_X.u[:,self._Nedge:self._Sedge:self._interval,\
                    self._Wedge:self._Eedge:self._interval]
        d = lbm_D.u[:,self._Nedge:self._Sedge:self._interval,\
                    self._Wedge:self._Eedge:self._interval]
        
        self._X = np.hstack((self._X, x[0].reshape(-1, 1)))
        self._D = np.hstack((self._D, d[0].reshape(-1, 1)))

# #test
# rc = RCwithLBM([2020010100, 2020010200], [2020010100, 2020010200], delta=0)
# rc.data_into_LBM_and_set_result(0)
# print(rc.calc_Wout(0))
