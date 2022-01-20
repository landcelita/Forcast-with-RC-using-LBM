import sys
sys.path.append("../LBM")
from LBM.LBM import LBM
import numpy as np
from .Ridge import Ridge
sys.path.append("../data")
from data.extract import fetchdata

# とりあえずいまは風速x方向のみだけど、後でそこも変数化しとく

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
    def __init__(self, train, test, delta=3, step=0, v_real=0.000015, dx=5600,\
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
        self._step = step
        self._N_xd = ((Sedge-1-Nedge)//10 + 1) * ((Eedge-1-Wedge)//10 + 1)
                        # the number of nodes in the reservoir
        self._Nvert = (Sedge-1-Nedge)//10 + 1
        self._Nhorz = (Eedge-1-Wedge)//10 + 1
        self._X = np.empty((self._N_xd, 0))
        self._D = np.empty((self._N_xd, 0))
        self._Wout = None

    def data_into_LBM_and_set_result(self):
        for tr in self._train:
            lbm_X, lbm_D = self._data_into_LBM(tr)
            for i in range(self._step):
                lbm_X.forward_a_step()
            self._set_result(lbm_X, lbm_D)

    def learn(self, beta=0.001):
        ridge = Ridge(self._X.shape[0], self._D.shape[0], beta)
        ridge.set_DX(self._D, self._X)
        self._Wout = ridge.get_Wout_opt()
        # print(np.dot(self._Wout, self._X) - self._D) # print error
        return self._Wout

    def testing(self):
        preds = []
        diffs = []

        for test in self._test:
            lbm_X, lbm_D = self._data_into_LBM(test)
            for i in range(self._step):
                lbm_X.forward_a_step()
            x = lbm_X.u[:,self._Nedge:self._Sedge:self._interval,\
                    self._Wedge:self._Eedge:self._interval]
            d = lbm_D.u[:,self._Nedge:self._Sedge:self._interval,\
                    self._Wedge:self._Eedge:self._interval]
            pred = np.dot(self._Wout, x[0].reshape(-1, 1))\
                    .reshape((self._Nvert, self._Nhorz))
            ans = d[0] # 水平成分のみ
            diff = abs(pred - ans) * self._c # convert into m/s
            preds.append(pred)
            diffs.append(diff)

        return preds, diffs
            

    def _data_into_LBM(self, train):
        train_ans = train + self._delta
        pres, wind = fetchdata(train)
        pres_ans, wind_ans = fetchdata(train_ans)
        lbm_X = LBM(pres.shape, viscosity=self._viscosity, dt=self._dt)
        lbm_D = LBM(pres.shape, viscosity=self._viscosity, dt=self._dt)
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
        
        self._X = np.hstack((self._X, x[0].reshape(-1, 1))) # x方向の風速のみ
        self._D = np.hstack((self._D, d[0].reshape(-1, 1))) # 同様

# #test
# for st in range(40):
#     rc = RCwithLBM([2020010100, 2020010200, 2020010300, 2020010400, 2020010500, 2020010600],\
#                 [2020123000, 2020123100], delta=3, step=st)
#     rc.data_into_LBM_and_set_result()
#     rc.learn(0)
#     preds, diffs = rc.testing()
#     print(f"{st}: {np.average(diffs)}")
