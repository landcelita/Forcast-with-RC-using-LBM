import numpy as np

class Ridge:
    def __init__(self, N_x, N_y, beta):
        '''
        Attributes:
            N_x: the dimension of output from the reservoir
            N_y: the dimension of the product of output from 
                the reservoir and the weight matrix
            beta: the regularization parameter
        '''
        self.beta = beta
        self.X_XT = np.zeros((N_x, N_x))
        self.D_XT = np.zeros((N_y, N_y))
        self.N_x = N_x

    def set_DX(self, D, X):
        self.X_XT = np.dot(X, X.T)
        self.D_XT = np.dot(D, X.T)

    def get_Wout_opt(self):
        inv_X_XT_betaI = np.linalg.inv(self.X_XT \
                                + self.beta * np.identity(self.N_x))
        Wout_opt = np.dot(self.D_XT, inv_X_XT_betaI)                             
        return Wout_opt

# test
# x = np.array([[5,6,7],[1,2,3],[4,5,6]])
# d = np.array([[2,3,4],[4,5,6],[5,6,7]])
# r = Ridge(3,3,0.3)
# r.set_DX(d, x)
# print(r.get_Wout_opt())
