from types import prepare_class
import numpy as np

D = 2
Q = 9 # Here, 2D9Q model is implemented.
X = 0
Y = 1
W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
DIRECTIONS = [(0,0), (0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

class LBM:
    """The class of Lattice Boltzmann Method's field.

    This class makes Lattice Boltzmann Method(LBM)'s field which
    can be set with the velocity and density (or pressure). First,
    initialize the field with the args of size and viscosity.
    Then, you can assign the the velocity and/or density of the field.
    Use forward_a_step() to stream and collide the field.
    
    Attributes:
        u (np.ndarray): The velocity of the field. 
            u[0] is a horizontal, and u[1] is a vertical component.
        rho (np.ndarray): The density of the field.
        field (np.ndarray): The field value of LBM.
            directions are [0, N, S, E, W, NE, SE, NW, SW],
            corresponding to field[0], [1], ..., [8].
    
    """
    def __init__(self, size, viscosity=0.02):
        self._height, self._width = size
        self._u = np.zeros((D, self._height, self._width), np.float64)
        self._field = np.ones((Q, self._height, self._width), np.float64)
        self._rho = np.sum(self._field, axis=0)
        self._omega = 1 / (3 * viscosity + 0.5)

    def _shape_validator(self, value1, value2):
        if value1.shape != value2.shape:
            raise ValueError("Arrays' shape doesn't match.")

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        self._shape_validator(self._field, value)
        self._field = value
        self._recalc_rho()
        self._recalc_u()

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        self._shape_validator(self._u, value)
        self._u = value
        self._field = self._f_eq()
    
    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._shape_validator(self._rho, value)
        self._field *= value / self._rho
        self._rho = value

    def _recalc_rho(self):
        self._rho = np.sum(self._field, axis=0)

    def _recalc_u(self):
        self._u = np.zeros_like(self._u)
        for i in range(D):
            for j in range(Q):
                if DIRECTIONS[j][i] == 1:
                    self._u[i] += self.field[j]
                elif DIRECTIONS[j][i] == -1:
                    self._u[i] -= self.field[j]
            self._u[i] /= self._rho

    def _stream_field(self):
        for i in range(Q):
            x, y = DIRECTIONS[i]
            px1 = px2 = py1 = py2 = 0
            if x == 1: px1 = 1
            elif x == -1: px2 = 1
            if y == 1: py2 = 1
            elif y == -1: py1 = 1
            
            self._field[i] = np.pad(self._field[i], ((py1, py2), (px1, px2)), mode='edge')\
                [py2:self._height+py2, px2:self._width+px2]

        self._recalc_rho()
        self._recalc_u()
            

    def _collide_field(self):
        self._field = (1 - self._omega) * self._field +\
            self._omega * self._f_eq()


    def _f_eq(self):
        u2 = self._u ** 2
        one_minus_15u = 1 - 1.5 * (u2[X] + u2[Y])
        f_eq = np.zeros_like(self._field)

        for i in range(Q):
            # pre-calc the inner production to reduce the computation time
            cv = np.zeros_like(one_minus_15u) 
            for j in range(D):
                if DIRECTIONS[i][j] == 1:
                    cv += self._u[j]
                elif DIRECTIONS[i][j] == -1:
                    cv -= self._u[j]

            f_eq[i] = W[i] * self._rho * (one_minus_15u + 3 * cv + 4.5 * cv ** 2)

        return f_eq


    def forward_a_step(self):
        self._stream_field()
        self._collide_field()
