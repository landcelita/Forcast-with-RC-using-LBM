import sys
sys.path.append("../LBM")
from LBM.LBM import LBM
import numpy as np
import os
sys.path.append("../data")
from data.extract import fetchdata
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.python.keras import layers, models
from keras.callbacks import Callback

load_dotenv(override=True)
TEMP_DIR = os.getenv("TEMP_DIR")

# とりあえずいまは風速x方向のみだけど、後でそこも変数化しとく
CNN_CONST_FACTOR = 1/30 # CNNに入れる際になるべく0-1に近づけたいので定数倍しておく

class CNNwithLBM:
    INPUT_HEIGHT = 200
    INPUT_WIDTH = 200
    INPUT_UPPER_EDGE = 100 # 入力する上端
    INPUT_LEFT_EDGE = 100 # 入力する左端
    ANS_UPPER_EDGE = 155 
    ANS_LEFT_EDGE = 155
    ANS_BOTTOM_EDGE = 255
    ANS_RIGHT_EDGE = 255
    ANS_INTERVAL = 10
    #　ANS_...が例えば上から順に105, 105, 204, 204, 10なら、
    # 正解の風速の点は[105, 115, ... , 195] x [105, 115, ..., 195] (直積)
    # のような座標100個
    ANS_VERT_DIM = (ANS_BOTTOM_EDGE - ANS_UPPER_EDGE + ANS_INTERVAL - 1) // ANS_INTERVAL
        # 正解の風速の縦の座標[105, 115, ...]の次元
    ANS_HORI_DIM = (ANS_RIGHT_EDGE - ANS_LEFT_EDGE + ANS_INTERVAL - 1) // ANS_INTERVAL
        # 正解の風速の横の座標[105, 115, ...]の次元

    def __init__(self, delta=3, step=0, v_real=0.000015, dx=5600,\
                dt=10):
        # train, test: [20200101, ...]の形式のList[int]
        # delta: 何時間後を予想するか
        # step: LBMで何step回すか >0
        # v_real: 動粘性係数
        # dx: 格子間隔, dt: 代表時間

        self._viscosity = v_real * dt / dx**2
        self._dx = dx
        self._dt = dt
        self._c = dx / dt
        self._delta = delta
        self._step = step
            
        self._model = models.Sequential()
        self._model.add(layers.Conv2D(64, (5, 3),
            input_shape=(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1), use_bias=False))
        self._model.add(layers.AveragePooling2D((2, 2)))
        self._model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False))
        self._model.add(layers.AveragePooling2D((2, 2)))
        self._model.add(layers.Conv2D(8, (3, 3), padding='same', use_bias=False))
        self._model.add(layers.AveragePooling2D((2, 2)))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(8192, use_bias=False))
        self._model.add(layers.Dense(self.ANS_VERT_DIM * self.ANS_HORI_DIM, use_bias=False))
        self._model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam',
                    metrics=['mae'])

        self._model.summary()

    def learn_and_test(self, trains: list, tests: list, use_saved_data=False):
        cnn_trains_inputs = np.empty((len(trains), self.INPUT_WIDTH, self.INPUT_HEIGHT, 1))
        cnn_trains_correct_answers = np.empty((len(trains), 
                                            self.ANS_VERT_DIM * self.ANS_HORI_DIM))
        cnn_tests_inputs = np.empty((len(tests), self.INPUT_WIDTH, self.INPUT_HEIGHT, 1))
        cnn_tests_correct_answers = np.empty((len(tests), 
                                            self.ANS_VERT_DIM * self.ANS_HORI_DIM))

        if not use_saved_data:
            for i in range(len(trains)):
                lbm_x, lbm_y = self._data_into_LBM(trains[i])
                for _ in range(self._step):
                    lbm_x.forward_a_step()
                cnn_trains_inputs[i,:,:,0] = lbm_x.u[0,          # 風速の水平成分のみ
                                    self.INPUT_UPPER_EDGE:self.INPUT_UPPER_EDGE+self.INPUT_HEIGHT,
                                    self.INPUT_LEFT_EDGE:self.INPUT_LEFT_EDGE+self.INPUT_WIDTH]\
                                    * self._c * CNN_CONST_FACTOR
                cnn_trains_correct_answers[i,:] = lbm_y.u[0,   # 風速の水平成分のみ
                                    self.ANS_LEFT_EDGE:self.ANS_RIGHT_EDGE:self.ANS_INTERVAL,
                                    self.ANS_UPPER_EDGE:self.ANS_BOTTOM_EDGE:self.ANS_INTERVAL]\
                                    .flatten() * self._c * CNN_CONST_FACTOR

            print("learn data completed")

            for i in range(len(tests)):
                lbm_x, lbm_y = self._data_into_LBM(tests[i])
                for _ in range(self._step):
                    lbm_x.forward_a_step()
                cnn_tests_inputs[i,:,:,0] = lbm_x.u[0,          # 風速の水平成分のみ
                                    self.INPUT_UPPER_EDGE:self.INPUT_UPPER_EDGE+self.INPUT_HEIGHT,
                                    self.INPUT_LEFT_EDGE:self.INPUT_LEFT_EDGE+self.INPUT_WIDTH]\
                                    * self._c * CNN_CONST_FACTOR
                cnn_tests_correct_answers[i,:] = lbm_y.u[0,   # 風速の水平成分のみ
                                    self.ANS_LEFT_EDGE:self.ANS_RIGHT_EDGE:self.ANS_INTERVAL,
                                    self.ANS_UPPER_EDGE:self.ANS_BOTTOM_EDGE:self.ANS_INTERVAL]\
                                    .flatten() * self._c * CNN_CONST_FACTOR

            print("test data completed")

            np.save(TEMP_DIR+'cnn_trains_inputs', cnn_trains_inputs)
            np.save(TEMP_DIR+'cnn_trains_correct_answers', cnn_trains_correct_answers)
            np.save(TEMP_DIR+'cnn_tests_inputs', cnn_tests_inputs)
            np.save(TEMP_DIR+'cnn_tests_correct_answers', cnn_tests_correct_answers)
        else:
            cnn_trains_inputs = np.load(TEMP_DIR+'cnn_trains_inputs.npy')
            cnn_trains_correct_answers = np.load(TEMP_DIR+'cnn_trains_correct_answers.npy')
            cnn_tests_inputs = np.load(TEMP_DIR+'cnn_tests_inputs.npy')
            cnn_tests_correct_answers = np.load(TEMP_DIR+'cnn_tests_correct_answers.npy')

            print("use saved data")
        
        self._model.fit(cnn_trains_inputs,
                        cnn_trains_correct_answers,
                        batch_size=16,
                        epochs=50,
                        verbose=1,
                        callbacks=[TrueMAE(self._model, 
                                        cnn_tests_inputs,
                                        cnn_tests_correct_answers)])

        vs = self._model.evaluate(cnn_tests_inputs, cnn_tests_correct_answers)
        predictions = self._model.predict(cnn_tests_inputs)

        print(vs[0] / CNN_CONST_FACTOR)
        print(vs[1] / CNN_CONST_FACTOR)

        # print(cnn_tests_correct_answers[0] / CNN_CONST_FACTOR)
        # print(predictions[0] / CNN_CONST_FACTOR)

    
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

class TrueMAE(Callback):
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        pred = self.model.predict(self.X_val)
        true_mae = np.average(np.abs(self.y_val - pred) / CNN_CONST_FACTOR)
        print("true_mae =", true_mae)

# #test
# for st in range(40):
#     rc = RCwithLBM([2020010100, 2020010200, 2020010300, 2020010400, 2020010500, 2020010600],\
#                 [2020123000, 2020123100], delta=3, step=st)
#     rc.data_into_LBM_and_set_result()
#     rc.learn(0)
#     preds, diffs = rc.testing()
#     print(f"{st}: {np.average(diffs)}")
