from Learning.CNNwithLBM import CNNwithLBM
from datetime import timedelta
from datetime import datetime as dt
import time
import numpy as np
from data.extract import fetchdata


TRAIN_YEAR_FROM = "2011"
TRAIN_YEARS = 9
TRAIN_DAYS_FROM = "0701"
TRAIN_DAYS_A_YEAR = 92
TRAIN_TIME = "00"

TEST_YEAR_FROM = "2020"
TEST_YEARS = 1
TEST_DAYS_FROM = "0701"
TEST_DAYS_A_YEAR = 92
TEST_TIME = "00"

def get_train_and_test_date():
    """[20200701, 20200702, 20200703, ...]という配列をtrain, testに入れて返す"""
    train = []
    for yr in range(int(TRAIN_YEAR_FROM), int(TRAIN_YEAR_FROM)+int(TRAIN_YEARS)):
        start = dt.strptime(str(yr)+TRAIN_DAYS_FROM, '%Y%m%d')
        for i in range(int(TRAIN_DAYS_A_YEAR)):
            d = (start + timedelta(days=i)).strftime("%Y%m%d") + TRAIN_TIME
            train.append(int(d))

    test = []
    for yr in range(int(TEST_YEAR_FROM), int(TEST_YEAR_FROM)+int(TEST_YEARS)):
        start = dt.strptime(str(yr)+TEST_DAYS_FROM, '%Y%m%d')
        for i in range(int(TEST_DAYS_A_YEAR)):
            d = (start + timedelta(days=i)).strftime("%Y%m%d") + TEST_TIME
            test.append(int(d))

    return train, test

def exec(step, dt, dx, v_real):
    txt = open(f"dt{dt}_dx{dx}_v_real{v_real}.txt", 'a')

    str = f"====step: {step}, dt: {dt}, dx: {dx}, v_real: {v_real}===="
    print(str)
    txt.write(str + "\n")
    nowtime = time.time()

    # 訓練とテストの日付のリストを取ってくる
    trains, tests = get_train_and_test_date()
    cnn = CNNwithLBM(step=step, dt=dt, dx=dx, v_real=v_real)

    # n時とn+3時とのMAEを出力するだけ
    diffs = []
    for date in tests:
        _, wind0 = fetchdata(date)
        _, wind3 = fetchdata(date+3)
        diffs.append(np.average(abs(wind0[0] - wind3[0]))) # 水平成分のみ
    str = f"change in wind speed for 3 hr: {np.average(diffs)} ({time.time()-nowtime} s)"
    print(str)
    txt.write(str + "\n")

    # 学習とテスト
    cnn.learn_and_test(trains, tests, use_saved_data=True)

    txt.close()

def main():
    dt=10
    dx=5600
    v_real=0.000015

    exec(0, dt, dx, v_real) # とりあえず0step

    # for step in range(0,21):
    #     exec(step, dt, dx, v_real)

    

if __name__ == "__main__":
    main()