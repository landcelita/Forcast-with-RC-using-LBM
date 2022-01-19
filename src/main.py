from Learning.RCwithLBM import RCwithLBM
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

def main():
    nowtime = time.time()

    train, test = get_train_and_test_date()
    rc = RCwithLBM(train, test, step=5)

    diffs = []
    for date in test:
        _, wind0 = fetchdata(date)
        _, wind3 = fetchdata(date+3)
        diffs.append(np.average(abs(wind0[0] - wind3[0]))) # 水平成分のみ
    print(f"change in wind speed for 3 hr: {np.average(diffs)}")
    print(f"{time.time()-nowtime} s")
    nowtime = time.time()


    rc.data_into_LBM_and_set_result()
    print("result set")
    print(f"{time.time()-nowtime} s")
    nowtime = time.time()

    rc.learn(beta=0.001)
    print("learned")
    print(f"{time.time()-nowtime} s")
    nowtime = time.time()

    preds, diffs = rc.testing()
    print(f"L1 mean error: {np.average(diffs)} m/s")
    print("tested")
    print(f"{time.time()-nowtime} s")

    

if __name__ == "__main__":
    main()