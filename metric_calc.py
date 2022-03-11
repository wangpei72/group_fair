import sys

import numpy as np
import time
import predictor
from group_fairness_metric import equality_of_oppo, disparte_impact, statistical_parity_difference
sys.path.append("../")


def cal_metric_avg(tuple_list):
    di_avg = 0.
    spd_avg = 0.
    eoop_avg = 0.
    eood_avg = 0.
    for i in range(5):  # 这里的range需要与predictor中getresults的range保持一致
        X, y, y_, accu = tuple_list[i]
        for j in range(20):
            di = disparte_impact.D_I_adult_age(
                X=X[j], y=y_[j]
            )
            spd = statistical_parity_difference.S_P_D_adult_age(
                X=X[j], y=y_[j]
            )
            eoop = equality_of_oppo.E_Oppo_adult_age(
                X=X[j], y_true=y[j], y_pre=y_[j]
            )
            eood = equality_of_oppo.E_Odds_adult_age(
                X=X[j], y_true=y[j], y_pre=y_[j]
            )
            di_avg += di
            spd_avg += spd
            eoop_avg += eoop
            eood_avg += eood
            if i == 4:
                di_avg /= 5.
                spd_avg /= 5.
                eoop_avg /= 5.
                eood_avg /= 5.
    return di_avg, spd_avg, eoop_avg, eood_avg


def npy_saver(tuple_res):
    for item in tuple_res:


if __name__ == '__main__':
    tuple_list =predictor.get_5_results()
    cal_metric_avg(tuple_list)
