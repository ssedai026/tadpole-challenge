from util import load_csv_as_dataframe
import pandas as pd
from feature_extractor import FeatureExtractor
import numpy as np
import pickle
from dateutil import parser
import monthdelta
import csv
from util import read_csv_file
from dateutil.relativedelta import relativedelta
import timeit


class LeaderBoard():
    def __init__(self, lb1_lb2_file='data/LeaderBoardData/TADPOLE_LB1_LB2.csv', d1_file='data/d1_data.csv'):
        lb1_lb2 = load_csv_as_dataframe(lb1_lb2_file)
        lb1_lb2['LB1'] = pd.to_numeric(lb1_lb2['LB1'])
        lb1_lb2['LB2'] = pd.to_numeric(lb1_lb2['LB2'])
        lb1_lb2['RID'] = pd.to_numeric(lb1_lb2['RID'])
        lb1 = lb1_lb2[(lb1_lb2.LB1 == 1)]
        lb2 = lb1_lb2[(lb1_lb2.LB2 == 1)]
        # if both zeros they belongto lb4
        print ('lb1lb2, lb1, lb2', len(lb1_lb2), len(lb1), len(lb2))

        d1 = load_csv_as_dataframe(d1_file)
        d1['RID'] = pd.to_numeric(d1['RID'])
        keys = ['RID', 'EXAMDATE']
        i1 = lb1.set_index(keys).index
        i2 = d1.set_index(keys).index
        self.d1_lb1 = d1[i2.isin(i1)]

        i1_ = lb2.set_index(keys).index
        self.d1_lb2 = d1[i2.isin(i1_)]

        print ('d1, intersection between d1 and (lb1,lb2)', len(d1), len(self.d1_lb1), len(self.d1_lb2))

    def get_lb1(self):
        return self.d1_lb1

    def get_lb2(self):
        return self.d1_lb2

    def get_d2(self, d2_file='data/d2_data.csv'):
        d2 = load_csv_as_dataframe(d2_file)
        return d2

    def get_d1(self, d1_file='data/d1_data.csv'):
        d1 = load_csv_as_dataframe(d1_file)
        return d1



class FuturePredictor:
    def __init__(self, vv_predictor, dx_predictor):
        self.vv_predictor = vv_predictor
        self.dx_predictor = dx_predictor

    def generate_pred_data(self, features, forecast_start_age_in_months, num_months):
        # find last age and add one
        n = features.shape[0]
        # last_age = features[n-1, age_index]


        # uncommet two lines below for d3 case. Last visits of d2 is a d3 dataset
        features = features[[n-1],:]
        n=1

        predictions_vv = []
        predictions_dx = []
        for new_age_interval in range(0, num_months):  # five year forcast
            new_age = new_age_interval + forecast_start_age_in_months
            new_age = np.ones((n, 1)) * new_age
            features_pred = np.hstack([features, new_age])
            y_pred_vv = self.vv_predictor.predict(features_pred)
            y_pred_vv = np.mean(y_pred_vv)

            y_pred_dx = self.dx_predictor.predict_proba(features_pred)
            y_pred_dx = np.mean(y_pred_dx, axis=0)



            predictions_vv.append(y_pred_vv)
            predictions_dx.append(y_pred_dx)
        return predictions_vv, predictions_dx

    def majority_vote(self, values):
        counts = np.bincount(values)
        return np.argmax(counts)


class SubFormatter:
    def __init__(self):
        self.results = []
        rows, first_header = read_csv_file('data/LeaderBoardData/TADPOLE_Submission_Leaderboard_TeamName.csv')
        self.results.append(first_header)
        # hrow=['RID', 'Forecast Month', 'Forecast Date', 'Ventricles_ICV']

    def prepare_row(self, rid, forecast_start_date, predictions_mri, predictions_dx):

        forecast_start_date = self.format_date(forecast_start_date)
        forecast_date = forecast_start_date

        for i, (pred_vv, pred_dx) in enumerate(zip(predictions_mri, predictions_dx)):
            f_month = i + 1
            # print 'fm', f_month
            arow = []
            arow.append(rid)
            arow.append(f_month)
            arow.append(str(forecast_date.year) + '-' + str(forecast_date.month))

            arow.append(pred_dx[0])
            arow.append(pred_dx[1])
            arow.append(pred_dx[2])

            self.add_blank_cells(arow, 3)
            arow.append(pred_vv)
            self.add_blank_cells(arow, 3)
            self.results.append(arow)

            forecast_date = forecast_date + monthdelta.monthdelta(1)

    def add_blank_cells(self, row, n):
        for i in range(n):
            row.append(0)

    def format_date(self, date_str):
        dt = parser.parse(date_str)
        return dt

    def export_csv(self, filename):

        with open(filename, 'w+') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for row in self.results:
                wr.writerow(row)


def get_age_at(last_age_in_months, last_exam_date, new_exam_date):
    """
    Computes the age in month at the new exam date
    :param last_age_in_months:
    :param last_exam_date:
    :param new_exam_date:
    :return:  the age in months at the new_exam_date
    """
    # n = features.shape[0]
    # last_age = features[n - 1, age_index] # this is in months
    lastexamdate = parser.parse(last_exam_date)
    forecast_start = parser.parse(new_exam_date)
    rd = relativedelta(forecast_start, lastexamdate)
    month_diff = rd.years * 12 + rd.months + rd.days / 30.0

    new_age_in_months = last_age_in_months + month_diff
    return new_age_in_months


if (__name__ == '__main__'):

    lb = LeaderBoard()
    lb1 = lb.get_d1()  # lb.get_lb1()
    lb2 = lb.get_d2()  # lb.get_lb2()


    cross_section = True
    reg = pickle.load(open("models/d1_reg_vv.pickle", "rb"))
    clf_dx = pickle.load(open("models/d1_clf_dx.pickle", "rb"))

    fp = FuturePredictor(reg, clf_dx)
    fmt = SubFormatter()

    dp = FeatureExtractor(dict_file='data/TADPOLE_D1_D2_Dict.csv', data=lb2)
    lb2_test = dp.prepare_data_test()
    N = 0
    for features, labels, rid, lastexamdate in lb2_test:
        features = np.nan_to_num(features)
        print ('###', rid, features.shape)
        N = N + features.shape[0]

        age_index = 7
        forecast_start_date = '2018-01-15'
        num_months_forecast = 62
        last_age_in_months = features[features.shape[0] - 1, age_index]

        forecast_start_age_in_months = get_age_at(last_age_in_months, lastexamdate, forecast_start_date)
        start_time = timeit.default_timer()

        if(cross_section):
            n = features.shape[0]
            features = features[[n - 1], :]

        predictions_vv, predictions_dx = fp.generate_pred_data(features, forecast_start_age_in_months,
                                                               num_months_forecast)
        elapsed = timeit.default_timer() - start_time

        fmt.prepare_row(rid, forecast_start_date, predictions_vv, predictions_dx)

        print (N)
    fmt.export_csv('results/TADPOLE_Submission_Leaderboard_d3.csv')

