import csv
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def read_csv_file(file_path):


    with open(file_path) as f:
        reader = csv.reader(f)
        rows = []
        first_header = next(reader)
        len_head = len(first_header)
        for row in reader:

            if (len(row) != len_head):
                print ('Inconsistent row size in the matrix', len(
                    row), ' expected same as length of the header', len_head)
            rows.append(row)
    return rows, first_header


def string2float(x):
    if (type(x) is list):
        return string2float_arr(x)
    else:
        y = 0
        success = True
        try:
            y = float(x)
        except ValueError:
            success = False

        return y, success


def string2float_arr(x):
    y = np.zeros((len(x),))
    success = np.ones((len(x),))

    for i in range(len(x)):
        try:
            y[i] = float(x[i])

        except ValueError:
            success[i] = 0

    return y, success


def load_csv_as_dataframe(csv_file):
    """
    Assumes the first row in the file is a header
    :param self:
    :param csv_file:
    :return:
    """
    rows, first_header = read_csv_file(csv_file)
    df = pd.DataFrame(data=rows, columns=first_header)
    return df


def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)
