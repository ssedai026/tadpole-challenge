from sklearn.ensemble import GradientBoostingRegressor, \
    GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

import numpy as np

from sklearn.metrics import accuracy_score
from util import multiclass_roc_auc_score
import pickle
import os



def getGBMRegressor():
    #Configure GBM regression model
    params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 3,
              'learning_rate': 0.01, 'loss': 'huber', 'subsample': 0.2}
    reg = GradientBoostingRegressor(**params)
    return reg, 'gbm'


def getGBMClassifier():
    # Configure GBM classification model
    params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 3,
              'learning_rate': 0.01, 'loss': 'deviance', 'subsample': 0.2}  # exponential, deviance
    clf = GradientBoostingClassifier(**params)
    return clf, 'gbm'


def getRFRegressor():
    return  RandomForestRegressor(n_estimators=500, max_depth=None, random_state=0, bootstrap=True), 'rf'

def getRFClassifier():
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split = 2, random_state = 0), 'rf'




if (__name__ == '__main__'):

    #Load the training  data
    npzfile = np.load('data/cur_data_d1.npz')  #  data/d1.npz
    npzfile.files
    features = npzfile['features']
    labels = npzfile['labels']
    groups = npzfile['groups']
    features = np.nan_to_num(features)
    print ('#total number of patient', len(set(groups)))

    #split into train val
    gss = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42) # change train size to 100% to train model for submission

    idx_trn, idx_test = next(gss.split(labels, groups=groups))
    print('#train patient', idx_trn.shape[0], ' #test patient', idx_test.shape[0])

    X = features[idx_trn, :]
    print ('X ', X.shape)


    # Training of ventricle volume prediction
    y_vv = labels[idx_trn, 0]
    reg, type = getGBMRegressor()  # RandomForestRegressor(n_estimators=500, max_depth=None, random_state=0, bootstrap=True)
    reg.fit(X, y_vv)
    # Xt,yt, groupst = load_d2()
    Xt = features[idx_test, :]
    yt_vv = labels[idx_test, 0]

    ytvv_pred = reg.predict(Xt)
    print ('Mean absolute error for ventricle volume prediction using GBM regressor', np.mean(np.abs(yt_vv - ytvv_pred)))


    #training of disease state classification on validation set
    y = labels[idx_trn, 1]
    yt = labels[idx_test, 1]
    clf, type = getGBMClassifier()  # RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split = 2, random_state = 0)
    clf.fit(X, y)
    yt_pred = clf.predict(Xt)
    print ('Accuracy for 3 class problem using GBM classifier', accuracy_score(yt, yt_pred, normalize=True))
    print ('AUC for 3 class problem using GBM classifier', multiclass_roc_auc_score(yt, yt_pred))

    #save models
    if(not os.path.exists('models')):
        os.mkdir('models')
    pickle.dump(reg, open("models/d1_reg_vv.pickle", "wb"))
    pickle.dump(clf, open("models/d1_clf_dx.pickle", "wb"))
    print ('Models saved to models directory')

