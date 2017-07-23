__author__ = 'Nick Sarris (ngs5st)'

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb ## CATBOOST 3.6

class XgbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
                              watchlist, early_stopping_rounds=10)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

class LgbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 200)

    def train(self, xtra, ytra, xte, yte):
        ytra = ytra.ravel()
        yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra)
        dvalid = lgb.Dataset(xte, label=yte)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(x)

class CtbWrapper(object):
    def __init__(self, seed=2017, params=None):
        self.seed = seed
        self.nrounds = 250

    def train(self, xtra, ytra, xte, yte):
        self.gbdt = ctb.CatBoostRegressor(depth=6,
            iterations=self.nrounds, random_seed=self.seed,
            use_best_model=True, thread_count=16)

        xtra = pd.DataFrame(xtra)
        ytra = pd.DataFrame(ytra)
        xte = pd.DataFrame(xte)
        yte = pd.DataFrame(yte)

        self.gbdt.fit(X=xtra, y=ytra, eval_set=(xte,yte),
                      use_best_model=True)

    def predict(self, x):
        return self.gbdt.predict(x)

def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def create_features(train, test):

    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
    train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
    train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
    test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

    train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
    train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
    train.loc[:, 'pickup_day'] = train['pickup_datetime'].dt.day
    train.loc[:, 'pickup_month'] = train['pickup_datetime'].dt.month
    train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
    train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
    train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).map(
        lambda x: x.total_seconds())

    test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
    test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
    test.loc[:, 'pickup_day'] = test['pickup_datetime'].dt.day
    test.loc[:, 'pickup_month'] = test['pickup_datetime'].dt.month
    test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
    test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
    test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).map(
        lambda x: x.total_seconds())

    train.loc[:, 'distance_haversine'] = haversine_array(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
        train['pickup_latitude'].values, train['pickup_longitude'].values,
        train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test.loc[:, 'distance_haversine'] = haversine_array(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(
        test['pickup_latitude'].values, test['pickup_longitude'].values,
        test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    train.loc[:, 'average_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
    train.loc[:, 'average_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

    for gby_col in ['pickup_hour', 'pickup_day', 'pickup_date', 'pickup_weekday']:
        gby = train.groupby(gby_col).mean()[['average_speed_h', 'average_speed_m']]
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
        test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

    train['direction_ns'] = (train.pickup_latitude>train.dropoff_latitude)*1+1
    indices = train[(train.pickup_latitude == train.dropoff_longitude) & (train.pickup_latitude!=0)].index
    train.loc[indices,'direction_ns'] = 0

    train['direction_ew'] = (train.pickup_longitude>train.dropoff_longitude)*1+1
    indices = train[(train.pickup_longitude == train.dropoff_longitude) & (train.pickup_longitude!=0)].index
    train.loc[indices,'direction_ew'] = 0

    test['direction_ns'] = (test.pickup_latitude>test.dropoff_latitude)*1+1
    indices = test[(test.pickup_latitude == test.dropoff_longitude) & (test.pickup_latitude!=0)].index
    test.loc[indices,'direction_ns'] = 0

    test['direction_ew'] = (test.pickup_longitude>test.dropoff_longitude)*1+1
    indices = test[(test.pickup_longitude == test.dropoff_longitude) & (test.pickup_longitude!=0)].index
    test.loc[indices,'direction_ew'] = 0

    cols_to_drop = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration',
                    'check_trip_duration', 'pickup_date', 'average_speed_h',
                    'average_speed_m']

    features = [f for f in train.columns if f not in cols_to_drop]

    train_x = train[features]
    labels = np.log(train['trip_duration'].values + 1)
    test_x = test[features]

    for f in train_x.columns:
        if train_x[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_x[f].values))
            train_x[f] = lbl.transform(list(train_x[f].values))
            test_x[f] = lbl.transform(list(test_x[f].values))

    return train_x, labels, test_x

def get_oof(clf, ntrain, ntest, kf, train, labels, test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = train[train_index]
        y_tr = labels[train_index]
        x_te = train[test_index]
        y_te = labels[test_index]

        clf.train(x_tr, y_tr, x_te, y_te)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def model_1(train, labels, test):

    train = np.array(train)
    test = np.array(test)
    labels = np.array(labels)

    ntrain = train.shape[0]
    ntest = test.shape[0]

    kf = KFold(ntrain, n_folds=5,
        shuffle=True, random_state=2017)

    lgb_params = {}
    lgb_params['boosting_type'] = 'gbdt'
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'rmse'
    lgb_params['num_leaves'] = 96
    lgb_params['max_depth'] = 6
    lgb_params['feature_fraction'] = 0.9
    lgb_params['bagging_fraction'] = 0.95
    lgb_params['bagging_freq'] = 5
    lgb_params['learning_rate'] = 0.1
    lgb_params['early_stopping_round'] = 20

    xgb_params = {}
    xgb_params["objective"] = "reg:linear"
    xgb_params["eta"] = 0.2
    xgb_params["subsample"] = 0.95
    xgb_params["silent"] = 1
    xgb_params["lambda"] = 1
    xgb_params["max_depth"] = 10
    xgb_params["min_child_weight"] = 10
    xgb_params['eval_metric'] = 'rmse'
    xgb_params['seed'] = 2017

    cg = CtbWrapper()
    xg = XgbWrapper(seed=2017, params=xgb_params)
    lg = LgbWrapper(seed=2017, params=lgb_params)

    lg_oof_train, lg_oof_test = get_oof(lg, ntrain, ntest, kf, train, labels, test)
    xg_oof_train, xg_oof_test = get_oof(xg, ntrain, ntest, kf, train, labels, test)
    cg_oof_train, cg_oof_test = get_oof(cg, ntrain, ntest, kf, train, labels, test)

    print("CG-CV: {}".format(mean_squared_error(labels, cg_oof_train)))
    print("XG-CV: {}".format(mean_squared_error(labels, xg_oof_train)))
    print("LG-CV: {}".format(mean_squared_error(labels, lg_oof_train)))

    x_train = np.concatenate((cg_oof_train, xg_oof_train, lg_oof_train), axis=1)
    x_test = np.concatenate((cg_oof_test, xg_oof_test, lg_oof_test), axis=1)
    
    return x_train, labels, x_test
    
def model_2(train, labels, test):

    dtrain = xgb.DMatrix(train, label=labels)
    dtest = xgb.DMatrix(test)

    xgb_params = {}
    xgb_params["objective"] = "reg:linear"
    xgb_params["eta"] = 0.1
    xgb_params["subsample"] = 0.7
    xgb_params["silent"] = 1
    xgb_params["max_depth"] = 6
    xgb_params['eval_metric'] = 'rmse'
    xgb_params['min_child_weight'] = 5
    xgb_params['seed'] = 22424

    res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=5, seed=2017, stratified=False,
                 early_stopping_rounds=25, verbose_eval=10, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]

    print('')
    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
    bst = xgb.train(xgb_params, dtrain, best_nrounds)

    preds = np.exp(bst.predict(dtest))
    return preds
    
def main():
    
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    x_train, labels, x_test = create_features(train, test)
    x_train, labels, x_test = model_1(x_train, labels, x_test)
    preds = model_2(x_train, labels, x_test)
    
    test['trip_duration'] = preds
    test[['id', 'trip_duration']].to_csv('first_submission.csv', index=False)

if __name__ == '__main__':
    main()



