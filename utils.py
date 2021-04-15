import sys
import logging
import numpy as np
import pandas as pd
from numpy.random import permutation
from collections import Counter

import nni
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score

logger = logging.getLogger('mnist_AutoML')

def load_data():
    train = pd.read_csv('./data/train.csv', index_col='index')
    test = pd.read_csv('./data/test.csv', index_col='index')

    categorical_columns = set(train.columns) - set(train.describe().columns)
    train = train.fillna('inoccupation')
    test = test.fillna('inoccupation')

    label_encoders = []
    for c in categorical_columns:
        encoder = LabelEncoder()
        train[c] = encoder.fit_transform(train[c])
        test[c] = encoder.transform(test[c])
        label_encoders.append(encoder)

    train = train.drop(['gender', 'FLAG_MOBIL', 'car', 'reality', 'phone', 'email'], axis=1)
    test = test.drop(['gender', 'FLAG_MOBIL', 'car', 'reality', 'phone', 'email'], axis=1)
    x_train = train.copy()
    y_train = x_train.pop('credit')

    return x_train, y_train, test

def smote_data(x_train, y_train, ratio):
    smote_on_1 = int(len(y_train) * ratio)
    smote_on_0 = int(len(y_train) * ratio)
    sm = SMOTE(sampling_strategy={0: smote_on_0, 1: smote_on_1}, k_neighbors=5, random_state=42)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
    y_train_res = pd.Series(y_train_res)
    print(Counter(y_train_res))
    return x_train_res, y_train_res

def run(x_train_res, y_train_res, x_train, y_train, params, model_num):
    bagging_predict_result = []
    data_index = permutation(len(y_train))
    if model_num == 1:
        model = CatBoostClassifier(learning_rate=params["learning_rate"], l2_leaf_reg=params["l2_leaf_reg"],
                                   depth=params["depth"], iterations=params["iterations"], border_count=params["border_count"], verbose=False)
        model.fit(X=x_train_res.iloc[data_index], y=y_train_res.iloc[data_index])
    elif model_num == 2:
        model = XGBClassifier(use_label_encoder=False, n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
                              min_child_weight=params["min_child_weight"], reg_lambda=params["reg_lambda"],
                              gamma=params["gamma"], depth=params["depth"],verbosity=0)
        model.fit(x_train_res.iloc[data_index], y_train_res.iloc[data_index])
    else:
        model = LGBMClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],subsample=params['subsample'],\
                             min_child_weight=params['min_child_weight'], reg_lambda =params['reg_lambda'],  silent=True)
        model.fit(x_train_res.iloc[data_index], y_train_res.iloc[data_index])
    y_pred = model.predict_proba(x_train)
    bagging_predict_result.append(y_pred)
    lloss = log_loss(y_train, y_pred)
    print(f'log loss:{lloss}\naccuracy: {accuracy_score(y_train, model.predict(x_train))}')
    nni.report_final_result(lloss)

def run_ensemble(x_train_res, y_train_res, x_train, y_train, test, model_num):
    bagging_predict_result = []
    predict_test = []
    for i in range(30):
        data_index = np.random.choice([j for j in range(len(y_train_res))], size=len(y_train_res))
        if model_num == 1:
            model = CatBoostClassifier(learning_rate=0.2, l2_leaf_reg=5, depth=12, iterations=500, border_count=20, verbose=False)
            model.fit(X=x_train_res.iloc[data_index], y=y_train_res.iloc[data_index])
        elif model_num == 2:
            model = XGBClassifier(use_label_encoder=False, n_estimators=100, learning_rate=0.5, min_child_weight=0.5, reg_lambda=1.5, gamma=0.5, depth=12,verbosity=0)
            model.fit(x_train_res.iloc[data_index], y_train_res.iloc[data_index])
        else:
            model = LGBMClassifier(n_estimators=300, learning_rate=0.2,subsample=0.75, min_child_weight=0.5, reg_lambda =3,  silent=True)
            model.fit(x_train_res.iloc[data_index], y_train_res.iloc[data_index])
        y_pred = model.predict(x_train)
        y_pred_prob = model.predict_proba(x_train)
        bagging_predict_result.append(y_pred_prob)
        predict_test.append(model.predict_proba(test))
        print(f'epoch: {i+1}\nlog loss:{log_loss(y_train, y_pred_prob)}\naccuracy: {accuracy_score(y_train, y_pred)}\nf1 score: {f1_score(y_train, y_pred, average="macro")}\n')
    mean_y_pred_prob_train = np.array(bagging_predict_result).mean(axis=0)
    mean_y_pred_train = np.argmax(mean_y_pred_prob_train, axis=1)
    print(f'log loss:{log_loss(y_train, mean_y_pred_prob_train)}\naccuracy: {accuracy_score(y_train, mean_y_pred_train)}\nf1 score: {f1_score(y_train, mean_y_pred_train, average="macro")}\n')
    return predict_test