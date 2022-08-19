import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Union
import json
import lightgbm as lgb
import os
from sklearn.metrics import mean_absolute_error


class LGBMModel:

    def __init__(self, **params):
        self.model = lgb.LGBMRegressor(**params)

    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            eval_metric: Union[Callable, str],
            early_stopping_rounds=1000):

        self.model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)],
                       eval_metric=eval_metric,
                       early_stopping_rounds=early_stopping_rounds)

    def predict(self, X_test: pd.DataFrame) -> np.array:
        return self.model.predict(X_test)

    def save_model(self, path_to_save_model: str):
        os.makedirs(os.path.dirname(path_to_save_model), exist_ok=True)
        self.model.booster_.save_model(path_to_save_model)

    def load_model(self, path_to_model: str):
        self.model = lgb.Booster(model_file=path_to_model)


if __name__ == '__main__':

    with open('../configs/app_config.json') as json_file:
        config = json.load(json_file)

    df_train = pd.read_csv('/home/juravlik/PycharmProjects/DynamicPricingSystem/data/prepared/train_set.csv')

    drop_features = ['dates', 'SKU', 'creation_date', 'expiration_date']

    train_set = df_train[~((df_train.year == 2019) & (df_train.month == 11))].drop(columns=drop_features)
    val_set = df_train[(df_train.year == 2019) & (df_train.month == 11)].drop(columns=drop_features)

    X_train, y_train = train_set.drop(columns=['num_purchases']), train_set[['num_purchases']]
    X_val, y_val = val_set.drop(columns=['num_purchases']), val_set[['num_purchases']]

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 8,
        "num_leaves": 128,
        "max_bin": 512,
        "num_iterations": 1000
    }

    lgb_model = LGBMModel(**params)

    lgb_model.fit(X_train, y_train, X_val, y_val,
                  eval_metric='rmse', early_stopping_rounds=1000)

    lgb_model.save_model('/home/juravlik/PycharmProjects/DynamicPricingSystem/models/lgb_1.txt')
    print(mean_absolute_error(lgb_model.predict(X_val), y_val))

    lgb_model.load_model('/home/juravlik/PycharmProjects/DynamicPricingSystem/models/lgb_1.txt')
    print(mean_absolute_error(lgb_model.predict(X_val), y_val))
