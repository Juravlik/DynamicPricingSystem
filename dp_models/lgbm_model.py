import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Union
import json
import lightgbm as lgb
import os


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
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)

    def save_best_model(self, path_to_save_model: str):
        os.makedirs(path_to_save_model, exist_ok=True)
        self.model.booster_.save_model(path_to_save_model)

    def load_model(self, path_to_model: str):
        self.model = lgb.Booster(model_file=path_to_model)


if __name__ == '__main__':

    with open('../configs/app_config.json') as json_file:
        config = json.load(json_file)

    df_trans = pd.read_csv(config['transactions_path'])
    df_samples = pd.read_csv(config['sample_1000_path'])