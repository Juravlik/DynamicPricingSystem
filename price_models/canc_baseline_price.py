import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Union
from data_preparation.features_creating import add_date_features


class CancBaseline:

    def __init__(self, strategy: str, coef: float):
        self.strategy = strategy
        self.coef = coef
        self.canc_feats = None

    def fit(self, df_canc: pd.DataFrame, df_transactions: pd.DataFrame):

        df_canc['mean_canc_price'] = (df_canc['ret_net_1_price'] + df_canc['ret_net_2_price']
                                      + df_canc['ret_net_3_price']) / 3
        df_canc['min_canc_price'] = df_canc[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].min(axis=1)
        df_canc['max_canc_price'] = df_canc[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].max(axis=1)

        self.canc_feats = df_canc

        self.trans_last_price = (
            df_transactions
            .groupby(['dates', 'SKU'])['price']
            .agg(last_train_price='mean')
            .reset_index()
            .sort_values('dates')
            .groupby('SKU')
            .last()
            .reset_index()[['SKU', 'last_train_price']]
        )

    def predict(self, df_batch: pd.DataFrame) -> np.array:

        df_batch = add_date_features(df_batch)

        df_batch = df_batch.merge(self.canc_feats, how='left', on=['year', 'week_num', 'SKU'])
        df_batch = df_batch.merge(self.trans_last_price, how='left', on=['SKU'])

        if self.strategy == 'min_related':
            df_batch['price'] = df_batch['min_canc_price'] * self.coef
        if self.strategy == 'max_related':
            df_batch['price'] = df_batch['max_canc_price'] * self.coef
        if self.strategy == 'mean':
            df_batch['price'] = df_batch['mean_canc_price']

        df_batch.loc[df_batch.price.isna(), 'price'] = df_batch['last_train_price']

        df_batch['dates'] = df_batch['dates'].dt.strftime('%Y-%m-%d')

        return df_batch[['dates', 'SKU', 'user_id', 'price']]

    @staticmethod
    def __preprocess_canc_df(df_canc: pd.DataFrame) -> pd.DataFrame:
        df_canc['mean_canc_price'] = (df_canc['ret_net_1_price'] + df_canc['ret_net_2_price']
                                      + df_canc['ret_net_3_price']) / 3
        df_canc['min_canc_price'] = df_canc[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].min(axis=1)
        df_canc['max_canc_price'] = df_canc[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].max(axis=1)

        return df_canc
