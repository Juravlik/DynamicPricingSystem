import pandas as pd
import numpy as np
from typing import List, Optional, Callable
import json


def create_one_hot_encoding(df: pd.DataFrame,
                            feature_name: str,
                            prefix: str = '') -> pd.DataFrame:
    one_hot = pd.get_dummies(df[feature_name], prefix=prefix)

    df = df.drop(feature_name, axis=1)
    df = df.join(one_hot)

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df['dates'] = pd.to_datetime(df['dates'])
    df['year'] = df['dates'].dt.year
    df['month'] = df['dates'].dt.month
    df['day'] = df['dates'].dt.day
    df['week_num'] = df['dates'].dt.week

    return df


def prepare_static_sku_features(df_samples: pd.DataFrame) -> pd.DataFrame:
    df_samples = df_samples.rename(columns={'sku_id': 'SKU'})

    df_samples['creation_expiration_days'] = (
            pd.to_datetime(df_samples['expiration_date']) - pd.to_datetime(df_samples['creation_date'])).dt.days

    # ohe
    for feat in ['fincode', 'ui1_code', 'vendor', 'brand_code']:
        df_samples = create_one_hot_encoding(df_samples, feat, prefix=feat)

    return df_samples.drop(columns=['ui2_code', 'ui3_code'])


def create_train_set(df_trans: pd.DataFrame,
                     df_samples: pd.DataFrame,
                     df_canc: pd.DataFrame,
                     df_promo: pd.DataFrame,
                     df_sales_plan: pd.DataFrame,
                     df_wholesale: pd.DataFrame) -> pd.DataFrame:

    df_trans['dates'] = pd.to_datetime(df_trans['dates'])

    df_num_purchases = (
        df_trans
        .groupby(['dates', 'SKU'])['user']
        .agg(num_purchases='count')
        .reset_index()
    )
    df_num_purchases = add_date_features(df_num_purchases)

    df_price = (
        df_trans
        .groupby(['dates', 'SKU'])['price']
        .agg(price='mean')
        .reset_index()
    )

    df_all = df_num_purchases.merge(df_samples, on=['SKU'], how='left')
    df_all = df_all.merge(df_canc, on=['SKU', 'year', 'week_num'], how='left')
    df_all = df_all.merge(df_promo, on=['SKU', 'week_num', 'year'], how='left')
    df_all = df_all.merge(df_sales_plan, on=['SKU', 'year', 'month'], how='left')
    df_all = df_all.merge(df_wholesale, on=['SKU', 'year', 'month', 'week_num'], how='left')
    df_all = df_all.merge(df_price, on=['dates', 'SKU'], how='left')

    df_all['mean_canc_price'] = (df_all['ret_net_1_price'] + df_all['ret_net_2_price'] + df_all['ret_net_3_price']) / 3
    df_all['min_canc_price'] = df_all[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].min(axis=1)
    df_all['max_canc_price'] = df_all[['ret_net_1_price', 'ret_net_2_price', 'ret_net_3_price']].max(axis=1)
    df_all['price_minus_min_canc_price'] = df_all['price'] - df_all['min_canc_price']
    df_all['price_minus_max_canc_price'] = df_all['price'] - df_all['max_canc_price']
    df_all['price_minus_mean_canc_price'] = df_all['price'] - df_all['mean_canc_price']
    df_all['price_div_min_canc_price'] = df_all['price'] / df_all['min_canc_price']
    df_all['price_div_max_canc_price'] = df_all['price'] / df_all['max_canc_price']
    df_all['price_div_mean_canc_price'] = df_all['price'] / df_all['mean_canc_price']

    df_all['promo_price'] = df_all['price'] * df_all['discount']
    df_all['date_creation_date'] = (
            pd.to_datetime(df_all['dates']) - pd.to_datetime(df_all['creation_date'])).dt.days
    df_all['expiration_dates_date'] = (
            pd.to_datetime(df_all['expiration_date']) - pd.to_datetime(df_all['dates'])).dt.days

    for i in range(1, 8):
        df_all['price_lag{}'.format(i)] = df_all.groupby(['SKU'])['price'].shift(i)
        df_all['num_purchases_lag{}'.format(i)] = df_all.groupby(['SKU'])['num_purchases'].shift(i)

    return df_all


if __name__ == '__main__':

    # df_samples = pd.read_csv('/home/juravlik/PycharmProjects/DynamicPricingSystem/data/raw/sample_1000.csv')
    # df_samples = prepare_static_sku_features(df_samples)
    # df_samples.to_csv('/home/juravlik/PycharmProjects/DynamicPricingSystem/data/prepared/sample_1000.csv',
    #                   index=False)
    #
    with open('../configs/app_config.json') as json_file:
        config = json.load(json_file)

    df_trans = pd.read_csv(config['transactions_path'])
    df_samples = pd.read_csv(config['sample_1000_path'])
    df_canc = pd.read_csv(config['canc_path'])
    df_promo = pd.read_csv(config['promo_path'])
    df_sales_plan = pd.read_csv(config['sales_plan_path'])
    df_wholesale = pd.read_csv(config['wholesale_trade_table_path'])

    df_train_set = create_train_set(df_trans, df_samples, df_canc,
                                    df_promo, df_sales_plan, df_wholesale)

    df_train_set.to_csv('/home/juravlik/PycharmProjects/DynamicPricingSystem/data/prepared/train_set.csv',
                        index=False)

