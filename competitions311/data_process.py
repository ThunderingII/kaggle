import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from util.base_util import timer
from util.base_util import get_logger

log = get_logger()


def data_prepare():
    df_train = pd.read_csv('origin_data/train.csv')
    df_test = pd.read_csv('origin_data/test.csv')

    # None process
    none_list = ['age', '2_total_fee', '3_total_fee']
    remove_N_with_None(df_train, none_list)
    remove_N_with_None(df_test, none_list)
    fill_nan_mean(df_train, none_list)
    fill_nan_mean(df_test, none_list)

    # process gender
    def gender_process(x):
        if '1' in str(x):
            return 1
        elif '2' in str(x):
            return 2
        else:
            return 0

    df_train['gender'] = df_train['gender'].apply(gender_process)
    df_test['gender'] = df_test['gender'].apply(gender_process)

    # process age
    df_train['age_group'] = df_train['age'].apply(lambda x: x // 10)
    df_test['age_group'] = df_test['age'].apply(lambda x: x // 10)

    # process nan
    category_list = ['gender', 'service_type', 'is_mix_service', 'many_over_bill', 'contract_type',
                     'is_promise_low_consume', 'net_service', 'complaint_level', 'age_group']

    dummies(df_train, category_list)
    dummies(df_test, category_list)

    label = df_train['current_service']

    df_train['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']
    df_test['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']

    df_train['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)
    df_test['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)

    print('before align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    print('after align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train['current_service'] = label

    return df_train, df_test


def remove_N_with_None(df, columns, func=float):
    for c in columns:
        df[c] = df[c].apply(lambda x: np.nan if 'N' in str(x) else func(x))


def fill_nan_mean(df, columns):
    for c in columns:
        df[c].fillna(df[c].mean(), inplace=True)


def dummies(df, columns):
    le = preprocessing.LabelEncoder()
    if df is None:
        return
    for c in columns:
        if c not in df.columns:
            print('{} not in df'.format(c))
            continue
        if df[c].value_counts().count() <= 2:
            le.fit(df[c])
            df[c] = le.transform(df[c])
        else:
            d = pd.get_dummies(df[c], prefix=c)
            print('{} has divide into:\n\t\t\t\t{}'.format(c, list(d.columns)))
            for nc in d.columns:
                df[nc] = d[nc]
            df.drop(columns=[c], inplace=True)


def one_hot2label_index(y_pre_origin):
    y_pre_np = np.array(y_pre_origin)
    class_size = y_pre_np.shape[1]
    y_pre = np.zeros([len(y_pre_np), 1], dtype=np.int32)
    log.info('data shape {},data length {}'.format(y_pre_np.shape, len(y_pre_np)))
    for index, d in enumerate(y_pre_np):
        max = d[0]
        max_i = 0
        for i in range(1, class_size):
            if d[i] > max:
                max = d[i]
                max_i = i
        y_pre[index][0] = max_i
    return y_pre


encode_map = {}
decode_list = []


# get label index from label's value
def label2index(df, label_name, inplace=True):
    global decode_list
    global encode_map
    decode_list = sorted(list(df[label_name].value_counts().index))

    label_size = len(decode_list)

    for i in range(label_size):
        encode_map[decode_list[i]] = i
        log.info('{} \'s index is {}'.format(decode_list[i], i))
    t = df[label_name]
    t = t.apply(lambda x: encode_map[x])
    if inplace:
        df[label_name] = t
    print(df[label_name].value_counts())
    log.info('-' * 100)
    return t


def index2label(y_pre_label_index):
    return list(map(lambda x: decode_list[x], y_pre_label_index))
