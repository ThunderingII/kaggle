import pandas as pd
import numpy as np

from sklearn import preprocessing
from util.base_util import get_logger
from util.base_util import timer
from competitions311 import base_data_process
from util import base_util
import os, random
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')
ID = 'user_id'
LABEL = 'current_service'
log = get_logger()


def data_prepare(df_train, df_test):
    conti_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'contract_time',
                  'former_complaint_fee', 'former_complaint_num', 'last_month_traffic', 'local_caller_time',
                  'local_trafffic_month', 'month_traffic', 'online_time', 'pay_num', 'pay_times',
                  'service1_caller_time',
                  'service2_caller_time', 'pay_num_per_time', 'll']

    normalize_process(df_train, df_test, conti_list)
    # label 2 index
    base_data_process.label2index(df_train, LABEL)

    base_util.pickle_dump((base_data_process.encode_map, base_data_process.decode_list), 'origin_data/label2index.pkl')

    with timer('save train data'):
        df_train.to_csv('origin_data/train_modified.csv', index=False)
    with timer('save test data'):
        df_test.to_csv('origin_data/test_modified.csv', index=False)


def normalize_process(df_train, df_test, conti_list):
    for col in conti_list:
        df_train[col] = (df_train[col] - df_train[col].min()) / (df_train[col].max() - df_train[col].min())
        df_test[col] = (df_test[col] - df_test[col].min()) / (df_test[col].max() - df_test[col].min())


def read_corpus(corpus_path):
    df = pd.read_csv(corpus_path)
    # drop user_id
    id_col = df[ID]
    df.drop([ID], axis=1, inplace=True)
    return id_col, df


def load_label2index():
    # 返回了 map 和list
    return base_util.pickle_load('origin_data/label2index.pkl')


def batch_yield(df, batch_size):
    if batch_size == -1:
        batch_size = len(df)
    # equal to shuffle
    df = df.sample(frac=1)
    # len(df) // batch_size * batch_size <= len(df)
    total_batch = len(df) // batch_size

    for i in range(total_batch):
        data = df.iloc[i * batch_size:i * batch_size + batch_size, :]
        labels = data[LABEL]
        data.drop([LABEL], axis=1, inplace=True)
        yield data, labels


def save_result(ids, labels, submit_path):
    df_test = pd.DataFrame()
    df_test[ID] = ids
    df_test[LABEL] = labels
    df_test.columns = [ID, 'predict']
    print('====shape df_test====', df_test.shape)
    df_test.to_csv(submit_path, index=False)


if __name__ == '__main__':
    df_train, df_test = base_data_process.eda(age2group=True, one_hot=True)
    data_prepare(df_train, df_test)
