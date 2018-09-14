import pandas as pd
import time
from contextlib import contextmanager
import numpy as np
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from util.base_util import timer


def data_prepare():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

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
    # process nan
    category_list = ['gender', 'service_type', 'is_mix_service', 'many_over_bill', 'contract_type',
                     'is_promise_low_consume', 'net_service', 'complaint_level']

    dummies(df_train, category_list)
    dummies(df_test, category_list)

    label = df_train['current_service']

    df_train['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']
    df_test['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']

    df_train['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)
    df_test['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)

    df_train['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']
    df_test['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']

    print('before align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    print('after align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train['current_service'] = label

    label_encode(df_train, 'current_service')

    return df_train, df_test


encode_map = {}
decode_list = None


# get label index from label's value
def label_encode(df, label_name):
    global decode_list
    decode_list = list(df[label_name].value_counts().index)

    label_size = len(decode_list)

    for i in range(label_size):
        encode_map[decode_list[i]] = i

    df[label_name] = df[label_name].apply(lambda x: encode_map[x])

    print(df[label_name].value_counts())


def label_decode(df, label_name):
    df[label_name] = df[label_name].apply(lambda x: decode_list[x])
    return df[label_name]


def model(train, test, num_folds=5, stratified=True, num_boost_round=1000):
    global decode_list
    # Divide in training/validation and test data
    ID_COLUMN_NAME = 'user_id'
    LABEL_COLUMN_NAME = 'current_service'
    LABEL_SIZE = train[LABEL_COLUMN_NAME].value_counts().count()

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))

    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    sub_preds = np.zeros(shape=(test.shape[0], LABEL_SIZE))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train.columns if
             f not in [LABEL_COLUMN_NAME, ID_COLUMN_NAME]]
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train[LABEL_COLUMN_NAME])):
        dtrain = lgb.Dataset(data=train[feats].iloc[train_idx],
                             label=train[LABEL_COLUMN_NAME].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train[feats].iloc[valid_idx],
                             label=train[LABEL_COLUMN_NAME].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        # LightGBM parameters found by Bayesian optimization
        params = {
            'objective': 'multiclass',
            'num_leaves': 31,
            'seed': 0,
            'num_threads': 20,
            'learning_rate': 0.02,
            'num_class': LABEL_SIZE
        }
        with timer('train model'):
            clf = lgb.train(
                num_boost_round=num_boost_round,
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                early_stopping_rounds=100
            )
        with timer('fold {} predict'.format(i_fold)):
            v_data = clf.predict(dvalid.data)
            y_pre = []
            for d in v_data:
                max = d[0]
                max_i = 0
                for i in range(1, 15):
                    if d[i] > max:
                        max = d[i]
                        max_i = i
                y_pre.append(max_i)

            sub_preds += clf.predict(test[feats])

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold {} f1 : {}'.format(i_fold + 1, f1_score(dvalid.label, y_pre, average='macro')))
        del clf, dtrain, dvalid
        gc.collect()

    y_pre = []
    for d in sub_preds:
        max = d[0]
        max_i = 0
        for i in range(1, 15):
            if d[i] > max:
                max = d[i]
                max_i = i
        y_pre.append(decode_list[max_i])

    # Write submission file and plot feature importance
    sub_df = test[[ID_COLUMN_NAME]].copy()
    sub_df['predict'] = y_pre
    with timer('write result'):
        sub_df[[ID_COLUMN_NAME, 'predict']].to_csv('result.csv', index=False)
    display_importances(feature_importance_df)


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


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
            d = pd.get_dummies(df[c])
            new_d_cols = list(map(lambda x: c + '_' + str(x), d.columns))
            print('{} has divide into:\n\t\t\t\t{}'.format(c, new_d_cols))
            d.columns = new_d_cols
            for nc in new_d_cols:
                df[nc] = d[nc]
            df.drop(columns=[c], inplace=True)


if __name__ == '__main__':
    with timer('data process'):
        df_train, df_test = data_prepare()
    with timer('model process'):
        model(df_train, df_test, num_folds=5, num_boost_round=10000)
