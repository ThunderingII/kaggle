import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.utils.vis_utils import plot_model
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

import plotly
from sklearn.preprocessing import Imputer


def model():
    # save filepath to variable for easier access
    data_file_path = '../data/melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = pd.read_csv(data_file_path)
    # print a summary of the data in Melbourne data
    # print(melbourne_data.describe())
    melbourne_data.dropna(inplace=True)
    # print(melbourne_data.describe())
    # print(melbourne_data.columns)

    melbourne_features = ['Rooms', 'Distance', 'Bathroom', 'Landsize', 'BuildingArea']
    X = melbourne_data[melbourne_features]
    Y = melbourne_data['Price']

    x_train = X.iloc[:4190]
    x_val = X.iloc[4190:5190]
    x_test = X.iloc[5190]

    y_train = Y.iloc[:4190]
    y_val = Y.iloc[4190:5190]
    y_test = Y.iloc[5190]

    new_data = x_train.copy()
    # make new columns indicating what will be imputed
    cols_with_missing = (col for col in new_data.columns
                         if new_data[col].isnull().any())

    for col in cols_with_missing:
        print(col)
        new_data[col + '_was_missing'] = new_data[col].isnull()
    # Imputation
    my_imputer = Imputer()
    new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
    new_data.columns = x_train.columns


    print(new_data.dtypes)
    # print(new_data)

    print(new_data.describe())
    print(x_train.describe())

    # nn_model(x_train, x_val, y_train, y_val)

    model = DecisionTreeRegressor(random_state=1)
    model.fit(x_train, y_train)
    y_pre = model.predict(x_val)
    print(mean_absolute_error(y_val, y_pre))

    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(x_train, y_train)
    y_pre_f = forest_model.predict(x_val)
    print(mean_absolute_error(y_val, y_pre_f))


def nn_model(x_train, x_val, y_train, y_val):
    nnmodel = Sequential()
    nnmodel.add(Dense(units=3, input_dim=5, kernel_initializer='normal', activation='relu'))
    nnmodel.add(Dense(units=1, kernel_initializer='normal'))
    nnmodel.compile(loss='mse', optimizer='adam')
    nnmodel.fit(x_train.values, y_train.values, 32, 5)
    y_pre_nn = nnmodel.predict(x_val.values)
    plotly.offline.plot({
        "data": [go.Scatter(x=y_val.values, y=y_pre_nn.reshape(-1), mode='markers'),
                 go.Scatter(x=[0, 8000000], y=[0, 8000000])],
        "layout": go.Layout(title="y_val y_pre_nn")
    })
    print(mean_absolute_error(y_val.values, y_pre_nn))
    print(y_pre_nn.reshape(-1))


if __name__ == '__main__':
    model()
