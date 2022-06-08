'''
importing required modules 
'''

import json
import requests
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow

''' 
Collecting the dataset required for training the model from
crypto-compare.com using their api

'''
api_url = "https://min-api.cryptocompare.com/data/histoday"
coin = "BTC"
conversion = "USD"

response = requests.get(api_url+'?fsym='+coin+'&tsym='+conversion+'&limit=500')
my_dataset = pd.DataFrame(json.loads(response.content)['Data'])
my_dataset = my_dataset.set_index('time')
my_dataset.index = pd.to_datetime(my_dataset.index, yearfirst=True, unit='s')
target_column = 'close'

# droping the two columns << "ConversionType" and "ConversionSymbol" from the my_dataset
my_dataset.drop(["conversionType", "conversionSymbol"],
                axis='columns', inplace=True)

# getting last 5 data from the my_dataset
my_dataset.tail(5)


''' 
splitting up the data for training and testing the model 

training_set : used to fit the model 
testing_set : with testing_set, we can evaluate our model performance

-------------    --------
Training Set      80 %
Testing Set       20 %
------------     --------
'''


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


train, test = train_test_split(my_dataset, test_size=0.2)


'''
plotting our splitted data of training set and testing set in a graph

'''


def plot_line(line1, line2, label1=None, label2=None, title='', linewidth=2):
    fig, axis = plt.subplots(1, figsize=(13, 7))
    axis.plot(line1, label=label1, linewidth=linewidth)
    axis.plot(line2, label=label2, linewidth=linewidth)
    axis.set_ylabel('price [USD]', fontsize=14)
    axis.set_xlabel('Time Period', fontsize=14)
    axis.set_title(title, fontsize=16)
    axis.legend(loc='best', fontsize=16)


# plot_line(train[target_column], test[target_column], 'training', 'testing')

'''
Normalising our data 

Normalization is a data pre-processing tool used to bring the 
numerical data to a common scale without distorting its shape. 
Generally, when we input the data to a machine or deep learning
algorithm we tend to change the values to a balanced scale.

---------------------------------
Here we use two normalization technique 
1. zero score normalization
2. Min Max normalization

'''

# zero score normalization


def zero_score_normalization(df):
    return df / df.iloc[0] - 1

# Min max normalization


def min_max_normalization(df):
    return (df - df.min()) / (df.max() - df.min())


# -------------------------------------------------------
'''
now , we have to extract the small sequence of data ( say window ) from the whole dataset 
that we will fed into the model ,  inorder to fine tuning the model

'''


def extract_window_data(df, window_size=5, zero_score=True):
    # initialize an empty of windowed data
    # list
    window_data = []
    # initialize a for loop with the range of (0 , lenght(df) - window_size)
    for i in range(len(df) - window_size):
        # store the 5 values of the data frame in `temp` for every iteration
        temp = df[i: (i + window_size)].copy()
        if zero_score:  # if the normalization methoad is zero_score(TRUE)
            temp = zero_score_normalization(temp)
        # append the normalized values into the `window_data` list we initialised before
        window_data.append(temp.values)
    return np.array(window_data)  # convert the `window_data` into numpy array


'''
now we are done with pre-processing our data and then we will prepare our data
for fed into the model

'''


def prepare_data(df, target_column, window_len=10, zero_score=True, test_size=0.2):
    # spliting data into training and testing
    train_data, test_data = train_test_split(df, test_size=test_size)

    # extracting X train
    X_train = extract_window_data(train_data, window_len, zero_score)
    X_test = extract_window_data(test_data, window_len, zero_score)

    # getting `close` column values with window length
    y_train = train_data[target_column][window_len:].values
    y_test = test_data[target_column][window_len:].values

    # if zero score normalization is True
    if zero_score:
        y_train = y_train / train_data[target_column][:-window_len].values - 1
        y_test = y_test / test_data[target_column][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


#  Now , we are done with pre-processing and preparing the data for feed into the model .
#  the next step and important step is to build the model


# we are going to use LSTM neural network for this project


''' 
Long short-term memory (LSTM) is an artificial neural network used in the 
fields of artificial intelligence and deep learning. it is classified under the
`Recurrent Neural Network` which is capable of processing the sequences of data(speech , audio , video).

'''


def build_lstm_model(input_data, output_size, neurons=100, activation_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    the_model = Sequential()
    the_model.add(LSTM(neurons, input_shape=(
        input_data.shape[1], input_data.shape[2])))
    the_model.add(Dropout(dropout))
    the_model.add(Dense(units=output_size))
    the_model.add(Activation(activation_func))
    the_model.compile(loss=loss, optimizer=optimizer)

    return the_model


# configuration for our LSTM model
np.random.seed(42)
window_len = 5
test_size = 0.2
zero_score = True
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.2
optimizer = 'adam'


train, test, X_train, X_test, y_train, y_test = prepare_data(
    my_dataset, target_column, window_len=window_len, zero_score=zero_score, test_size=test_size)


model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)


plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
plt.plot(history.history['val_loss'], 'g',
         linewidth=2, label='Validation loss')
plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()


targets = test[target_column][window_len:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)


MAE = mean_squared_error(preds, y_test)


R2 = r2_score(y_test, preds)


preds = test[target_column].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
plot_line(targets, preds, 'actual', 'prediction', linewidth=3)

plt.savefig('output.png')
