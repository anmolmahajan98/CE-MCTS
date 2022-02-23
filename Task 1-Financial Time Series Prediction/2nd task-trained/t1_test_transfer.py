from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
from keras import optimizers
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import sys
import re
import pickle
import random
import scipy.stats
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#load data
with open("month_wise_expenses_unique_v2.txt", "rb") as fp:
    month_wise_expenses_unique_v2 = pickle.load(fp)

with open("all_rl_labels_v2.txt", "rb") as fp:
    all_rl_labels_v2 = pickle.load(fp)

with open("month_wise_expenses_unique.txt", "rb") as fp:
    month_wise_expenses_unique = pickle.load(fp)

with open("all_rl_labels.txt", "rb") as fp:
    all_rl_labels = pickle.load(fp)

all_mse_avg = 0
all_sd = 0
for index in range(0, 5):
    training_set = []
    test_set = []

    training_set_v2 = []
    test_set_v2 = []

    train_labels = []
    test_labels = []

    train_labels_v2 = []
    test_labels_v2 = []

    folds = []
    folds_labels = []

    folds_v2 = []
    folds_labels_v2 = []

    all_folds = month_wise_expenses_unique
    all_folds_labels = all_rl_labels

    all_folds_v2 = month_wise_expenses_unique_v2
    all_folds_labels_v2 = all_rl_labels_v2

    k = 5
    fold_size = int(len(all_rl_labels)/k)

    #created separate folds
    for i in range(k):
        particular_fold = []
        particular_fold_label = []
        particular_fold_v2 = []
        particular_fold_label_v2 = []

        for j in range(i*fold_size, i*fold_size + fold_size):
            particular_fold.append(all_folds[j])
            particular_fold_label.append(all_folds_labels[j])
            particular_fold_v2.append(all_folds_v2[j])
            particular_fold_label_v2.append(all_folds_labels_v2[j])

        folds.append(particular_fold)
        folds_labels.append(particular_fold_label)
        folds_v2.append(particular_fold_v2)
        folds_labels_v2.append(particular_fold_label_v2)

    #splitting into train-test folds
    for i in range(index, index+1):
        print("Test fold index:", i)

        test_set = folds[i]
        test_labels = folds_labels[i]
        test_set_v2 = folds_v2[i]
        test_labels_v2 = folds_labels_v2[i]

        for j in range(k):
            if j != i:
                training_set.extend(folds[j])
                train_labels.extend(folds_labels[j])
                training_set_v2.extend(folds_v2[j])
                train_labels_v2.extend(folds_labels_v2[j])

    X_train = np.array([[[z for z in y] for y in x] for x in training_set], ndmin=4)
    X_test = np.array([[[z for z in y] for y in x] for x in test_set], ndmin=4)
    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)

    X_train_v2 = np.array([[[z for z in y] for y in x] for x in training_set_v2], ndmin=4)
    X_test_v2 = np.array([[[z for z in y] for y in x] for x in test_set_v2], ndmin=4)
    Y_train_v2 = np.array(train_labels_v2)
    Y_test_v2 = np.array(test_labels_v2)

    X_train = array(X_train).reshape(2560, 31, 15)
    Y_train = array(Y_train).reshape(2560, 1)
    X_test = array(X_test).reshape(640, 31, 15)
    Y_test = array(Y_test).reshape(640, 1)

    X_train_v2 = array(X_train_v2).reshape(8960, 31, 15)
    Y_train_v2 = array(Y_train_v2).reshape(8960, 1)
    X_test_v2 = array(X_test_v2).reshape(2240, 31, 15)
    Y_test_v2 = array(Y_test_v2).reshape(2240, 1)

    print(X_train.shape)
    print(X_test.shape)
    print(X_train_v2.shape)
    print(X_test_v2.shape)

    #load model
    model = Sequential()
    model.add(LSTM(units = 512, return_sequences = True, input_shape = (31, 15)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 512, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 512, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 512))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.load_weights('cv_sttl_32_lstm_weights_'+str(index)+'.h5', by_name=True, skip_mismatch=True)
    model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])

    history = model.fit(X_test_v2, Y_test_v2, epochs = 10, validation_split=0.1, verbose=1, batch_size = 32)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print("MSE, MSE on test set:")
    print(scores)

    all_predicted_labels = []
    all_original_labels = []

    for i in range(0, 640, 10):
        predicted_labels = []
        original_labels = []
        for j in range(i, i+10):
            test_input = X_test[j]
            test_input = test_input.reshape(1, 31, 15)
            test_output = model.predict(test_input, verbose=0)
            #print("Output from LSTM:", test_output)
            predicted_labels.append(test_output[0][0])
            #print("Original_label:", Y_test_current[i])
            original_labels.append(Y_test[j][0])
        all_predicted_labels.append(predicted_labels)
        all_original_labels.append(original_labels)

    mse_vals = []
    mse = 0

    for i in range(0, len(all_original_labels)):
        mse_vals.append(mean_squared_error(all_original_labels[i], all_predicted_labels[i]))

    for i in range(0, len(mse_vals)):
        mse+=mse_vals[i]

    print(float(mse/64))
    mse_vals_numpy = np.array(mse_vals)
    print(np.std(mse_vals_numpy))

    all_mse_avg+=float(mse/64)
    all_sd+=np.std(mse_vals_numpy)

print(all_mse_avg/5)
print(all_sd/5)