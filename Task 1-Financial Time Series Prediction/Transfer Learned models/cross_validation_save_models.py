from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
from keras import optimizers

import pandas as pd
import numpy as np
import re
import pickle

import matplotlib.pyplot as plt

with open("month_wise_expenses_unique.txt", "rb") as fp:
    month_wise_expenses_unique = pickle.load(fp)

with open("all_rl_labels.txt", "rb") as fp:
    all_rl_labels = pickle.load(fp)

folds = []
folds_labels = []

all_folds = month_wise_expenses_unique
all_folds_labels = all_rl_labels

k = 5
fold_size = int(len(all_rl_labels)/k)

for i in range(k):
    particular_fold = []
    particular_fold_label = []
    #particular_fold_v2 = []
    #particular_fold_label_v2 = []

    for j in range(i*fold_size, i*fold_size + fold_size):
        particular_fold.append(all_folds[j])
        particular_fold_label.append(all_folds_labels[j])
        #particular_fold_v2.append(all_folds_v2[j])
        #particular_fold_label_v2.append(all_folds_labels_v2[j])

    folds.append(particular_fold)
    folds_labels.append(particular_fold_label)
    #folds_v2.append(particular_fold_v2)
    #folds_labels_v2.append(particular_fold_label_v2)

#splitting into train-test folds
iterator_val = 4
for i in range(4, k):
    print("Test fold index:", iterator_val)

    training_set = []
    test_set = []

    #training_set_v2 = []
    #test_set_v2 = []

    train_labels = []
    test_labels = []

    #train_labels_v2 = []
    #test_labels_v2 = []

    test_set = folds[i]
    test_labels = folds_labels[i]
    #test_set_v2 = folds_v2[i]
    #test_labels_v2 = folds_labels_v2[i]

    for j in range(0, k):
        if j != i:
            print("Value of j:", j)
            training_set.extend(folds[j])
            train_labels.extend(folds_labels[j])
            #training_set_v2.extend(folds_v2[j])
            #train_labels_v2.extend(folds_labels_v2[j])

    X_train = np.array([[[z for z in y] for y in x] for x in training_set], ndmin=4)
    X_test = np.array([[[z for z in y] for y in x] for x in test_set], ndmin=4)
    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)

    #X_train_v2 = np.array([[[z for z in y] for y in x] for x in training_set_v2], ndmin=4)
    #X_test_v2 = np.array([[[z for z in y] for y in x] for x in test_set_v2], ndmin=4)
    #Y_train_v2 = np.array(train_labels_v2)
    #Y_test_v2 = np.array(test_labels_v2)

    X_train = array(X_train).reshape(2560, 31, 15)
    Y_train = array(Y_train).reshape(2560, 1)
    X_test = array(X_test).reshape(640, 31, 15)
    Y_test = array(Y_test).reshape(640, 1)

    #X_train_v2 = array(X_train_v2).reshape(8960, 31, 15)
    #Y_train_v2 = array(Y_train_v2).reshape(8960, 1)
    #X_test_v2 = array(X_test_v2).reshape(2240, 31, 15)
    #Y_test_v2 = array(Y_test_v2).reshape(2240, 1)

    print(X_train.shape)
    print(X_test.shape)
    #print(X_train_v2.shape)
    #print(X_test_v2.shape)

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

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = adam, loss = 'mean_squared_error')

    model.load_weights('hdfc_regressor_lstm_weights_v3.h5', by_name=True, skip_mismatch=True)
    model.summary()
    history = model.fit(X_train, Y_train, epochs = 30, validation_split=0.1, verbose=1, batch_size = 32)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print("MSE, MSE on test set:")
    print(scores)

    predicted_labels = []
    original_labels = []

    for j in range(0, 640):
        test_input = X_test[j]
        test_input = test_input.reshape(1, 31, 15)
        test_output = model.predict(test_input, verbose=0)
        print("Output from LSTM:", test_output)
        predicted_labels.append(test_output[0][0])
        print("Original_label:", Y_test[j])
        original_labels.append(Y_test[j][0])

    predicted_values = []
    original_values = []
    total_matches = []
    total_mismatches = []
    discount_factor = 0.5

    #reverseRL
    for a in range(len(predicted_labels)-1, -1, -10):
        month_wise_predicted_vals = [0]*10
        iterate = 9
        for j in range(a, a-10, -1):
            if a == j:
                current_value = predicted_labels[j]
                previous_value = predicted_labels[j]
                month_wise_predicted_vals[iterate] = current_value
            else:
                current_value = predicted_labels[j] - discount_factor*(previous_value)
                previous_value = current_value
                month_wise_predicted_vals[iterate] = current_value
            iterate-=1

        for k in range(len(month_wise_predicted_vals)):
            current_element = month_wise_predicted_vals[k]
            current_element_rounded = round(current_element)
            month_wise_predicted_vals[k] = current_element_rounded
        predicted_values.append(month_wise_predicted_vals)

    for b in range(len(original_labels)-1, -1, -10):
        month_wise_predicted_vals = [0]*10
        iterate = 9
        for j in range(i, i-10, -1):
            if b == j:
                current_value = original_labels[j]
                previous_value = original_labels[j]
                month_wise_predicted_vals[iterate] = current_value
            else:
                current_value = original_labels[j] - discount_factor*(previous_value)
                previous_value = current_value
                month_wise_predicted_vals[iterate] = current_value
            iterate-=1

        for k in range(len(month_wise_predicted_vals)):
            current_element = month_wise_predicted_vals[k]
            current_element_rounded = int(round(current_element))
            month_wise_predicted_vals[k] = current_element_rounded
        original_values.append(month_wise_predicted_vals)

    for c in range(len(predicted_values)):
        mismatch_count = 0
        match_count = 0
        for j in range(len(predicted_values[i])):
            if predicted_values[c][j] != original_values[c][j]:
                mismatch_count+=1
            else:
                match_count+=1
        total_matches.append(match_count)
        total_mismatches.append(mismatch_count)

    print(total_matches)
    print(total_mismatches)

    filename = 'cv_sttl_32_lstm_weights_' + str(iterator_val) + '.h5'
    model.save_weights(filename)
    iterator_val+=1
