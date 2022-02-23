#Version 1 -- Evaluation function based on matches of no-goal test Data
#Version 2 -- Evaluation based on both no-foal test data and goal-trainind data

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

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

individual_id = int(sys.argv[1])

X_test_current_start_index = individual_id*10
X_test_current_end_index = X_test_current_start_index + 10
X_test_v2_current_start_index = individual_id*35
X_test_v2_current_end_index = X_test_v2_current_start_index + 35

print("Individual ID:", individual_id)

#load data
with open("./month_wise_expenses_unique_v2.txt", "rb") as fp:
    month_wise_expenses_unique_v2 = pickle.load(fp)

with open("./all_rl_labels_v2.txt", "rb") as fp:
    all_rl_labels_v2 = pickle.load(fp)

with open("./month_wise_expenses_unique.txt", "rb") as fp:
    month_wise_expenses_unique = pickle.load(fp)

with open("./all_rl_labels.txt", "rb") as fp:
    all_rl_labels = pickle.load(fp)

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
for i in range(0, 1):
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

#define change type
change_1 = 1
change_2 = 2
change_3 = 3
change_4 = 4

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
model.load_weights('./cv_sttl_32_lstm_weights_0.h5', by_name=True, skip_mismatch=True)
model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=['accuracy'])

#deifne weights
counter = 0
for counter in range(0, 9):
    current_layer = model.layers[counter].get_weights()
    if counter == 0:
        lstm_cell_kernel = current_layer[0]
        lstm_cell_recurrent_kernel = current_layer[1]
        lstm_cell_bias = current_layer[2]
    elif counter == 2:
        lstm_cell_1_kernel = current_layer[0]
        lstm_cell_1_recurrent_kernel = current_layer[1]
        lstm_cell_1_bias = current_layer[2]
    elif counter == 4:
        lstm_cell_2_kernel = current_layer[0]
        lstm_cell_2_recurrent_kernel = current_layer[1]
        lstm_cell_2_bias = current_layer[2]
    elif counter == 6:
        lstm_cell_3_kernel = current_layer[0]
        lstm_cell_3_recurrent_kernel = current_layer[1]
        lstm_cell_3_bias = current_layer[2]
    elif counter == 8:
        dense_kernel = current_layer[0]
        dense_bias = current_layer[1]

#define alpha values
lstm_cell_kernel_alpha = np.ones((15, 2048))
lstm_cell_recurrent_kernel_alpha = np.ones((512, 2048))
lstm_cell_bias_alpha = np.ones((2048,))
lstm_cell_1_kernel_alpha = np.ones((512, 2048))
lstm_cell_1_recurrent_kernel_alpha = np.ones((512, 2048))
lstm_cell_1_bias_alpha = np.ones((2048,))
lstm_cell_2_kernel_alpha = np.ones((512, 2048))
lstm_cell_2_recurrent_kernel_alpha = np.ones((512, 2048))
lstm_cell_2_bias_alpha = np.ones((2048,))
lstm_cell_3_kernel_alpha = np.ones((512, 2048))
lstm_cell_3_recurrent_kernel_alpha = np.ones((512, 2048))
lstm_cell_3_bias_alpha = np.ones((2048,))
dense_kernel_alpha = np.ones((512, 1))
dense_bias_alpha = np.ones((1, ))

#define layer identifiers
layers_identifiers = ['lstm_cell_kernel', 'lstm_cell_recurrent_kernel', 'lstm_cell_bias', 'lstm_cell_1_kernel', 'lstm_cell_1_recurrent_kernel', 'lstm_cell_1_bias', 'lstm_cell_2_kernel', 'lstm_cell_2_recurrent_kernel', 'lstm_cell_2_bias', 'lstm_cell_3_kernel', 'lstm_cell_3_recurrent_kernel', 'lstm_cell_3_bias', 'dense_kernel', 'dense_bias']

#define changes
def change1(change_track):
    #[identifier_change, change_type, position_of_change, random_alpha_change,  alpha, weights]
    layer_index = random.randint(1, 13)
    if layer_index == 1:
        rand_index_x = random.randint(0, 14)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_kernel_alpha.tolist()
        weights_to_use = lstm_cell_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 2:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_recurrent_kernel_alpha.tolist()
        weights_to_use = lstm_cell_recurrent_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 3:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_bias_alpha.tolist()
        weights_to_use = lstm_cell_bias.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 4:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_kernel_alpha.tolist()
        weights_to_use = lstm_cell_1_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 5:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_recurrent_kernel_alpha.tolist()
        weights_to_use = lstm_cell_1_recurrent_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 6:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_bias_alpha.tolist()
        weights_to_use = lstm_cell_1_bias.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 7:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_kernel_alpha.tolist()
        weights_to_use = lstm_cell_2_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 8:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_recurrent_kernel_alpha.tolist()
        weights_to_use = lstm_cell_2_recurrent_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 9:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_bias_alpha.tolist()
        weights_to_use = lstm_cell_2_bias.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 10:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_kernel_alpha.tolist()
        weights_to_use = lstm_cell_3_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 11:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_recurrent_kernel_alpha.tolist()
        weights_to_use = lstm_cell_3_recurrent_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 12:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_bias_alpha.tolist()
        weights_to_use =lstm_cell_3_bias.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 13:
        rand_index = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = dense_kernel_alpha.tolist()
        weights_to_use = dense_kernel.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_1)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        list_to_append.append(weights_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

#multiply alpha by random int(-2, 2)
def change2(change_track):
    #[identifier, change_type, position_of_change, random_alpha_change, alpha]
    layer_index = random.randint(1, 13)
    if layer_index == 1:
        rand_index_x = random.randint(0, 14)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 2:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_recurrent_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 3:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_bias_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 4:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 5:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_recurrent_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 6:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_1_bias_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 7:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 8:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_recurrent_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 9:
        rand_index = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_2_bias_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 10:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 11:
        rand_index_x = random.randint(0, 511)
        rand_index_y = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index_x, rand_index_y]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_recurrent_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 12:
        rand_index = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = lstm_cell_3_bias_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 13:
        rand_index = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index]
        random_alpha_change = [random.randint(-2, 2)]
        alphas_to_use = dense_kernel_alpha.tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_2)
        list_to_append.append(position_of_change)
        list_to_append.append(random_alpha_change)
        list_to_append.append(alphas_to_use)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

#swapping of two random layers
def change3(change_track):
    #[identifier_change, change_type, position_of_change, identifier_alpha, identifier_modify, identifier_alpha_swap, identifier_modify_swap]
    random_list = [2, 3, 4, 7, 10, 5, 8, 11, 6, 9, 12, 13]
    layer_index = random.choice(random_list)
    if layer_index == 2 or layer_index == 5 or layer_index == 8 or layer_index == 11:
        random_list_original = [2, 5, 8, 11]
        layer_index_swap = random.choice(random_list_original)
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        if layer_index == 2:
            identifier_alpha = lstm_cell_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_recurrent_kernel[rand_index][:].tolist()
        if layer_index == 5:
            identifier_alpha = lstm_cell_1_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_1_recurrent_kernel[rand_index][:].tolist()
        if layer_index == 8:
            identifier_alpha = lstm_cell_2_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_2_recurrent_kernel[rand_index][:].tolist()
        if layer_index == 11:
            identifier_alpha = lstm_cell_3_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_3_recurrent_kernel[rand_index][:].tolist()
        if layer_index_swap == 2:
            identifier_alpha_swap = lstm_cell_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index_swap == 5:
            identifier_alpha_swap = lstm_cell_1_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_1_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index_swap == 8:
            identifier_alpha_swap = lstm_cell_2_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_2_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index_swap == 11:
            identifier_alpha_swap = lstm_cell_3_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_3_recurrent_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        list_to_append.append([layer_index-1, layer_index_swap-1])
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 4 or layer_index == 7 or layer_index == 10:
        random_list_original = [4, 7, 10]
        layer_index_swap = random.choice(random_list_original)
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        if layer_index == 4:
            identifier_alpha = lstm_cell_1_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_1_kernel[rand_index][:].tolist()
        if layer_index == 7:
            identifier_alpha = lstm_cell_2_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_2_kernel[rand_index][:].tolist()
        if layer_index == 10:
            identifier_alpha = lstm_cell_3_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_3_kernel[rand_index][:].tolist()
        if layer_index_swap == 4:
            identifier_alpha_swap = lstm_cell_1_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_1_kernel[rand_index_swap][:].tolist()
        if layer_index_swap == 7:
            identifier_alpha_swap = lstm_cell_2_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_2_kernel[rand_index_swap][:].tolist()
        if layer_index_swap == 10:
            identifier_alpha_swap = lstm_cell_3_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_3_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        list_to_append.append([layer_index-1, layer_index_swap-1])
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 3:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 6:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_1_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_1_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_1_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_1_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 9:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_2_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_2_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_2_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_2_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 12:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_3_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_3_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_3_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_3_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 13:
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = dense_kernel_alpha[rand_index][:].tolist()
        identifier_modify = dense_kernel[rand_index][:].tolist()
        identifier_alpha_swap = dense_kernel_alpha[rand_index_swap][:].tolist()
        identifier_modify_swap = dense_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_3)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

#product of alpha and weights added to product of some other random alphas and weights of other layer
def change4(change_track):
    #[identifier_change, change_type, position_of_change, identifier_alpha, identifier_modify, identifier_alpha_swap, identifier_modify_swap]
    random_list = [2, 3, 4, 7, 10, 5, 8, 11, 6, 9, 12, 13]
    #random_list = [3, 6, 9, 12, 13]
    layer_index = random.choice(random_list)
    if layer_index == 2 or layer_index == 5 or layer_index == 8 or layer_index == 11:
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        if layer_index == 2:
            identifier_alpha = lstm_cell_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_recurrent_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index == 5:
            identifier_alpha = lstm_cell_1_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_1_recurrent_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_1_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_1_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index == 8:
            identifier_alpha = lstm_cell_2_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_2_recurrent_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_2_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_2_recurrent_kernel[rand_index_swap][:].tolist()
        if layer_index == 11:
            identifier_alpha = lstm_cell_3_recurrent_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_3_recurrent_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_3_recurrent_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_3_recurrent_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        list_to_append.append([layer_index-1])
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 4 or layer_index == 7 or layer_index == 10:
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        if layer_index == 4:
            identifier_alpha = lstm_cell_1_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_1_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_1_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_1_kernel[rand_index_swap][:].tolist()
        if layer_index == 7:
            identifier_alpha = lstm_cell_2_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_2_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_2_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_2_kernel[rand_index_swap][:].tolist()
        if layer_index == 10:
            identifier_alpha = lstm_cell_3_kernel_alpha[rand_index][:].tolist()
            identifier_modify = lstm_cell_3_kernel[rand_index][:].tolist()
            identifier_alpha_swap = lstm_cell_3_kernel_alpha[rand_index_swap][:].tolist()
            identifier_modify_swap = lstm_cell_3_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        list_to_append.append([layer_index-1])
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 3:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 6:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_1_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_1_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_1_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_1_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 9:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_2_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_2_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_2_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_2_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 12:
        rand_index = random.randint(0, 2047)
        rand_index_swap = random.randint(0, 2047)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = lstm_cell_3_bias_alpha[rand_index].tolist()
        identifier_modify = lstm_cell_3_bias[rand_index].tolist()
        identifier_alpha_swap = lstm_cell_3_bias_alpha[rand_index_swap].tolist()
        identifier_modify_swap = lstm_cell_3_bias[rand_index_swap].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

    if layer_index == 13:
        rand_index = random.randint(0, 511)
        rand_index_swap = random.randint(0, 511)
        list_to_append = []
        identifier_change = layers_identifiers[layer_index-1]
        position_of_change = [rand_index, rand_index_swap]
        identifier_alpha = dense_kernel_alpha[rand_index][:].tolist()
        identifier_modify = dense_kernel[rand_index][:].tolist()
        identifier_alpha_swap = dense_kernel_alpha[rand_index_swap][:].tolist()
        identifier_modify_swap = dense_kernel[rand_index_swap][:].tolist()
        list_to_append.append(identifier_change)
        list_to_append.append(change_4)
        list_to_append.append(position_of_change)
        list_to_append.append(identifier_alpha)
        list_to_append.append(identifier_modify)
        list_to_append.append(identifier_alpha_swap)
        list_to_append.append(identifier_modify_swap)
        change_track.append(list_to_append)
        change_track_total.append(list_to_append)

#define calculate best matches
def calculate_best_matches(total_matches, total_mismatches, total_matches_v2, total_mismatches_v2, matches_to_beat, mse_to_maintain, mse_matches_train):
    min_mse = min(mse_matches_train)
    best_total_matches_v2 = total_matches_v2[mse_matches_train.index(min_mse)]

    if best_total_matches_v2 >= matches_to_beat and min_mse <=mse_to_maintain:
        return best_total_matches_v2, min_mse
    else:
        return matches_to_beat, mse_to_maintain

#define calculate new weights
def calculate_new_weights(change_track, current_best_matches_index, matches_to_beat, new_weights, new_alphas):
    item = change_track[current_best_matches_index]
    lstm_cell_kernel_to_use = np.array(new_weights[0], dtype = 'float32')
    lstm_cell_recurrent_kernel_to_use = np.array(new_weights[1], dtype = 'float32')
    lstm_cell_bias_to_use = np.array(new_weights[2], dtype = 'float32')
    lstm_cell_1_kernel_to_use = np.array(new_weights[3], dtype = 'float32')
    lstm_cell_1_recurrent_kernel_to_use = np.array(new_weights[4], dtype = 'float32')
    lstm_cell_1_bias_to_use = np.array(new_weights[5], dtype = 'float32')
    lstm_cell_2_kernel_to_use = np.array(new_weights[6], dtype = 'float32')
    lstm_cell_2_recurrent_kernel_to_use = np.array(new_weights[7], dtype = 'float32')
    lstm_cell_2_bias_to_use = np.array(new_weights[8], dtype = 'float32')
    lstm_cell_3_kernel_to_use = np.array(new_weights[9], dtype = 'float32')
    lstm_cell_3_recurrent_kernel_to_use = np.array(new_weights[10], dtype = 'float32')
    lstm_cell_3_bias_to_use = np.array(new_weights[11], dtype = 'float32')
    dense_kernel_to_use = np.array(new_weights[12], dtype = 'float32')

    lstm_cell_kernel_alpha = np.ones((15, 2048))
    lstm_cell_recurrent_kernel_alpha = np.ones((512, 2048))
    lstm_cell_bias_alpha = np.ones((2048,))
    lstm_cell_1_kernel_alpha = np.ones((512, 2048))
    lstm_cell_1_recurrent_kernel_alpha = np.ones((512, 2048))
    lstm_cell_1_bias_alpha = np.ones((2048,))
    lstm_cell_2_kernel_alpha = np.ones((512, 2048))
    lstm_cell_2_recurrent_kernel_alpha = np.ones((512, 2048))
    lstm_cell_2_bias_alpha = np.ones((2048,))
    lstm_cell_3_kernel_alpha = np.ones((512, 2048))
    lstm_cell_3_recurrent_kernel_alpha = np.ones((512, 2048))
    lstm_cell_3_bias_alpha = np.ones((2048,))
    dense_kernel_alpha = np.ones((512, 1))
    dense_bias_alpha = np.ones((1, ))
    if item[1] == 1:
        if item[0] == "lstm_cell_kernel":
            lstm_cell_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_kernel_to_use = lstm_cell_kernel_alpha*(lstm_cell_kernel_to_use)

        elif item[0] == "lstm_cell_recurrent_kernel":
            lstm_cell_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_bias":
            lstm_cell_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_bias_alpha[item[2][0]] = item[3][0]
            lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

        elif item[0] == "lstm_cell_1_kernel":
            lstm_cell_1_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)

        elif item[0] == "lstm_cell_1_recurrent_kernel":
            lstm_cell_1_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_1_bias":
            lstm_cell_1_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_bias_alpha[item[2][0]] = item[3][0]
            lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

        elif item[0] == "lstm_cell_2_kernel":
            lstm_cell_2_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)

        elif item[0] == "lstm_cell_2_recurrent_kernel":
            lstm_cell_2_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_2_bias":
            lstm_cell_2_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_bias_alpha[item[2][0]] = item[3][0]
            lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

        elif item[0] == "lstm_cell_3_kernel":
            lstm_cell_3_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

        elif item[0] == "lstm_cell_3_recurrent_kernel":
            lstm_cell_3_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
            lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_3_bias":
            lstm_cell_3_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_bias_alpha[item[2][0]] = item[3][0]
            lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

        elif item[0] == "dense_kernel":
            dense_kernel_alpha = np.array(item[4], dtype = 'float32')
            dense_kernel_alpha[item[2][0]] = item[3][0]
            dense_kernel_to_use = dense_kernel_alpha*dense_kernel_to_use

    elif item[1] == 2:
        if item[0] == "lstm_cell_kernel":
            lstm_cell_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_kernel_alpha[item[2][0]][:])
            lstm_cell_kernel_to_use = lstm_cell_kernel_alpha*(lstm_cell_kernel_to_use)

        elif item[0] == "lstm_cell_recurrent_kernel":
            lstm_cell_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_recurrent_kernel_alpha[item[2][0]][:])
            lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_bias":
            lstm_cell_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_bias_alpha[:] = item[3][0]*(lstm_cell_bias_alpha[:])
            lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

        elif item[0] == "lstm_cell_1_kernel":
            lstm_cell_1_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_1_kernel_alpha[item[2][0]][:])
            lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)

        elif item[0] == "lstm_cell_1_recurrent_kernel":
            lstm_cell_1_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:])
            lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_1_bias":
            lstm_cell_1_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_1_bias_alpha[:] = item[3][0]*(lstm_cell_1_bias_alpha[:])
            lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

        elif item[0] == "lstm_cell_2_kernel":
            lstm_cell_2_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_2_kernel_alpha[item[2][0]][:])
            lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)

        elif item[0] == "lstm_cell_2_recurrent_kernel":
            lstm_cell_2_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:])
            lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_2_bias":
            lstm_cell_2_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_2_bias_alpha[:] = item[3][0]*(lstm_cell_2_bias_alpha[:])
            lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

        elif item[0] == "lstm_cell_3_kernel":
            lstm_cell_3_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_3_kernel_alpha[item[2][0]][:])
            lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

        elif item[0] == "lstm_cell_3_recurrent_kernel":
            lstm_cell_3_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:])
            lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_3_bias":
            lstm_cell_3_bias_alpha = np.array(item[4], dtype = 'float32')
            lstm_cell_3_bias_alpha[:] = item[3][0]*(lstm_cell_3_bias_alpha[:])
            lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

        elif item[0] == "dense_kernel":
            dense_kernel_alpha = np.array(item[4], dtype = 'float32')
            dense_kernel_alpha[:] = item[3][0]*(dense_kernel_alpha[:])
            dense_kernel_to_use = dense_kernel_alpha*dense_kernel_to_use

    elif item[1] == 3:
        if item[0] == "lstm_cell_recurrent_kernel" or item[0] == "lstm_cell_1_recurrent_kernel" or item[0] == "lstm_cell_2_recurrent_kernel" or item[0] == "lstm_cell_3_recurrent_kernel":
            alpha_to_swap = item[3]
            item_to_swap = item[4]
            alpha_with_swap = item[5]
            item_with_swap = item[6]
            layer_index = item[7][0]
            layer_index_swap = item[7][1]
            if layer_index == 1:
                lstm_cell_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
            if layer_index == 4:
                lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:]= alpha_with_swap
                lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
            if layer_index == 7:
                lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
            if layer_index == 10:
                lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)
            if layer_index_swap == 1:
                lstm_cell_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
            if layer_index_swap == 4:
                lstm_cell_1_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_1_recurrent_kernel_alpha[item[2][1]][:]= alpha_to_swap
                lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
            if layer_index_swap == 7:
                lstm_cell_2_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_2_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
            if layer_index_swap == 10:
                lstm_cell_3_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_3_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

        elif item[0] == "lstm_cell_1_kernel" or item[0] == "lstm_cell_2_kernel" or item[0] == "lstm_cell_3_kernel":
            alpha_to_swap = item[3]
            item_to_swap = item[4]
            alpha_with_swap = item[5]
            item_with_swap = item[6]
            layer_index = item[7][0]
            layer_index_swap = item[7][1]
            if layer_index == 3:
                lstm_cell_1_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_1_kernel_alpha[item[2][0]][:]= alpha_with_swap
                lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
            if layer_index == 6:
                lstm_cell_2_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_2_kernel_alpha[item[2][0]][:] = alpha_with_swap
                lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
            if layer_index == 9:
                lstm_cell_3_kernel_to_use[item[2][0]][:] = item_with_swap
                lstm_cell_3_kernel_alpha[item[2][0]][:] = alpha_with_swap
                lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)
            if layer_index_swap == 3:
                lstm_cell_1_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_1_kernel_alpha[item[2][1]][:]= alpha_to_swap
                lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
            if layer_index_swap == 6:
                lstm_cell_2_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_2_kernel_alpha[item[2][1]][:] = alpha_to_swap
                lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
            if layer_index_swap == 9:
                lstm_cell_3_kernel_to_use[item[2][1]][:] = item_to_swap
                lstm_cell_3_kernel_alpha[item[2][1]][:] = alpha_to_swap
                lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

        elif item[0] == "lstm_cell_bias":
            lstm_cell_bias_alpha_to_swap = item[3]
            lstm_cell_bias_to_swap = item[4]
            lstm_cell_bias_alpha_with_swap = item[5]
            lstm_cell_bias_with_swap = item[6]
            lstm_cell_bias_to_use[item[2][0]] = lstm_cell_bias_with_swap
            lstm_cell_bias_to_use[item[2][1]] = lstm_cell_bias_to_swap
            lstm_cell_bias_alpha[item[2][0]] = lstm_cell_bias_alpha_with_swap
            lstm_cell_bias_alpha[item[2][1]] = lstm_cell_bias_alpha_to_swap
            lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

        elif item[0] == "lstm_cell_1_bias":
            lstm_cell_1_bias_alpha_to_swap = item[3]
            lstm_cell_1_bias_to_swap = item[4]
            lstm_cell_1_bias_alpha_with_swap = item[5]
            lstm_cell_1_bias_with_swap = item[6]
            lstm_cell_1_bias_to_use[item[2][0]] = lstm_cell_1_bias_with_swap
            lstm_cell_1_bias_to_use[item[2][1]] = lstm_cell_1_bias_to_swap
            lstm_cell_1_bias_alpha[item[2][0]] = lstm_cell_1_bias_alpha_with_swap
            lstm_cell_1_bias_alpha[item[2][1]] = lstm_cell_1_bias_alpha_to_swap
            lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

        elif item[0] == "lstm_cell_2_bias":
            lstm_cell_2_bias_alpha_to_swap = item[3]
            lstm_cell_2_bias_to_swap = item[4]
            lstm_cell_2_bias_alpha_with_swap = item[5]
            lstm_cell_2_bias_with_swap = item[6]
            lstm_cell_2_bias_to_use[item[2][0]] = lstm_cell_2_bias_with_swap
            lstm_cell_2_bias_to_use[item[2][1]] = lstm_cell_2_bias_to_swap
            lstm_cell_2_bias_alpha[item[2][0]] = lstm_cell_2_bias_alpha_with_swap
            lstm_cell_2_bias_alpha[item[2][1]] = lstm_cell_2_bias_alpha_to_swap
            lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

        elif item[0] == "lstm_cell_3_bias":
            lstm_cell_3_bias_alpha_to_swap = item[3]
            lstm_cell_3_bias_to_swap = item[4]
            lstm_cell_3_bias_alpha_with_swap = item[5]
            lstm_cell_3_bias_with_swap = item[6]
            lstm_cell_3_bias_to_use[item[2][0]] = lstm_cell_3_bias_with_swap
            lstm_cell_3_bias_to_use[item[2][1]] = lstm_cell_3_bias_to_swap
            lstm_cell_3_bias_alpha[item[2][0]] = lstm_cell_3_bias_alpha_with_swap
            lstm_cell_3_bias_alpha[item[2][1]] = lstm_cell_3_bias_alpha_to_swap
            lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

        elif item[0] == "dense_kernel":
            dense_kernel_alpha_to_swap = item[3][0]
            dense_kernel_to_swap = item[4][0]
            dense_kernel_alpha_with_swap = item[5][0]
            dense_kernel_with_swap = item[6][0]
            dense_kernel_to_use[item[2][0]][:] = dense_kernel_with_swap
            dense_kernel_to_use[item[2][1]][:] = dense_kernel_to_swap
            dense_kernel_alpha[item[2][0]][:] = dense_kernel_alpha_with_swap
            dense_kernel_alpha[item[2][1]][:] = dense_kernel_alpha_to_swap
            dense_kernel_to_use = dense_kernel_alpha*(dense_kernel_to_use)

    elif item[1] == 4:
        if item[0] == "lstm_cell_recurrent_kernel" or item[0] == "lstm_cell_1_recurrent_kernel" or item[0] == "lstm_cell_2_recurrent_kernel" or item[0] == "lstm_cell_3_recurrent_kernel":
            layer_index = item[7][0]
            alpha_to_swap = item[3]
            item_to_swap = item[4]
            alpha_with_swap = item[5]
            item_with_swap = item[6]
            if layer_index == 1:
                lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
                lstm_cell_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_recurrent_kernel_to_use[item[2][1]][:]
            if layer_index == 4:
                lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
                lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_1_recurrent_kernel_to_use[item[2][1]][:]
            if layer_index == 7:
                lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
                lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_2_recurrent_kernel_to_use[item[2][1]][:]
            if layer_index == 10:
                lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)
                lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_3_recurrent_kernel_to_use[item[2][1]][:]

        elif item[0] == "lstm_cell_1_kernel" or item[0] == "lstm_cell_2_kernel" or item[0] == "lstm_cell_3_kernel":
            layer_index = item[7][0]
            alpha_to_swap = item[3]
            item_to_swap = item[4]
            alpha_with_swap = item[5]
            item_with_swap = item[6]
            if layer_index == 3:
                lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
                lstm_cell_1_kernel_to_use[item[2][0]][:] = lstm_cell_1_kernel_to_use[item[2][0]][:] + lstm_cell_1_kernel_to_use[item[2][1]][:]
            if layer_index == 6:
                lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
                lstm_cell_2_kernel_to_use[item[2][0]][:] = lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_2_kernel_to_use[item[2][1]][:]
            if layer_index == 9:
                lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)
                lstm_cell_3_kernel_to_use[item[2][0]][:] = lstm_cell_3_kernel_to_use[item[2][0]][:] + lstm_cell_3_kernel_to_use[item[2][1]][:]

        elif item[0] == "lstm_cell_bias":
            lstm_cell_bias_alpha_to_swap = item[3]
            lstm_cell_bias_to_swap = item[4]
            lstm_cell_bias_alpha_with_swap = item[5]
            lstm_cell_bias_with_swap = item[6]
            lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)
            lstm_cell_bias_to_use[item[2][0]] = lstm_cell_bias_to_use[item[2][0]] + lstm_cell_bias_to_use[item[2][1]]

        elif item[0] == "lstm_cell_1_bias":
            lstm_cell_1_bias_alpha_to_swap = item[3]
            lstm_cell_1_bias_to_swap = item[4]
            lstm_cell_1_bias_alpha_with_swap = item[5]
            lstm_cell_1_bias_with_swap = item[6]
            lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)
            lstm_cell_1_bias_to_use[item[2][0]] = lstm_cell_1_bias_to_use[item[2][0]] + lstm_cell_1_bias_to_use[item[2][1]]

        elif item[0] == "lstm_cell_2_bias":
            lstm_cell_2_bias_alpha_to_swap = item[3]
            lstm_cell_2_bias_to_swap = item[4]
            lstm_cell_2_bias_alpha_with_swap = item[5]
            lstm_cell_2_bias_with_swap = item[6]
            lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)
            lstm_cell_2_bias_to_use[item[2][0]] = lstm_cell_2_bias_to_use[item[2][0]] + lstm_cell_2_bias_to_use[item[2][1]]

        elif item[0] == "lstm_cell_3_bias":
            lstm_cell_3_bias_alpha_to_swap = item[3]
            lstm_cell_3_bias_to_swap = item[4]
            lstm_cell_3_bias_alpha_with_swap = item[5]
            lstm_cell_3_bias_with_swap = item[6]
            lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)
            lstm_cell_3_bias_to_use[item[2][0]] = lstm_cell_3_bias_to_use[item[2][0]] + lstm_cell_3_bias_to_use[item[2][1]]

        elif item[0] == "dense_kernel":
            dense_kernel_alpha_to_swap = item[3][0]
            dense_kernel_to_swap = item[4][0]
            dense_kernel_alpha_with_swap = item[5][0]
            dense_kernel_with_swap = item[6][0]
            dense_kernel_to_use = dense_kernel_alpha*(dense_kernel_to_use)
            dense_kernel_to_use[item[2][0]][:] = dense_kernel_to_use[item[2][0]][:] + dense_kernel_to_use[item[2][1]][:]

    new_weights_modified = []
    new_weights_modified.append(lstm_cell_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_recurrent_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_bias_to_use.tolist())
    new_weights_modified.append(lstm_cell_1_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_1_recurrent_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_1_bias_to_use.tolist())
    new_weights_modified.append(lstm_cell_2_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_2_recurrent_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_2_bias_to_use.tolist())
    new_weights_modified.append(lstm_cell_3_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_3_recurrent_kernel_to_use.tolist())
    new_weights_modified.append(lstm_cell_3_bias_to_use.tolist())
    new_weights_modified.append(dense_kernel_to_use.tolist())
    return new_weights_modified

#apprun()
change_track_total = []
all_matches = []
all_mismatches = []
all_matches_v2 = []
all_mismatches_v2 = []
all_original_labels = []
all_predicted_labels = []

def apprun(total_gen):
    change_track_matches = []
    change_track = []
    matches_to_beat = 0
    mse_to_maintain = 1
    for gen_index in range(total_gen):
        print("Generation Number:", gen_index)
        print("Initial length of changes:" ,len(change_track))
        print("")
        total_matches = []
        total_mismatches = []
        total_matches_v2 = []
        total_mismatches_v2 = []
        mse_matches_train = []
        for i in range(10):
            print("Variation Number:", i+1)
            random_change = random.randint(1, 4)
            if random_change == 1:
                change1(change_track)
                print("random index", random_change)
                print("Updated length of changes:", len(change_track))

            elif random_change == 2:
                change2(change_track)
                print("random index", random_change)
                print("Updated length of changes:", len(change_track))

            elif random_change == 3:
                change3(change_track)
                print("random index", random_change)
                print("Updated length of changes:", len(change_track))

            elif random_change == 4:
                change4(change_track)
                print("random index", random_change)
                print("Updated length of changes:", len(change_track))

        print("Length of all changes:", len(change_track))
        #print(change_track)
        if gen_index == 0:
            new_weights = []
            new_weights.append(lstm_cell_kernel.tolist())
            new_weights.append(lstm_cell_recurrent_kernel.tolist())
            new_weights.append(lstm_cell_bias.tolist())
            new_weights.append(lstm_cell_1_kernel.tolist())
            new_weights.append(lstm_cell_1_recurrent_kernel.tolist())
            new_weights.append(lstm_cell_1_bias.tolist())
            new_weights.append(lstm_cell_2_kernel.tolist())
            new_weights.append(lstm_cell_2_recurrent_kernel.tolist())
            new_weights.append(lstm_cell_2_bias.tolist())
            new_weights.append(lstm_cell_3_kernel.tolist())
            new_weights.append(lstm_cell_3_recurrent_kernel.tolist())
            new_weights.append(lstm_cell_3_bias.tolist())
            new_weights.append(dense_kernel.tolist())

            new_alphas = []
            lstm_cell_kernel_alpha = np.ones((15, 2048))
            lstm_cell_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_bias_alpha = np.ones((2048,))
            lstm_cell_1_kernel_alpha = np.ones((512, 2048))
            lstm_cell_1_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_1_bias_alpha = np.ones((2048,))
            lstm_cell_2_kernel_alpha = np.ones((512, 2048))
            lstm_cell_2_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_2_bias_alpha = np.ones((2048,))
            lstm_cell_3_kernel_alpha = np.ones((512, 2048))
            lstm_cell_3_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_3_bias_alpha = np.ones((2048,))
            dense_kernel_alpha = np.ones((512, 1))
            dense_bias_alpha = np.ones((1, ))

            new_alphas.append(lstm_cell_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_recurrent_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_bias_alpha.tolist())
            new_alphas.append(lstm_cell_1_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_1_recurrent_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_1_bias_alpha.tolist())
            new_alphas.append(lstm_cell_2_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_2_recurrent_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_2_bias_alpha.tolist())
            new_alphas.append(lstm_cell_3_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_3_recurrent_kernel_alpha.tolist())
            new_alphas.append(lstm_cell_3_bias_alpha.tolist())
            new_alphas.append(dense_kernel_alpha.tolist())

        for item in change_track:
            print("Item Index: ", change_track.index(item))
            print("Change Type:", item[1])
            print("")

            lstm_cell_kernel_to_use = np.array(new_weights[0], dtype = 'float32')
            lstm_cell_recurrent_kernel_to_use = np.array(new_weights[1], dtype = 'float32')
            lstm_cell_bias_to_use = np.array(new_weights[2], dtype = 'float32')
            lstm_cell_1_kernel_to_use = np.array(new_weights[3], dtype = 'float32')
            lstm_cell_1_recurrent_kernel_to_use = np.array(new_weights[4], dtype = 'float32')
            lstm_cell_1_bias_to_use = np.array(new_weights[5], dtype = 'float32')
            lstm_cell_2_kernel_to_use = np.array(new_weights[6], dtype = 'float32')
            lstm_cell_2_recurrent_kernel_to_use = np.array(new_weights[7], dtype = 'float32')
            lstm_cell_2_bias_to_use = np.array(new_weights[8], dtype = 'float32')
            lstm_cell_3_kernel_to_use = np.array(new_weights[9], dtype = 'float32')
            lstm_cell_3_recurrent_kernel_to_use = np.array(new_weights[10], dtype = 'float32')
            lstm_cell_3_bias_to_use = np.array(new_weights[11], dtype = 'float32')
            dense_kernel_to_use = np.array(new_weights[12], dtype = 'float32')
            dense_bias_to_use = dense_bias

            lstm_cell_kernel_alpha = np.ones((15, 2048))
            lstm_cell_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_bias_alpha = np.ones((2048,))
            lstm_cell_1_kernel_alpha = np.ones((512, 2048))
            lstm_cell_1_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_1_bias_alpha = np.ones((2048,))
            lstm_cell_2_kernel_alpha = np.ones((512, 2048))
            lstm_cell_2_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_2_bias_alpha = np.ones((2048,))
            lstm_cell_3_kernel_alpha = np.ones((512, 2048))
            lstm_cell_3_recurrent_kernel_alpha = np.ones((512, 2048))
            lstm_cell_3_bias_alpha = np.ones((2048,))
            dense_kernel_alpha = np.ones((512, 1))
            dense_bias_alpha = np.ones((1, ))

            #changes
            if item[1] == 1:
                if item[0] == "lstm_cell_kernel":
                    lstm_cell_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_kernel_to_use = lstm_cell_kernel_alpha*(lstm_cell_kernel_to_use)

                elif item[0] == "lstm_cell_recurrent_kernel":
                    lstm_cell_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_bias":
                    lstm_cell_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_bias_alpha[item[2][0]] = item[3][0]
                    lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

                elif item[0] == "lstm_cell_1_kernel":
                    lstm_cell_1_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)

                elif item[0] == "lstm_cell_1_recurrent_kernel":
                    lstm_cell_1_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_1_bias":
                    lstm_cell_1_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_bias_alpha[item[2][0]] = item[3][0]
                    lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

                elif item[0] == "lstm_cell_2_kernel":
                    lstm_cell_2_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)

                elif item[0] == "lstm_cell_2_recurrent_kernel":
                    lstm_cell_2_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_2_bias":
                    lstm_cell_2_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_bias_alpha[item[2][0]] = item[3][0]
                    lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

                elif item[0] == "lstm_cell_3_kernel":
                    lstm_cell_3_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

                elif item[0] == "lstm_cell_3_recurrent_kernel":
                    lstm_cell_3_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_recurrent_kernel_alpha[item[2][0]][item[2][1]] = item[3][0]
                    lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_3_bias":
                    lstm_cell_3_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_bias_alpha[item[2][0]] = item[3][0]
                    lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

                elif item[0] == "dense_kernel":
                    dense_kernel_alpha = np.array(item[4], dtype = 'float32')
                    dense_kernel_alpha[item[2][0]] = item[3][0]
                    dense_kernel_to_use = dense_kernel_alpha*dense_kernel_to_use

            elif item[1] == 2:
                if item[0] == "lstm_cell_kernel":
                    lstm_cell_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_kernel_alpha[item[2][0]][:])
                    lstm_cell_kernel_to_use = lstm_cell_kernel_alpha*(lstm_cell_kernel_to_use)

                elif item[0] == "lstm_cell_recurrent_kernel":
                    lstm_cell_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_recurrent_kernel_alpha[item[2][0]][:])
                    lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_bias":
                    lstm_cell_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_bias_alpha[:] = item[3][0]*(lstm_cell_bias_alpha[:])
                    lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

                elif item[0] == "lstm_cell_1_kernel":
                    lstm_cell_1_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_1_kernel_alpha[item[2][0]][:])
                    lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)

                elif item[0] == "lstm_cell_1_recurrent_kernel":
                    lstm_cell_1_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:])
                    lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_1_bias":
                    lstm_cell_1_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_1_bias_alpha[:] = item[3][0]*(lstm_cell_1_bias_alpha[:])
                    lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

                elif item[0] == "lstm_cell_2_kernel":
                    lstm_cell_2_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_2_kernel_alpha[item[2][0]][:])
                    lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)

                elif item[0] == "lstm_cell_2_recurrent_kernel":
                    lstm_cell_2_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:])
                    lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_2_bias":
                    lstm_cell_2_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_2_bias_alpha[:] = item[3][0]*(lstm_cell_2_bias_alpha[:])
                    lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

                elif item[0] == "lstm_cell_3_kernel":
                    lstm_cell_3_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_3_kernel_alpha[item[2][0]][:])
                    lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

                elif item[0] == "lstm_cell_3_recurrent_kernel":
                    lstm_cell_3_recurrent_kernel_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:] = item[3][0]*(lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:])
                    lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_3_bias":
                    lstm_cell_3_bias_alpha = np.array(item[4], dtype = 'float32')
                    lstm_cell_3_bias_alpha[:] = item[3][0]*(lstm_cell_3_bias_alpha[:])
                    lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

                elif item[0] == "dense_kernel":
                    dense_kernel_alpha = np.array(item[4], dtype = 'float32')
                    dense_kernel_alpha[:] = item[3][0]*(dense_kernel_alpha[:])
                    dense_kernel_to_use = dense_kernel_alpha*dense_kernel_to_use

            elif item[1] == 3:
                if item[0] == "lstm_cell_recurrent_kernel" or item[0] == "lstm_cell_1_recurrent_kernel" or item[0] == "lstm_cell_2_recurrent_kernel" or item[0] == "lstm_cell_3_recurrent_kernel":
                    alpha_to_swap = item[3]
                    item_to_swap = item[4]
                    alpha_with_swap = item[5]
                    item_with_swap = item[6]
                    layer_index = item[7][0]
                    layer_index_swap = item[7][1]
                    if layer_index == 1:
                        lstm_cell_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                        lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
                    if layer_index == 4:
                        lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_1_recurrent_kernel_alpha[item[2][0]][:]= alpha_with_swap
                        lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
                    if layer_index == 7:
                        lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_2_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                        lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
                    if layer_index == 10:
                        lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_3_recurrent_kernel_alpha[item[2][0]][:] = alpha_with_swap
                        lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)
                    if layer_index_swap == 1:
                        lstm_cell_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                        lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
                    if layer_index_swap == 4:
                        lstm_cell_1_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_1_recurrent_kernel_alpha[item[2][1]][:]= alpha_to_swap
                        lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
                    if layer_index_swap == 7:
                        lstm_cell_2_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_2_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                        lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
                    if layer_index_swap == 10:
                        lstm_cell_3_recurrent_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_3_recurrent_kernel_alpha[item[2][1]][:] = alpha_to_swap
                        lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)

                elif item[0] == "lstm_cell_1_kernel" or item[0] == "lstm_cell_2_kernel" or item[0] == "lstm_cell_3_kernel":
                    alpha_to_swap = item[3]
                    item_to_swap = item[4]
                    alpha_with_swap = item[5]
                    item_with_swap = item[6]
                    layer_index = item[7][0]
                    layer_index_swap = item[7][1]
                    if layer_index == 3:
                        lstm_cell_1_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_1_kernel_alpha[item[2][0]][:]= alpha_with_swap
                        lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
                    if layer_index == 6:
                        lstm_cell_2_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_2_kernel_alpha[item[2][0]][:] = alpha_with_swap
                        lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
                    if layer_index == 9:
                        lstm_cell_3_kernel_to_use[item[2][0]][:] = item_with_swap
                        lstm_cell_3_kernel_alpha[item[2][0]][:] = alpha_with_swap
                        lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)
                    if layer_index_swap == 3:
                        lstm_cell_1_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_1_kernel_alpha[item[2][1]][:]= alpha_to_swap
                        lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
                    if layer_index_swap == 6:
                        lstm_cell_2_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_2_kernel_alpha[item[2][1]][:] = alpha_to_swap
                        lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
                    if layer_index_swap == 9:
                        lstm_cell_3_kernel_to_use[item[2][1]][:] = item_to_swap
                        lstm_cell_3_kernel_alpha[item[2][1]][:] = alpha_to_swap
                        lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)

                elif item[0] == "lstm_cell_bias":
                    lstm_cell_bias_alpha_to_swap = item[3]
                    lstm_cell_bias_to_swap = item[4]
                    lstm_cell_bias_alpha_with_swap = item[5]
                    lstm_cell_bias_with_swap = item[6]
                    lstm_cell_bias_to_use[item[2][0]] = lstm_cell_bias_with_swap
                    lstm_cell_bias_to_use[item[2][1]] = lstm_cell_bias_to_swap
                    lstm_cell_bias_alpha[item[2][0]] = lstm_cell_bias_alpha_with_swap
                    lstm_cell_bias_alpha[item[2][1]] = lstm_cell_bias_alpha_to_swap
                    lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)

                elif item[0] == "lstm_cell_1_bias":
                    lstm_cell_1_bias_alpha_to_swap = item[3]
                    lstm_cell_1_bias_to_swap = item[4]
                    lstm_cell_1_bias_alpha_with_swap = item[5]
                    lstm_cell_1_bias_with_swap = item[6]
                    lstm_cell_1_bias_to_use[item[2][0]] = lstm_cell_1_bias_with_swap
                    lstm_cell_1_bias_to_use[item[2][1]] = lstm_cell_1_bias_to_swap
                    lstm_cell_1_bias_alpha[item[2][0]] = lstm_cell_1_bias_alpha_with_swap
                    lstm_cell_1_bias_alpha[item[2][1]] = lstm_cell_1_bias_alpha_to_swap
                    lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)

                elif item[0] == "lstm_cell_2_bias":
                    lstm_cell_2_bias_alpha_to_swap = item[3]
                    lstm_cell_2_bias_to_swap = item[4]
                    lstm_cell_2_bias_alpha_with_swap = item[5]
                    lstm_cell_2_bias_with_swap = item[6]
                    lstm_cell_2_bias_to_use[item[2][0]] = lstm_cell_2_bias_with_swap
                    lstm_cell_2_bias_to_use[item[2][1]] = lstm_cell_2_bias_to_swap
                    lstm_cell_2_bias_alpha[item[2][0]] = lstm_cell_2_bias_alpha_with_swap
                    lstm_cell_2_bias_alpha[item[2][1]] = lstm_cell_2_bias_alpha_to_swap
                    lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)

                elif item[0] == "lstm_cell_3_bias":
                    lstm_cell_3_bias_alpha_to_swap = item[3]
                    lstm_cell_3_bias_to_swap = item[4]
                    lstm_cell_3_bias_alpha_with_swap = item[5]
                    lstm_cell_3_bias_with_swap = item[6]
                    lstm_cell_3_bias_to_use[item[2][0]] = lstm_cell_3_bias_with_swap
                    lstm_cell_3_bias_to_use[item[2][1]] = lstm_cell_3_bias_to_swap
                    lstm_cell_3_bias_alpha[item[2][0]] = lstm_cell_3_bias_alpha_with_swap
                    lstm_cell_3_bias_alpha[item[2][1]] = lstm_cell_3_bias_alpha_to_swap
                    lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)

                elif item[0] == "dense_kernel":
                    dense_kernel_alpha_to_swap = item[3][0]
                    dense_kernel_to_swap = item[4][0]
                    dense_kernel_alpha_with_swap = item[5][0]
                    dense_kernel_with_swap = item[6][0]
                    dense_kernel_to_use[item[2][0]][:] = dense_kernel_with_swap
                    dense_kernel_to_use[item[2][1]][:] = dense_kernel_to_swap
                    dense_kernel_alpha[item[2][0]][:] = dense_kernel_alpha_with_swap
                    dense_kernel_alpha[item[2][1]][:] = dense_kernel_alpha_to_swap
                    dense_kernel_to_use = dense_kernel_alpha*(dense_kernel_to_use)

            elif item[1] == 4:
                if item[0] == "lstm_cell_recurrent_kernel" or item[0] == "lstm_cell_1_recurrent_kernel" or item[0] == "lstm_cell_2_recurrent_kernel" or item[0] == "lstm_cell_3_recurrent_kernel":
                    layer_index = item[7][0]
                    alpha_to_swap = item[3]
                    item_to_swap = item[4]
                    alpha_with_swap = item[5]
                    item_with_swap = item[6]
                    if layer_index == 1:
                        lstm_cell_recurrent_kernel_to_use = lstm_cell_recurrent_kernel_alpha*(lstm_cell_recurrent_kernel_to_use)
                        lstm_cell_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_recurrent_kernel_to_use[item[2][1]][:]
                    if layer_index == 4:
                        lstm_cell_1_recurrent_kernel_to_use = lstm_cell_1_recurrent_kernel_alpha*(lstm_cell_1_recurrent_kernel_to_use)
                        lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_1_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_1_recurrent_kernel_to_use[item[2][1]][:]
                    if layer_index == 7:
                        lstm_cell_2_recurrent_kernel_to_use = lstm_cell_2_recurrent_kernel_alpha*(lstm_cell_2_recurrent_kernel_to_use)
                        lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_2_recurrent_kernel_to_use[item[2][1]][:]
                    if layer_index == 10:
                        lstm_cell_3_recurrent_kernel_to_use = lstm_cell_3_recurrent_kernel_alpha*(lstm_cell_3_recurrent_kernel_to_use)
                        lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] = lstm_cell_3_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_3_recurrent_kernel_to_use[item[2][1]][:]

                elif item[0] == "lstm_cell_1_kernel" or item[0] == "lstm_cell_2_kernel" or item[0] == "lstm_cell_3_kernel":
                    layer_index = item[7][0]
                    alpha_to_swap = item[3]
                    item_to_swap = item[4]
                    alpha_with_swap = item[5]
                    item_with_swap = item[6]
                    if layer_index == 3:
                        lstm_cell_1_kernel_to_use = lstm_cell_1_kernel_alpha*(lstm_cell_1_kernel_to_use)
                        lstm_cell_1_kernel_to_use[item[2][0]][:] = lstm_cell_1_kernel_to_use[item[2][0]][:] + lstm_cell_1_kernel_to_use[item[2][1]][:]
                    if layer_index == 6:
                        lstm_cell_2_kernel_to_use = lstm_cell_2_kernel_alpha*(lstm_cell_2_kernel_to_use)
                        lstm_cell_2_kernel_to_use[item[2][0]][:] = lstm_cell_2_recurrent_kernel_to_use[item[2][0]][:] + lstm_cell_2_kernel_to_use[item[2][1]][:]
                    if layer_index == 9:
                        lstm_cell_3_kernel_to_use = lstm_cell_3_kernel_alpha*(lstm_cell_3_kernel_to_use)
                        lstm_cell_3_kernel_to_use[item[2][0]][:] = lstm_cell_3_kernel_to_use[item[2][0]][:] + lstm_cell_3_kernel_to_use[item[2][1]][:]

                elif item[0] == "lstm_cell_bias":
                    lstm_cell_bias_alpha_to_swap = item[3]
                    lstm_cell_bias_to_swap = item[4]
                    lstm_cell_bias_alpha_with_swap = item[5]
                    lstm_cell_bias_with_swap = item[6]
                    lstm_cell_bias_to_use = lstm_cell_bias_alpha*(lstm_cell_bias_to_use)
                    lstm_cell_bias_to_use[item[2][0]] = lstm_cell_bias_to_use[item[2][0]] + lstm_cell_bias_to_use[item[2][1]]

                elif item[0] == "lstm_cell_1_bias":
                    lstm_cell_1_bias_alpha_to_swap = item[3]
                    lstm_cell_1_bias_to_swap = item[4]
                    lstm_cell_1_bias_alpha_with_swap = item[5]
                    lstm_cell_1_bias_with_swap = item[6]
                    lstm_cell_1_bias_to_use = lstm_cell_1_bias_alpha*(lstm_cell_1_bias_to_use)
                    lstm_cell_1_bias_to_use[item[2][0]] = lstm_cell_1_bias_to_use[item[2][0]] + lstm_cell_1_bias_to_use[item[2][1]]

                elif item[0] == "lstm_cell_2_bias":
                    lstm_cell_2_bias_alpha_to_swap = item[3]
                    lstm_cell_2_bias_to_swap = item[4]
                    lstm_cell_2_bias_alpha_with_swap = item[5]
                    lstm_cell_2_bias_with_swap = item[6]
                    lstm_cell_2_bias_to_use = lstm_cell_2_bias_alpha*(lstm_cell_2_bias_to_use)
                    lstm_cell_2_bias_to_use[item[2][0]] = lstm_cell_2_bias_to_use[item[2][0]] + lstm_cell_2_bias_to_use[item[2][1]]

                elif item[0] == "lstm_cell_3_bias":
                    lstm_cell_3_bias_alpha_to_swap = item[3]
                    lstm_cell_3_bias_to_swap = item[4]
                    lstm_cell_3_bias_alpha_with_swap = item[5]
                    lstm_cell_3_bias_with_swap = item[6]
                    lstm_cell_3_bias_to_use = lstm_cell_3_bias_alpha*(lstm_cell_3_bias_to_use)
                    lstm_cell_3_bias_to_use[item[2][0]] = lstm_cell_3_bias_to_use[item[2][0]] + lstm_cell_3_bias_to_use[item[2][1]]

                elif item[0] == "dense_kernel":
                    dense_kernel_alpha_to_swap = item[3][0]
                    dense_kernel_to_swap = item[4][0]
                    dense_kernel_alpha_with_swap = item[5][0]
                    dense_kernel_with_swap = item[6][0]
                    dense_kernel_to_use = dense_kernel_alpha*(dense_kernel_to_use)
                    dense_kernel_to_use[item[2][0]][:] = dense_kernel_to_use[item[2][0]][:] + dense_kernel_to_use[item[2][1]][:]

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
            model.compile(optimizer = adam, loss = 'mean_squared_error')

            layer0_list = [lstm_cell_kernel_to_use, lstm_cell_recurrent_kernel_to_use, lstm_cell_bias_to_use]
            layer2_list = [lstm_cell_1_kernel_to_use, lstm_cell_1_recurrent_kernel_to_use, lstm_cell_1_bias_to_use]
            layer4_list = [lstm_cell_2_kernel_to_use, lstm_cell_2_recurrent_kernel_to_use, lstm_cell_2_bias_to_use]
            layer6_list = [lstm_cell_3_kernel_to_use, lstm_cell_3_recurrent_kernel_to_use, lstm_cell_3_bias_to_use]
            layer8_list = [dense_kernel_to_use, dense_bias_to_use]

            model.layers[0].set_weights(layer0_list)
            model.layers[2].set_weights(layer2_list)
            model.layers[4].set_weights(layer4_list)
            model.layers[6].set_weights(layer6_list)
            model.layers[8].set_weights(layer8_list)

            scores = model.evaluate(X_train, Y_train, verbose=0)
            print("MSE, MSE on test set:")
            print(scores)
            mse_matches_train.append(scores)
            #first person to optimize
            #Goal Data Test
            X_test_current = X_test[X_test_current_start_index:X_test_current_end_index, :, :]
            Y_test_current = Y_test[X_test_current_start_index:X_test_current_end_index, :]
            #No-Goal Data Test
            X_test_current_v2 = X_test_v2[X_test_v2_current_start_index:X_test_v2_current_end_index, :, :]
            Y_test_current_v2 = Y_test_v2[X_test_v2_current_start_index:X_test_v2_current_end_index, :]

            predicted_labels = []
            original_labels = []
            predicted_labels_v2 = []
            original_labels_v2 = []
            predicted_values = []
            original_values = []
            predicted_values_v2 = []
            original_values_v2 = []

            discount_factor = 0.5

            #print("Outputs for Goal Data:")
            for i in range(0, 10):
                test_input = X_test_current[i]
                test_input = test_input.reshape(1, 31, 15)
                test_output = model.predict(test_input, verbose=0)
                #print("Output from LSTM:", test_output)
                predicted_labels.append(test_output[0][0])
                #print("Original_label:", Y_test_current[i])
                original_labels.append(Y_test_current[i][0])
            #print("")
            #print("Outputs for No-Goal Data:")
            for i in range(0, 35):
                test_input_v2 = X_test_current_v2[i]
                test_input_v2 = test_input_v2.reshape(1, 31, 15)
                test_output_v2 = model.predict(test_input_v2, verbose=0)
                #print("Output from LSTM:", test_output_v2)
                predicted_labels_v2.append(test_output_v2[0][0])
                #print("Original_label:", Y_test_current_v2[i])
                original_labels_v2.append(Y_test_current_v2[i][0])
            #print("")

            all_original_labels.append(original_labels)
            all_predicted_labels.append(predicted_labels)

            for i in range(len(predicted_labels)-1, -1, -10):
                month_wise_predicted_vals = [0]*10
                iterate = 9
                for j in range(i, i-10, -1):
                    if i == j:
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
                    current_element_rounded = int(round(current_element))
                    month_wise_predicted_vals[k] = current_element_rounded
                predicted_values.append(month_wise_predicted_vals)

            for i in range(len(original_labels)-1, -1, -10):
                month_wise_predicted_vals = [0]*10
                iterate = 9
                for j in range(i, i-10, -1):
                    if i == j:
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

            for i in range(len(predicted_labels_v2)-1, -1, -35):
                month_wise_predicted_vals_v2 = [0]*35
                iterate_v2 = 34
                for j in range(i, i-35, -1):
                    if i == j:
                        current_value_v2 = predicted_labels_v2[j]
                        previous_value_v2 = predicted_labels_v2[j]
                        month_wise_predicted_vals_v2[iterate_v2] = current_value_v2
                    else:
                        current_value_v2 = predicted_labels_v2[j] - discount_factor*(previous_value_v2)
                        previous_value_v2 = current_value_v2
                        month_wise_predicted_vals_v2[iterate_v2] = current_value_v2
                    iterate_v2-=1

                for k in range(len(month_wise_predicted_vals_v2)):
                    current_element_v2 = month_wise_predicted_vals_v2[k]
                    current_element_rounded_v2 = int(round(current_element_v2))
                    month_wise_predicted_vals_v2[k] = current_element_rounded_v2
                predicted_values_v2.append(month_wise_predicted_vals_v2)

            for i in range(len(original_labels_v2)-1, -1, -35):
                month_wise_predicted_vals_v2 = [0]*35
                iterate_v2 = 34
                for j in range(i, i-35, -1):
                    if i == j:
                        current_value_v2 = original_labels_v2[j]
                        previous_value_v2 = original_labels_v2[j]
                        month_wise_predicted_vals_v2[iterate_v2] = current_value_v2
                    else:
                        current_value_v2 = original_labels_v2[j] - discount_factor*(previous_value_v2)
                        previous_value_v2 = current_value_v2
                        month_wise_predicted_vals_v2[iterate_v2] = current_value_v2
                    iterate_v2-=1

                for k in range(len(month_wise_predicted_vals_v2)):
                    current_element_v2 = month_wise_predicted_vals_v2[k]
                    current_element_rounded_v2 = int(round(current_element_v2))
                    month_wise_predicted_vals_v2[k] = current_element_rounded_v2
                original_values_v2.append(month_wise_predicted_vals_v2)

            for i in range(len(predicted_values)):
                mismatch_count = 0
                match_count = 0
                for j in range(len(predicted_values[i])):
                    if predicted_values[i][j] != original_values[i][j]:
                        mismatch_count+=1
                    else:
                        match_count+=1
            total_matches.append(match_count)
            total_mismatches.append(mismatch_count)
            all_matches.append(match_count)
            all_mismatches.append(mismatch_count)

            for i in range(len(predicted_values_v2)):
                mismatch_count_v2 = 0
                match_count_v2 = 0
                for j in range(len(predicted_values_v2[i])):
                    if predicted_values_v2[i][j] != original_values_v2[i][j]:
                        mismatch_count_v2+=1
                    else:
                        match_count_v2+=1
            total_matches_v2.append(match_count_v2)
            total_mismatches_v2.append(mismatch_count_v2)
            all_matches_v2.append(match_count_v2)
            all_mismatches_v2.append(mismatch_count_v2)

            print(total_matches)
            print(total_mismatches)
            print(total_matches_v2)
            print(total_mismatches_v2)
            print(mse_matches_train)

        current_best_matches, current_mse = calculate_best_matches(total_matches, total_mismatches, total_matches_v2, total_mismatches_v2, matches_to_beat, mse_to_maintain, mse_matches_train)
        current_best_matches_index = total_matches_v2.index(current_best_matches)

        if matches_to_beat > current_best_matches:
            print("New total number of matches not better than previous value")
            break

        elif matches_to_beat <= current_best_matches and current_mse <= mse_to_maintain:
            print("Previous best matches:", matches_to_beat)
            matches_to_beat = current_best_matches
            print("New best matches to beat:", matches_to_beat)
            print("Previous best MSE on Goal Training Data:", mse_to_maintain)
            mse_to_maintain = current_mse
            print("New best MSE on Goal Training Data:", mse_to_maintain)
            new_weights = calculate_new_weights(change_track, current_best_matches_index, matches_to_beat, new_weights, new_alphas)

        change_track = []
        print("Length of change tracks total", len(change_track_total))

    max_match_val = max(all_matches)
    frequency_max_match_val = all_matches.count(max_match_val)
    print("Maximum value of matches in goal data:", max_match_val)
    print("Frequency of maximum value attained:", frequency_max_match_val)

    predicted_bellman_vals_to_save = []
    original_bellman_vals_to_save = []
    pickle_to_save = []
    predicted_bellman_vals_to_save_id = len(all_predicted_labels) - 1
    predicted_bellman_vals_to_save.append(all_predicted_labels[predicted_bellman_vals_to_save_id])
    original_bellman_vals_to_save.append(all_original_labels[predicted_bellman_vals_to_save_id])
    pickle_to_save.append(predicted_bellman_vals_to_save)
    pickle_to_save.append(original_bellman_vals_to_save)
    pickle_to_save_filename = str(individual_id) + '_greedy_bellman_vals.txt'

    with open('./Greedy_bellman_vals_cv1/'+pickle_to_save_filename, "wb") as fp:   #Pickling
        pickle.dump(pickle_to_save, fp)

#main()
if __name__ == "__main__":
    total_gen = 100
    apprun(total_gen)
