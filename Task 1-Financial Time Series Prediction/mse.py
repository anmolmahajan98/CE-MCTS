from numpy import array
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
import natsort
import os
from sklearn.metrics import mean_squared_error
all_mse_total = 0
all_mse_avg = 0
all_sd = 0
for index in range(0, 5):
    l = os.listdir('./Bellman_vals_cv'+str(index+1)+'_v2/')
    l = natsort.natsorted(l,reverse=False)
    mse_total  = 0
    mse_vals_final = []
    for i in range(0, len(l)):
        file_to_load = l[i]
        mse_vals = []
        with open('./Bellman_vals_cv'+str(index+1)+'_v2/'+file_to_load, "rb") as fp:
            loaded_pickle = pickle.load(fp)
            #print(loaded_pickle)
            #print(loaded_pickle[0])
            #print(loaded_pickle[1])
            current_mse_vals = []
            loaded_pickle_original_vals = [] 
            loaded_pickle_predicted_vals = []
            for j in range(0, len(loaded_pickle[1])):
                #print(loaded_pickle[1][j])
                #print(loaded_pickle[0][j])
                loaded_pickle_original_vals = loaded_pickle[1][j]
                loaded_pickle_predicted_vals = loaded_pickle[0][j]
                loaded_pickle_original_vals = [abs(number) for number in loaded_pickle_original_vals]
                loaded_pickle_predicted_vals = [abs(number) for number in loaded_pickle_predicted_vals]
                current_mse_vals.append(mean_squared_error(loaded_pickle_predicted_vals, loaded_pickle_original_vals))
        #print(current_mse_vals)

            #print(current_mse_vals)          
    #         print(loaded_pickle_original_vals)
    #         print(loaded_pickle_predicted_vals)
        mse = min(current_mse_vals)
        mse_vals.append(abs(mse))
        mse_vals_final.append(abs(mse))
        mse_total+=mse
    print(mse_total)
    print(float(mse_total/len(l)))

    # print(mse_vals)
    mse_vals_numpy = np.array(mse_vals_final)
    #print(mse_vals_numpy)
    print(np.std(mse_vals_numpy))
    print("")
    all_mse_total+=mse_total
    all_mse_avg+=float(mse_total/len(l))
    all_sd+=np.std(mse_vals_numpy)

print(all_mse_total/5)
print(all_mse_avg/5)
print(all_sd/5)











