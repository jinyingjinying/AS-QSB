# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:08:07 2023

@author: yjin
"""


import os
import time
import joblib
import numpy as np
from sklearn import preprocessing
from pandas.core.frame import DataFrame


os.chdir('d://document//AS-QSB//machine_learning')

def main():
	
    start = time.time()
    
	# load model, choose models from svm/knn/rf.pkl
    model = joblib.load('saved_model/rf.pkl')
    # model = joblib.load('saved_model/knn.pkl')
    # model = joblib.load('saved_model/svm.pkl')
    print('load model done!')
    
	# load data
    data = np.loadtxt('data/predict/data.csv', delimiter=',')
    print('load data done!')
    
	# min-max scaler
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
    data = scaler.transform(data)
    print('min-max scaler done!')
    
    # predict
    name_list = []
    to_list = model.predict(data)
    print('predict done!')
    for i in to_list:
        name_list.append(i)
    name = DataFrame(name_list)
    print('dataframe done!')
    
    # results output
    name.to_csv('output/rf_results.csv',encoding = 'utf-8')
    # name.to_csv('output/knn_results.csv',encoding = 'utf-8')
    # name.to_csv('output/svm_results.csv',encoding = 'utf-8')
    print('output done!')
    
    end = time.time()
    elapsed_time = end - start 
    minutes, seconds = divmod(elapsed_time, 60) 
    print(f"Elapsed time: {int(minutes)} minutes {int(seconds)} seconds")
    
    
    '''
    Running time of these classifiers
    It takes about 40 hours for training the SVM classifier.
    It takes about 1 hour for training the KNN classifier.
    It takes about 2 minutes for training the RF classifier.
    '''
    
    
if __name__ == '__main__':
	main()  
