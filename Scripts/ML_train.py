# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:26:45 2023

@author: yjin
"""


import os
import time
import joblib
import sklearn
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Change the current working directory to the specified path
os.chdir('d://document//AS-QSB//machine_learning')

def model(model, data, label):
	if model == "svm":
		svm = sklearn.svm.SVC(C=1, gamma=0.125, kernel='rbf', probability=True)
		svm.fit(data, label)
		return svm
	if model == "knn":
		knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
		knn.fit(data, label)
		return knn
	if model == "rf":
		rf = RandomForestClassifier(n_estimators=20, max_depth = 100,
                                    max_features='auto', criterion='gini',
                                    random_state = 24, n_jobs=-1)
		rf.fit(data, label)
		return rf


def main():
	
	# input the type of classifiers (svm/knn/rf)
	estimator = 'rf'
	# estimator = 'knn'
	# estimator = 'svm'
	# estimator = 'rd'
	data_path = 'data/train/data.csv'
	label_path = 'data/train/label.csv'

	# read train
	data = np.loadtxt(data_path, delimiter=',')
	label = np.loadtxt(label_path, delimiter=',')
	
	# min-max scaler
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
	data = scaler.transform(data)
	
	# five-folds
	cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
	
	# indicators
	accuracy = metrics.make_scorer(metrics.accuracy_score)
	precision = metrics.make_scorer(metrics.precision_score)
	recall = metrics.make_scorer(metrics.recall_score)
	f1 = metrics.make_scorer(metrics.f1_score)
	scorer = {'accuracy': accuracy, 'precision': precision, 'recall': recall, "f1": f1}
	
	start = time.time()
	
	# 5-fold cross validation
	five_folds = model_selection.cross_validate(model(estimator, data, label), data, label, cv=cv, scoring=scorer)
	std_accuracy = np.std(five_folds['test_accuracy'])
	std_precision = np.std(five_folds['test_precision'])
	std_sensitive = np.std(five_folds['test_recall'])
	std_f1 = np.std(five_folds['test_f1'])
	
	mean_accuracy = np.mean(five_folds['test_accuracy'])
	mean_precision = np.mean(five_folds['test_precision'])
	mean_sensitive = np.mean(five_folds['test_recall'])
	mean_f1 = np.mean(five_folds['test_f1'])
	
	# results output
	print('Mean')
	print('{}: [Accuracy: {:.4f}, Precision: {:.4f}, Sensitive: {:.4f}, F1: {:.4f}]'.format(
		estimator, mean_accuracy, mean_precision, mean_sensitive, mean_f1))
	print('Std error')
	print('{}: [Accuracy: {:.4f}, Precision: {:.4f}, Sensitive: {:.4f}, F1: {:.4f}]'.format(
		estimator, std_accuracy, std_precision, std_sensitive, std_f1))
	
	lower1, upper1 = stats.norm.interval(0.95, loc=mean_accuracy, scale=std_accuracy)
	lower2, upper2 = stats.norm.interval(0.95, loc=mean_precision, scale=std_precision)
	lower3, upper3 = stats.norm.interval(0.95, loc=mean_sensitive, scale=std_sensitive)
	lower4, upper4 = stats.norm.interval(0.95, loc=mean_f1, scale=std_f1)
	
	print('Confidence interval')
	print('{}: {:.4f}, {:.4f}'.format(estimator, lower1, upper1))
	print('{}: {:.4f}, {:.4f}'.format(estimator, lower2, upper2))
	print('{}: {:.4f}, {:.4f}'.format(estimator, lower3, upper3))
	print('{}: {:.4f}, {:.4f}'.format(estimator, lower4, upper4))
        
	# save model
	joblib.dump(model(estimator, data, label), 'saved_model/rf2.pkl')
	# joblib.dump(model(estimator, data, label), 'saved_model/knn2.pkl')
	# joblib.dump(model(estimator, data, label), 'saved_model/svm2.pkl')
	print('save model done!')
	
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

