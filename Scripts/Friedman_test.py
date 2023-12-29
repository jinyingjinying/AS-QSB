# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:19:10 2023

@author: yjin
"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['axes.unicode_minus'] = False


"""
    Construct descending ordering matrix
"""

def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## Ascending sort index
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## Same ranking score order value
            if j < 3 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    return matrix


"""
    Friedman test
    Parameters: Number of data sets n, number of algorithms k, rank_matrix (k x n)
    Function returns the test result 
    (a one-dimensional array corresponding to the ordering of the columns of the sort matrix)
"""

def friedman(n, k, rank_matrix):
    # Computes the sort sum for each column
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result


"""
    Nemenyi test
    Parameters: Number of data sets n, number of algorithms k, rank_matrix (k x n)
    Function returns CD value
"""

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))

dummy = [0.4993, 0.5146, 0.5141, 0.5144]
svm = [0.8118, 0.8167, 0.8187, 0.8177]
dnn = [0.842, 0.8385, 0.8606, 0.8493]
knn = [0.9424, 0.9616, 0.9253, 0.9431]
rf = [0.9435, 0.9468, 0.9435, 0.9452]

data = [dummy, svm, dnn, knn, rf]

matrix = np.array(data)
matrix_r = rank_matrix(matrix.T)
Friedman = friedman(5, 5, matrix_r)
CD = nemenyi(5, 5, 2.728)

## friedman plot
rank_x = list(map(lambda x: np.mean(x), matrix))
name_y = ["Dummy","SVM","DNN","KNN","RF"]
min_ = [x for x in rank_x - CD/2]
max_ = [x for x in rank_x + CD/2]

fig, ax = plt.subplots(figsize=(8,6))

colors = ['#8A8D90','#4F9FC7','#FFCF7F','#D2B4DE', '#A9BCD0']

plt.ylim(-0.5,4.5)
plt.hlines(name_y,min_,max_, colors=colors,zorder=1)
plt.scatter(rank_x,name_y, c=colors,zorder=2)
plt.xlabel('Rank value',fontsize=15,fontweight='bold')
plt.yticks(fontsize=12)
plt.ylabel('Classifiers',fontsize=15,fontweight='bold')

plt.savefig("D:\\document\\AS-QSB\\friedman_test.png",dpi=500, bbox_inches = 'tight')
plt.show()

"""
	Mann-Whitney U test
"""

print(stats.mannwhitneyu(dummy,knn,alternative='two-sided'))
print(stats.mannwhitneyu(dummy,rf,alternative='two-sided'))

