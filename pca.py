# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:47:03 2019

@author: tanvip
"""

from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Import svm model
from sklearn import svm
import warnings
from sklearn.svm import SVC
import math
from matplotlib.backends.backend_pdf import PdfPages
###############################################################
filename = "T:\\USERS\\Tanvi_Patil\\Exer - 7\\machine-learning-ex7\\ex7\\"
filename1 = "T:\\USERS\\Tanvi_Patil\\Exer - 7\\machine-learning-ex7\\ex7\\ex7data2.mat"
mat = scipy.io.loadmat(filename1)

mat1 = []
mat2 = []
for i in range(0,len(mat['X'])):
    mat1.append(mat['X'][i][0])
    mat2.append(mat['X'][i][1])

plt.scatter(mat1, mat2, color='r')
#plt.scatter(negative_x1x2['X1'], negative_x1x2['X2'], color='g')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()    

df = pd.DataFrame({'x1':mat1,'x2':mat2})
#df1 = pd.DataFrame(data = mat1, index = range(0,len(mat1)), columns = ['x1'])
##########################################################################################
#normalizating by standardscalar

from sklearn import preprocessing
# Get column names first
names = df.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

#alternate method
mean = df.mean()
deviation = df.std()

for i in range(0,len(df)):
    df['x1'][i] = (df['x1'][i] - mean['x1'])/deviation['x1']
    df['x2'][i] = (df['x2'][i] - mean['x2'])/deviation['x2']

##########################################################################################
#Get covariance matrix and eigen values and vectors    

cov = scaled_df.cov()
array_cov = np.array(cov)
import numpy as np
from numpy import linalg as LA

w, v = LA.eig(array_cov)
top_eigen_vector = v[0]
scaled_df1 = scaled_df.transpose()
top_eigen_vector = np.array([-0.707106, -0.707106])
##########################################################################################
#""" Return columns of X projected onto line defined by u
def line_projection(top_eigen_vector, scaled_df1):
    u = top_eigen_vector.reshape(1, 2)  # A row vector
    c_values = u.dot(scaled_df1)  # c values for scaling u
    projected = u.T.dot(c_values)
    return projected

projected = line_projection(top_eigen_vector, scaled_df1)
projected1 = projected.transpose()

mat_projected_1 = []
mat_projected_2 = []
for i in range(0,len(projected1)):
    mat_projected_1.append(projected1[i][0])
    mat_projected_2.append(projected1[i][1])
    
plt.scatter(scaled_df['x1'], scaled_df['x2'], color='r')
plt.scatter(mat_projected_2, mat_projected_1, color='b')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

proj = pd.DataFrame({'x1' : mat_projected_2, 'x2' : mat_projected_1})

filename4 = "T:\\USERS\\Tanvi_Patil\\Exer - 7\\machine-learning-ex7\\ex7\\ex7faces.mat"
mat_face = scipy.io.loadmat(filename4)




