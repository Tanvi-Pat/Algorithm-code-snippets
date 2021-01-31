# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:52:01 2019

@author: tanvip
"""
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filename = "T:\\USERS\\Tanvi_Patil\\SVM\\machine-learning-ex6\\ex6\\ex6data1.mat"
filename2 = "T:\\USERS\\Tanvi_Patil\\SVM\\machine-learning-ex6\\ex6\\"
mat = scipy.io.loadmat(filename)
df = pd.DataFrame(np.hstack((mat['X'], mat['y'])))
df.head()
df.shape
positive = df[df[2]==1]
negative = df[df[2]==0]

positive_x1x2 = pd.DataFrame({'X1':positive[0],'X2':positive[1]})
negative_x1x2 = pd.DataFrame({'X1':negative[0],'X2':negative[1]})

plt.scatter(positive_x1x2['X1'], positive_x1x2['X2'], color='r')
plt.scatter(negative_x1x2['X1'], negative_x1x2['X2'], color='g')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(mat['X'], mat['y'], test_size=0.3,random_state=109)


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(C= 1, kernel='linear',probability = True,verbose = True) # Linear Kernel
clf1 = svm.SVC(C= 10, kernel='linear',probability = True,verbose = True)
clf2 = svm.SVC(C= 20, kernel='linear',probability = True,verbose = True)
clf3 = svm.SVC(C= 50, kernel='linear',probability = True,verbose = True)
clf4 = svm.SVC(C= 100, kernel='linear',probability = True,verbose = True)

#Train the model using the training sets
clf.fit(X_train, y_train)
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)

color = ['green' if c == 0 else 'red' for c in y_train]
###############################################################
# Plot data points and color using their class
color = ['green' if c == 0 else 'red' for c in y_train]
plt.scatter(X_train[:,0], X_train[:,1], c=color)

# Create the hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# Plot the hyperplane
plt.plot(xx, yy)
fig1 = plt.gcf();
fig1.savefig(filename2 + 'tessstttyyy.png', dpi=100)
##############################################################
#To save one plot
#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

################################################################
#To save all plots together
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(filename2+'linear_svm.pdf')

##c=1
plt.scatter(X_train[:,0], X_train[:,1], c=color)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 4.5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy,c='#DDDDF7', linewidth=1)
fig1 = plt.gcf();
#pp.savefig(fig1)

#c=10
plt.scatter(X_train[:,0], X_train[:,1], c=color)
w = clf1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 4.5)
yy = a * xx - (clf1.intercept_[0]) / w[1]
plt.plot(xx, yy,c='#C0BEF9', linewidth=1)
fig2 = plt.gcf();
#pp.savefig(fig2)

#c=30
plt.scatter(X_train[:,0], X_train[:,1], c=color)
w = clf2.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 4.5)
yy = a * xx - (clf2.intercept_[0]) / w[1]
plt.plot(xx, yy, c= '#6E6BF7', linewidth=1)
fig3 = plt.gcf();
#pp.savefig(fig3)

#c=60
plt.scatter(X_train[:,0], X_train[:,1], c=color)
w = clf3.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 4.5)
yy = a * xx - (clf3.intercept_[0]) / w[1]
plt.plot(xx, yy, c = '#1310BB', linewidth=1)
fig4 = plt.gcf();
#pp.savefig(fig4)

#c=100
plt.scatter(X_train[:,0], X_train[:,1], c=color)
w = clf4.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2.5, 4.5)
yy = a * xx - (clf4.intercept_[0]) / w[1]
plt.plot(xx, yy, c = '#050421', linewidth=1)
fig5 = plt.gcf();
pp.savefig(fig5)

pp.close()
################################################################













