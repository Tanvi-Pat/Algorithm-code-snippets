# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:52:01 2019

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

###############################################################
filename = "T:\\USERS\\Tanvi_Patil\\SVM\\machine-learning-ex6\\ex6\\ex6data3.mat"
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


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X_train, y_train, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])

    # plot the decision surface
    x1_min, x1_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    x2_min, x2_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    train_x_variable= pd.DataFrame(data= X_train, columns = ['x1','x2'])
    train_y_variable= pd.DataFrame(data= y_train, columns = ['y'])
    train = pd.concat([train_x_variable,train_y_variable],axis= 1)
    
    for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=train[train['y']==cl]['x1'], y=train[train['y']==cl]['x2'],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X_train[list(test_idx), :], y_train[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X_train[test_idx, :], y_train[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

        
clf1 = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
clf2 = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
clf3 = SVC(kernel='rbf', random_state=0, gamma=10, C=1)
clf4 = SVC(kernel='rbf', random_state=0, gamma=100, C=1)

clf5 = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1)
clf6 = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10)
clf7 = SVC(kernel='rbf', random_state=0, gamma=0.01, C=1000)
clf8 = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10000)


clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)
clf5.fit(X_train, y_train)
clf6.fit(X_train, y_train)
clf7.fit(X_train, y_train)
clf8.fit(X_train, y_train)
      
# Visualize the decision boundaries
plot_decision_regions(X_train, y_train, classifier=clf5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

###############################################################
#To save all plots together

from matplotlib.backends.backend_pdf import PdfPages

clf = [clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8]

#clf1.gamma
#clf1.C


filename3 = 'T:\\USERS\\Tanvi_Patil\\SVM\\machine-learning-ex6\\ex6\\rbf_svm_data3\\'
for i in range(0,8):
    print(i)
    pp = PdfPages(filename3+'rbf_svm_'+str(clf[i].gamma)+'_'+str(clf[i].C)+'.pdf')

    plot_decision_regions(X_train, y_train, classifier=clf[i])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.title('rbf_svm_'+str(clf[i].gamma)+'_'+str(clf[i].C))
    fig1 = plt.gcf();
    plt.show()
    pp.savefig(fig1)    
    pp.close()

###############################################################
#Merge PDFs into one    
import sys
import os
sys.path.append('T:/USERS/Tanvi_Patil/PyPDF2-1.26.0')
import PyPDF2
from PyPDF2 import PdfFileMerger


#Get all the PDF filenames
pdf2merge = []
for filename in os.listdir(filename3):
    if filename.endswith('.pdf'):
        pdf2merge.append(filename)

pdfWriter = PyPDF2.PdfFileWriter()

#loop through all PDFs
for filename in pdf2merge:
    pdfFileObj = open(filename3+filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    for pageNum in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(pageNum)
            pdfWriter.addPage(pageObj)
#save PDF to file, wb for write binary
pdfOutput = open(filename3+'SVM_rbf_dataset3.pdf', 'wb')
#Outputting the PDF
pdfWriter.write(pdfOutput)
#Closing the PDF writer
pdfOutput.close()

###############################################################
#Merge PDFs into one    (Alternate ay)
pdfs = pdf2merge

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(filename3 + pdf)

merger.write(filename3 + 'result.pdf')
merger.close()    

###############################################################


