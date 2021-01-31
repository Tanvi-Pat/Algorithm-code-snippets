# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:22:23 2019

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
    
df = pd.DataFrame({'x_mean':mat1,'y_mean':mat2})

k = 3

def distance(x1,x2,y1,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 
    return dist

#random sampling from dataframe
random_cluster = df.sample(n=k)
filename3 = 'T:\\USERS\\Tanvi_Patil\\Exer - 7\\k_mean_1\\'

for  iter in range(0,5):
    list_all = pd.DataFrame(columns = ('X','Y','cluster_id'))
    for i in range(0,len(mat['X'])):
        dist = []    
        for j in range(0,k):
            dist.append(distance(mat1[i],list(random_cluster['x_mean'])[j],mat2[i],list(random_cluster['y_mean'])[j]))
        min_dist = min(dist)
        lis = []
        for j in range(0,len(random_cluster)):
            if(dist[j] == min_dist):
                lis.append(j)
        for l in range(0,len(lis)):
            test = pd.DataFrame({'X' : [mat1[i]], 'Y' : [mat2[i]], 'cluster_id' : [lis[l]]})
            list_all = list_all.append(test)
    
    pp = PdfPages(filename3+'k_means_iter'+str(iter)+'.pdf')
    plt.scatter(mat1, mat2, color='r')
    s= [300]
    plt.scatter(random_cluster['x_mean'],random_cluster['y_mean'], color ='b')
    plt.xlabel('X1')
    plt.ylabel('X2')
    fig1 = plt.gcf();
    plt.show()
    pp.savefig(fig1)
    pp.close()
    display(random_cluster)
    
    random_cluster = pd.DataFrame(columns = ('x_mean','y_mean'))
    for i in range(0,k):
        x1 = list_all[list_all['cluster_id'] == i]['X']
        y1 = list_all[list_all['cluster_id'] == i]['Y']
        test = pd.DataFrame({'x_mean' : [x1.mean()], 'y_mean' : [y1.mean()]})
        random_cluster = random_cluster.append(test)    
 
#list_all[list_all['cluster_id']==2]

      
            
#########################################################################################
#Merge PDFs into one    
import sys
import os
sys.path.append('T:/USERS/Tanvi_Patil/PyPDF2-1.26.0')
import PyPDF2
from PyPDF2 import PdfFileMerger
filename3 = 'T:\\USERS\\Tanvi_Patil\\Exer - 7\\k_mean_1\\'

#Get all the PDF filenames
pdf2merge = []
for filename in os.listdir(filename3):
    if filename.endswith('.pdf'):
        pdf2merge.append(filename)

pdfWriter = PyPDF2.PdfFileWriter()

#loop through all PDFs
filename= 'k_means_iter0.pdf'
for filename in pdf2merge:
    pdfFileObj = open(filename3+filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    for pageNum in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(pageNum)
            pdfWriter.addPage(pageObj)
#save PDF to file, wb for write binary
pdfOutput = open(filename3+'k_means_1_collated.pdf', 'wb')
#Outputting the PDF
pdfWriter.write(pdfOutput)
#Closing the PDF writer
pdfOutput.close()
######################################################################################

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
filename = "T:\\USERS\\Tanvi_Patil\\Exer - 7\\machine-learning-ex7\\ex7\\"
# Read Images 
img = mpimg.imread(filename + 'bird_small.png') 
reshape()  
# Output Images 
plt.imshow(img) 
img.shape()
import cv2
frame = cv2.imread("temp/i.jpg")


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    

    