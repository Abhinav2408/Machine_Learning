import sys
import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC as svm

traindataaddress = 'C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part2_data/part2_data/train_data.pickle'
testdataaddress = 'C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part2_data/part2_data/test_data.pickle'

if(len(sys.argv)!=1):
    traindataaddress = sys.argv[1] + "/train_data.pickle"
    testdataaddress = sys.argv[2] + "/test_data.pickle"

entrynum = '2020MT10780'
#d=0, and d+1 = 1
#we need 0 and 1 classes

netdata = pd.read_pickle(traindataaddress)

totaldata = netdata['data']
totallabel = netdata['labels']

data = np.array([totaldata[i] for i in range(len(totaldata)) if totallabel[i]<=1])
labels = np.array([totallabel[i] for i in range(len(totallabel)) if totallabel[i]<=1])
trainingdata = [[]]
for i in data:
    trainingdata.append(i.ravel(order='C'))

trainingdata.pop(0)
trainingdata = np.array(trainingdata)
trainingdata = trainingdata.astype(float)
trainingdata=trainingdata/255
labels = labels.astype(float)


for i in range(len(labels)):
    if labels[i]==[0]:
        labels[i][0] = -1



testingtempdata = pd.read_pickle(testdataaddress)
testdataitems = testingtempdata['data']
testdatalabels = testingtempdata['labels']


tempdata = np.array([testdataitems[i] for i in range(len(testdataitems)) if testdatalabels[i]<=1])
testlabels = np.array([testdatalabels[i] for i in range(len(testdatalabels)) if testdatalabels[i]<=1])

testingdata = [[]]
for i in tempdata:
    testingdata.append(i.ravel(order='C'))

testingdata.pop(0)
testingdata = np.array(testingdata)
testingdata = testingdata.astype(float)
testingdata = testingdata/255

testlabels = testlabels.astype(float)

for i in range(len(testlabels)):
    if testlabels[i]==[0]:
        testlabels[i][0] = -1



linst = time.time()

linsvc = svm(kernel='linear',gamma='scale')
linsvc.fit(trainingdata,labels[:,0])
predictedlin = linsvc.predict(testingdata)

linend = time.time()
linacc = 100*linsvc.score(testingdata,testlabels)

wlin = linsvc.coef_
blin = linsvc.intercept_
linvector = linsvc.n_support_
print("The time taken using LIBSVM and Linear Kernel = {:2.3f}sec".format(linend-linst))  
print("Number of linear support vectors are",linvector)
print("Linear weights are :")
print(wlin)

print("b in Linear is :")
print(blin)
print("Accuracy is",linacc,"%")






gaussst = time.time()

gausssvc = svm(kernel='rbf',gamma=0.001)
gausssvc.fit(trainingdata,labels[:,0])
predictedgauss = gausssvc.predict(testingdata)

gaussend = time.time()
gaussacc = 100*gausssvc.score(testingdata,testlabels)

bgauss = gausssvc.intercept_
gaussvector = gausssvc.n_support_

print("The time taken using LIBSVM and Gaussian Kernel = {:2.3f}sec".format(gaussend-gaussst))  

print("Number of gaussian support vectors are",gaussvector)
print("b in Gauss is :")
print(bgauss)
print("Accuracy is",gaussacc,"%")