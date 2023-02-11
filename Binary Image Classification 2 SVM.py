import sys
import time
import cvxopt
import pandas as pd
import numpy as np
from scipy import spatial

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


trainingdata = trainingdata/255


labels = labels.astype(float)

for i in range(len(labels)):
    if labels[i]==[0]:
        labels[i][0] = -1

m = np.shape(trainingdata)[0]
c=1
g = 0.001

sttime = time.time()
Kxz = np.zeros((m,m),dtype=float)

comdist = spatial.distance.pdist(trainingdata,'sqeuclidean')


Kxz = np.exp(-1*g*spatial.distance.squareform(comdist))



P = np.multiply(Kxz, np.matmul(labels,np.transpose(labels)))
P = cvxopt.matrix(P)
q = -1*np.ones((m,1))
q = cvxopt.matrix(q)
A = cvxopt.matrix(np.transpose(labels))
bcoeff = cvxopt.matrix(0.00)
G = cvxopt.matrix(np.vstack((-1*np.identity(m),np.identity(m))))
h = cvxopt.matrix(np.hstack((np.zeros(m),c*np.ones(m))))
sol = cvxopt.solvers.qp(P,q,G,h,A,bcoeff)
curralpha = np.array(sol['x'])
error = 1e-3
nSV = 0
for i in curralpha:
    if i>error:
        nSV+=1

print("Number of support vectors are",nSV)
print("Percentage of support vectors =", (100*nSV)/m,"%")

s = (curralpha>error)

svmidx = np.where(s==True)[0]

s = s.flatten()
comdist = spatial.distance.pdist(trainingdata[svmidx],'sqeuclidean')
newKxz = np.exp(-1*g*spatial.distance.squareform(comdist))


svmalpha = np.array([curralpha[i] for i in range(len(curralpha)) if curralpha[i]>error])
svmdata = np.array([trainingdata[i] for i in range(len(trainingdata)) if curralpha[i]>error])
svmlabels = np.array([labels[i] for i in range(len(labels)) if curralpha[i]>error])



######   weight must be sum of alphai*yi*phixi. But it will be costly. We can find individual weight for each element by multiplying it by phixj, making it kernel(i,j) 
###### b is again mean of yi - wi*phixi = yi - aboveweight
wxtrain = np.matmul(np.transpose(newKxz),curralpha[s]*labels[s])
b = np.mean(labels[s] - wxtrain)
endtime = time.time()
print("The time taken using CVXOPT and Gaussian Kernel = {:2.3f}sec".format(endtime-sttime))  
print("W is not fixed")
print("b is",b)




##########    testing    ##########



testingtempdata = pd.read_pickle(testdataaddress)
testdataitems = testingtempdata['data']
testdatalabels = testingtempdata['labels']


tempdata = np.array([testdataitems[i] for i in range(len(testdataitems)) if testdatalabels[i]<=1])
testlabels = np.array([testdatalabels[i] for i in range(len(testdatalabels)) if testdatalabels[i]<=1])

testlabels = testlabels.astype(float)
testingdata = [[]]
for i in tempdata:
    testingdata.append(i.ravel(order='C'))
testingdata.pop(0)
testingdata = np.array(testingdata)
testingdata = testingdata.astype(float)
testingdata = testingdata/255
for i in range(len(testlabels)):
    if testlabels[i]==[0]:
        testlabels[i][0] = -1

n = np.shape(testlabels)[0]

predictedlabels = np.zeros((n,1))


newcomdist = spatial.distance.cdist(trainingdata[svmidx],testingdata,'sqeuclidean')
testKxz = np.exp(-1*g*newcomdist)

wxtest = np.dot(np.transpose(testKxz),labels[svmidx]*curralpha[svmidx])
pred = wxtest + b
predictedlabels = [1 if i>=0 else -1 for i in pred]

correctcnt = 0
incorrectcnt = 0

for i in range(len(testlabels)):
    if testlabels[i]==predictedlabels[i]:
        correctcnt+=1
    else:
        incorrectcnt+=1

accuracy = (100*correctcnt)/(correctcnt+incorrectcnt)

print("Accuracy is",accuracy,"%")




import matplotlib.pyplot as plt

top5images = svmdata[-5:]

for i in range(len(top5images)):
    img = top5images[i]
    image = img.reshape(3,32,32)
    image = image.transpose(1,2,0)
    image = 255*image
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.title("Image")
    plt.savefig("image"+str(i)+".jpg")
    plt.show()

