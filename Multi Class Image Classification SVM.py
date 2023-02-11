import time
import cvxopt
import pandas as pd
import numpy as np
from scipy import spatial
import sys

traindataaddress = 'C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part2_data/part2_data/train_data.pickle'
testdataaddress = 'C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part2_data/part2_data/test_data.pickle'

if(len(sys.argv)!=1):
    traindataaddress = sys.argv[1] + "/train_data.pickle"
    testdataaddress = sys.argv[2] + "/test_data.pickle"

netdata = pd.read_pickle(traindataaddress)
totaldata = netdata['data']
totallabel = netdata['labels']

testingtempdata = pd.read_pickle(testdataaddress)
testdata = testingtempdata['data']
testlabels = testingtempdata['labels']
n = np.shape(testlabels)[0]
testdatafreq = [[0,0,0,0,0] for _ in range(len(testdata))]
testdatascores = [[0,0,0,0,0] for _ in range(len(testdata))]

testingdata = [[]]
for idx in testdata:
    testingdata.append(idx.ravel(order='C'))
testingdata.pop(0)
testingdata = np.array(testingdata)
testingdata = testingdata.astype(float)
testingdata = testingdata/255

c= 1
g = 0.001

sttime = time.time()
for i in range(5):
    for j in range(i+1,5,1):
        data = np.array([totaldata[k] for k in range(len(totaldata)) if (totallabel[k]==i or totallabel[k] ==j)])
        labels = np.array([totallabel[k] for k in range(len(totallabel)) if (totallabel[k]==i or totallabel[k] ==j)])
        trainingdata = [[]]
        for a in data:
            trainingdata.append(a.ravel(order='C'))
        trainingdata.pop(0)
        trainingdata = np.array(trainingdata)
        trainingdata = trainingdata.astype(float)
        trainingdata = trainingdata/255
        labels = labels.astype(float)

        maxval = np.max(labels)
        minval = np.min(labels)

        for a in range(len(labels)):
            if labels[a]==[minval]:
                labels[a][0] = -1
            else:
                labels[a][0] = 1
        
        m = np.shape(trainingdata)[0]


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

        s = (curralpha>error)
        svmidx = np.where(s==True)[0]
        s = s.flatten()
        comdist = spatial.distance.pdist(trainingdata[svmidx],'sqeuclidean')
        newKxz = np.exp(-1*g*spatial.distance.squareform(comdist))
        wxtrain = np.matmul(np.transpose(newKxz),curralpha[s]*labels[s])
        b = np.mean(labels[s] - wxtrain)


        newcomdist = spatial.distance.cdist(trainingdata[svmidx],testingdata,'sqeuclidean')
        testKxz = np.exp(-1*g*newcomdist)
        wxtest = np.dot(np.transpose(testKxz),labels[svmidx]*curralpha[svmidx])
        pred = wxtest + b
        predictedlabels = [maxval if a>=0 else minval for a in pred]
        scores = [abs(a) for a in pred]
        for cnt in range(n):
            testdatafreq[cnt][int(predictedlabels[cnt])] +=1
            testdatascores[cnt][int(predictedlabels[cnt])] +=scores[cnt]
        print("Classifier",str(i),str(j),"Done")

prediction = np.zeros((1,n))
for i in range(n):
    val = np.max(testdatafreq[i])
    ties = [(j,testdatascores[i][j]) for j in range(5) if testdatafreq[i][j]==val]

    l1,l2 = zip(*ties)
    val2 = np.max(l2)
    
    ans = [l1[z] for z in range(len(l1)) if l2[z]==val2]
    prediction[0][i] = ans[0]

endtime = time.time()

correctcnt = 0
incorrectcnt = 0

for i in range(len(testlabels)):
    if testlabels[i][0]==prediction[0][i]:
        correctcnt+=1
    else:
        incorrectcnt+=1

accuracy = (100*correctcnt)/(correctcnt+incorrectcnt)

print("The time taken using CVXOPT for multiclass is = {:2.3f}sec".format(endtime-sttime)) 
print("Accuracy is",accuracy,"%")