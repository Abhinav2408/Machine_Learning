import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#taking data input


X = np.array([])
y = np.array([])

Xtest = np.array([])

if(len(sys.argv)==1):
    X = np.array(pd.read_csv('logisticX.csv',header=None))
    y = np.array(pd.read_csv('logisticY.csv',header=None))
    Xtest = X.copy()

else:
    X = np.array(pd.read_csv(sys.argv[1] + '/X.csv',header=None))
    y = np.array(pd.read_csv(sys.argv[1] + '/Y.csv',header=None))
    Xtest = np.array(pd.read_csv(sys.argv[2] + '/X.csv',header=None))



#data normalization
#m = np.shape(X)[0]

mean1 = np.mean(X[:,0])
var1 = np.std(X[:,0])
mean2 = np.mean(X[:,1])
var2 = np.std(X[:,1])

X[:,0] = (X[:,0] - mean1)/var1
X[:,1] = (X[:,1] - mean2)/var2

X = np.hstack([np.ones((np.shape(X)[0],1)),X])

mean1test = np.mean(Xtest[:,0])
var1test = np.std(Xtest[:,0])
mean2test = np.mean(Xtest[:,1])
var2test = np.std(Xtest[:,1])

Xtest[:,0] = (Xtest[:,0] - mean1test)/var1test
Xtest[:,1] = (Xtest[:,1] - mean2test)/var2test

Xtest = np.hstack([np.ones((np.shape(Xtest)[0],1)),Xtest])


#defining theta

theta = np.array([[0],[0],[0]])


def y_hypothesis(X,theta):
    temp = np.matmul(X,theta)
    return 1/(1+np.exp(-temp))

def llhood(y,X,theta):
    hyp = y_hypothesis(X,theta)
    return np.matmul(np.transpose(y),np.log(hyp)) + np.matmul(np.transpose(1-y),np.log(1-hyp))

def deltall(y,X,theta):
    hyp = y_hypothesis(X,theta)
    return np.matmul(np.transpose(X), hyp-y)


i = 0
def GradDescent(y,X,theta,i):
    prevcost = 1e5
    condition = True
    while condition and i<=200:
        i=i+1
        cost = (llhood(y,X,theta))
        if abs((cost-prevcost))<1e-10:
            condition = False

        yhyp = y_hypothesis(X,theta)
        diag = np.identity(np.shape(X)[0]) * np.matmul(np.transpose(yhyp),1-yhyp)
        hessian = np.matmul(np.transpose(X),np.matmul(diag,X))
        theta = theta - np.matmul(np.linalg.inv(hessian),deltall(y,X,theta))
        prevcost = cost

    return theta,i

theta,i = GradDescent(y,X,theta,i)


#getting output on testdata

ytest = y_hypothesis(Xtest,theta)
for i in range(0,np.shape(ytest)[0]):
    if(ytest[i]>0.5):
        ytest[i] = 1
    else:
        ytest[i] = 0

ytestmat = np.mat(ytest)
with open('result_3.txt','wb') as f:
    for line in ytestmat:
        np.savetxt(f, line, fmt='%.0f')

print(theta)
print("Iterations taken:",i)


def makeGraph(y,X,theta):
    type1 = []
    type2 = []

    for i in range(len(y)):
        if y[i]==1:
            type1.append([X[i,1],X[i,2]])
        else:
            type2.append([X[i,1],X[i,2]])
    
    type1 = np.array(type1)
    type2 = np.array(type2)


    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Logistic Regression")
    plt.scatter(type1[:,0],type1[:,1],label="Y = 1",color='red',marker='.')
    plt.scatter(type2[:,0],type2[:,1],label="Y = 0",color='green',marker='+')


    xdata = np.arange(-5,5,0.1)
    ydata = -(theta[0] + theta[1]*xdata)/theta[2]
    plt.plot(xdata,ydata,label="Decision Boundary",color="blue")
    plt.legend()
    plt.show()

makeGraph(y,X,theta)