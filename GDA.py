import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



#getting data input


X = np.array([])
y = np.array([])
Xtest = np.array([])



if(len(sys.argv)==1):
    X = np.loadtxt('q4x.dat')
    y = np.loadtxt('q4y.dat',dtype=str).reshape(-1,1)
    Xtest = X.copy()

else:
    X = np.array(pd.read_csv(sys.argv[1] + '/X.csv',header=None))
    y = np.array(pd.read_csv(sys.argv[1] + '/Y.csv',header=None))
    Xtest = np.array(pd.read_csv(sys.argv[2] + '/X.csv',header=None))


#modifying y

def changey(y):
    a = np.zeros(np.shape(y))
    for i in range(len(y)):
        if(y[i,0]=='Alaska'):
            a[i,0] = 0
        else:
            a[i,0] = 1
    return a

#normalization

X = (X - np.mean(X))/np.std(X)
Xtest = (Xtest - np.mean(Xtest))/np.std(Xtest)
y = changey(y)

m = np.shape(y)[0]

phi = np.sum([y==1])/m


#getting means and covariances

def getvalues(X,y):
    m = np.shape(y)[0]
    canadasize = np.sum(y)
    alaskasize = m - canadasize
    mean0 = np.sum((X*(y==0)),axis=0)/alaskasize
    mean1 = np.sum((X*(y==1)),axis=0)/canadasize

    Z =np.zeros(np.shape(X))
    for i in range(len(y)):
        if y[i,0]==0:
            Z[i,:] = X[i,:] - mean0
        else:
            Z[i,:] = X[i,:] - mean1

    sigma = np.matmul(np.transpose(Z),Z)/m
    sigma0 = np.matmul(np.transpose(Z*(y==0)),Z*(y==0))/alaskasize
    sigma1 = np.matmul(np.transpose(Z*(y==1)),Z*(y==1))/canadasize

    return mean0,mean1,sigma,sigma0,sigma1

mu0,mu1,sigma,sigma0,sigma1 = getvalues(X,y)


#defining probability

def prob(data,mu,sigma):
    x = np.transpose(data)
    return ((1/(2*3.142857))*math.sqrt(np.linalg.det(sigma)))*math.exp(-0.5*(np.matmul(np.transpose(x-mu),np.matmul(np.linalg.inv(sigma),x-mu))))


ytest = np.array([])

#getting and storing output

for i in range(0,np.shape(Xtest)[0]):
    data = Xtest[i,:]
    p0 = prob(data,mu0,sigma0)*(1-phi)
    p1 = prob(data,mu1,sigma1)*phi

    if(p0>p1):
        ytest = np.append(ytest,'Alaska')
    else:
        ytest = np.append(ytest,'Canada')


ytest = ytest.reshape(-1,1)

ytestmat = np.mat(ytest)
with open('result_4.txt','wb') as f:
    for line in ytestmat:
        np.savetxt(f, line, fmt='%s')

#plotting data



def scatterdata(y,X):
    type1 = []
    type2 = []

    for i in range(len(y)):
        if y[i,0]==1:
            type1.append([X[i,0],X[i,1]])
        else:
            type2.append([X[i,0],X[i,1]])
    
    type1 = np.array(type1)
    type2 = np.array(type2)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Scattering Data")
    plt.scatter(type1[:,0],type1[:,1],label="Y = Canada",color='red',marker='.')
    plt.scatter(type2[:,0],type2[:,1],label="Y = Alaska",color='green',marker='+')
    plt.legend()
    plt.show()

scatterdata(y,X)

def linearscatter(y,X,mu0,mu1,sigma,phi):

    #making linear equation
    con = math.log(phi/(1-phi))
    sigmainv = np.linalg.inv(sigma)
    c = con - 0.5*(np.matmul(np.matmul(np.transpose(mu1),sigmainv), mu1) - np.matmul(np.matmul(np.transpose(mu0),sigmainv), mu0))
    xfactor = np.matmul(mu1-mu0,sigmainv)

    xdata = np.array([np.min(X[:,0]),np.max(X[:,0])])
    ydata = -(xdata*xfactor[0] + c)/xfactor[1]

    type1 = []
    type2 = []

    for i in range(len(y)):
        if y[i,0]==1:
            type1.append([X[i,0],X[i,1]])
        else:
            type2.append([X[i,0],X[i,1]])
    
    type1 = np.array(type1)
    type2 = np.array(type2)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Linear GDA")
    plt.scatter(type1[:,0],type1[:,1],label="Y = Canada",color='red',marker='.')
    plt.scatter(type2[:,0],type2[:,1],label="Y = Alaska",color='green',marker='+')
    plt.plot(xdata,ydata,label="Linear Decision Boundary",color='blue')
    plt.legend()
    plt.show()

linearscatter(y,X,mu0,mu1,sigma,phi)





def quadraticscatter(y,X,mu0,mu1,sigma,sigma0,sigma1,phi):


    #making linear and quadratic equation
    con = math.log(phi/(1-phi))
    sigmainv = np.linalg.inv(sigma)

    clinear = con - 0.5*(np.matmul(np.matmul(np.transpose(mu1),sigmainv), mu1) - np.matmul(np.matmul(np.transpose(mu0),sigmainv), mu0))

    sigma0inv = np.linalg.inv(sigma0)
    sigma1inv = np.linalg.inv(sigma1)
    

    xtermlinear = np.matmul(mu1-mu0,sigmainv)

    xdatalinear = np.array([np.min(X[:,0]),np.max(X[:,0])])
    ydatalinear = -(xdatalinear*xtermlinear[0] + clinear)/xtermlinear[1]


    
    cquad = con - 0.5*(np.matmul(np.matmul(np.transpose(mu1),sigma1inv), mu1) - np.matmul(np.matmul(np.transpose(mu0),sigma0inv), mu0))
    x2termquad = 0.5*(sigma0inv-sigma1inv)
    xtermquad = np.matmul(sigma1inv,mu1) - np.matmul(sigma0inv,mu0)

    x1dat = np.linspace(np.min(X[:,0]),np.max(X[:,0]),500)
    x2dat = np.linspace(np.min(X[:,1]),np.max(X[:,1]),500)

    xdataquad = []
    ydataquad = []

    for i in x1dat:
        for j in x2dat:
            xx = np.array([[i],[j]])
            llval = np.matmul(np.transpose(xx), np.matmul(x2termquad,xx)) + np.matmul(np.transpose(xx),xtermquad) + cquad
            if abs(llval) < 0.001:
                xdataquad.append(i)
                ydataquad.append(j)

    type1 = []
    type2 = []

    for i in range(len(y)):
        if y[i,0]==1:
            type1.append([X[i,0],X[i,1]])
        else:
            type2.append([X[i,0],X[i,1]])
    
    type1 = np.array(type1)
    type2 = np.array(type2)


    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Linear and Quadratic Comparision")
    plt.scatter(type1[:,0],type1[:,1],label="Y = Canada",color='red',marker='.')
    plt.scatter(type2[:,0],type2[:,1],label="Y = Alaska",color='green',marker='+')
    
    plt.plot(xdataquad,ydataquad,label="Quadratic Decision Boundary",color='blue')
    plt.plot(xdatalinear,ydatalinear,label="Linear Decision Boundary",color='black')

    plt.legend()
    plt.show()


quadraticscatter(y,X,mu0,mu1,sigma,sigma0,sigma1,phi)