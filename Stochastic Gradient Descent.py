import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

#defining number of data points to be generated

n = 1000000

#intercept
x0 = np.ones((n,1))

#using built in normal distribution function

x1 = np.random.normal(3,2,n).reshape(-1,1)
x2 = np.random.normal(-1,2,n).reshape(-1,1)

#combining the data columns into a matrix
xData = np.hstack([x0,x1,x2])


#defining a normally distributed noise
noise = np.random.normal(0,math.sqrt(2),n).reshape(-1,1)

#defining theta
theta = np.array([[3],[1],[2]])

#defining y, and adding the noise
yData = np.matmul(xData,theta) + noise


#combining total data, and shuffling
netData = np.hstack([xData,yData])
np.random.shuffle(netData)


#breaking the shuffled data again into x and y
xData = netData[:,0:3]
yData = netData[:,3].reshape(-1,1)



#inputting the data to be tested
X = np.array([])
if(len(sys.argv)==1):
    X = np.array( pd.read_csv('q2test.csv'))
else:
    X = np.array( pd.read_csv(sys.argv[1] + '/X.csv',header=None))



#modifying the data
X = np.hstack([np.ones((np.shape(X)[0],1)),X])
y = np.matmul(X,theta)


#defining cost function

def findCost(y,X,theta):
    m = np.shape(X)[0]
    error = y - np.matmul(X,theta)
    return (np.dot(np.transpose(error),error))/(2*m)


#defining gradient of cost function

def deltaJ(y,X,theta):
    m = np.shape(X)[0]
    return np.matmul(np.transpose(X), np.matmul(X,theta) - y)/m


#defining batches

batch = np.array([10000,100,1,1000000])
avgcheck = np.array([10,100,1000,1])





for case in range(len(batch)):

    #defining initial theta, learning rate and batch size
    #we take batch size from the above defined array


    newtheta = np.array([[0],[0],[0]])
    learningrate = 0.001
    batchSize = batch[case]
    r = avgcheck[case]


    #defining seperated batches of defined batch size

    chosenBatch = [ (xData[i:i+batchSize,:], yData[i:i+batchSize]) for i in range(0,n,batchSize)]


    #defining a thetalist for plotting
    theta0List = np.array([0])
    theta1List = np.array([0])
    theta2List = np.array([0])


    print("Case : Batch Size =",batchSize)
    print("Please wait 30secs max")


    iter = 0
    curravg=0
    avgset = np.array([0])
    

    while(True and iter <= 1000000/(math.sqrt(batchSize))):
        iter+=1
        costinit = findCost(yData,xData,newtheta)
        cnt = 0
        
        diff = np.array([[10],[10],[10]])
        for currbatch in chosenBatch:
            
            xbatch = currbatch[0]
            ybatch = currbatch[1]

            curravg += findCost(ybatch,xbatch,newtheta)

            if(cnt!=0 and cnt%r==0):
                curravg = curravg/r
                avgset = np.append(avgset,curravg)
                curravg = 0
                if(abs(avgset[-1]-avgset[-2]) < 0.05):
                    break

                



            newtheta = newtheta - learningrate*deltaJ(ybatch,xbatch,newtheta)
            theta0List = np.append(theta0List,newtheta[0])
            theta1List = np.append(theta1List,newtheta[1])
            theta2List = np.append(theta2List,newtheta[2])
            cnt +=1

            
        costfinal = findCost(yData,xData,newtheta)
        if(abs(costfinal-costinit).item() < 1e-8):
            break

    #best case was with 10000 batch size
    if(case==0):
        ytest = np.matmul(X,newtheta)
        ytestmat = np.mat(ytest)
        with open('result_2.txt','wb') as f:
            for line in ytestmat:
                np.savetxt(f, line, fmt='%.15f')


    print(newtheta)
    print("Iterations =", iter)
    # print("Using test theta, error =", findCost(y,X,theta).item())
    # print("Using learned theta, error =", findCost(y,X,newtheta).item())







    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('theta2')
    ax.set_title("Changes in Theta Values")
    graph = ax.scatter([], [], [], marker='.', color='blue')
    ax.set_xlim(np.amin(theta0List), np.amax(theta0List))
    ax.set_ylim(np.amin(theta1List), np.amax(theta1List))
    ax.set_zlim(np.amin(theta2List), np.amax(theta2List))
    xvals = []
    yvals = []
    zvals = []
    graph.set_alpha(1)
    def animator(i):
        xvals.append(theta0List[i])
        yvals.append(theta1List[i])
        zvals.append(theta2List[i])
        graph._offsets3d = (xvals, yvals, zvals)
        return graph
    anim = FuncAnimation(fig, animator, frames=np.arange(0, theta0List.shape[0], 20), interval=10, repeat_delay=1000, blit=False)
    plt.show()

print("Ended the case of 1 million batch size early as it would have taken too long to process. Maximum iterations taken = 1001")
