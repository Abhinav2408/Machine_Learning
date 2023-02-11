import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#taking data input without any headers
X = np.array([])
y = np.array([])
Xtest = np.array([])


if(len(sys.argv)==1):
    X = np.array(pd.read_csv('X.csv',header=None))
    y = np.array(pd.read_csv('Y.csv',header=None))
    Xtest = X.copy()

else:
    X = np.array(pd.read_csv(sys.argv[1] + '/X.csv',header=None))
    y = np.array(pd.read_csv(sys.argv[1] + '/Y.csv',header=None))
    Xtest = np.array(pd.read_csv(sys.argv[2] + '/X.csv',header=None))

#normalization of data


X = (X-np.mean(X))/np.std(X)
Xtest = (Xtest - np.mean(Xtest))/np.std(Xtest)


#adding a column for intercept part

X = np.hstack([np.ones((np.shape(X)[0],1)),X])
Xtest = np.hstack([np.ones((np.shape(Xtest)[0],1)),Xtest])

#defining theta

theta = np.array([[0],[0]])


#cost finding function, i.e. J(theta)

def findCost(y,X,theta):
    m = np.shape(X)[0]
    error = y - np.matmul(X,theta)
    return (np.dot(np.transpose(error),error))/(2*m)


#finding gradient of J(theta)

def deltaJ(y,X,theta):
    m = np.shape(X)[0]
    return np.matmul(np.transpose(X), np.matmul(X,theta) - y)/m




#initializing some lists for future plotting

costList = np.array([])
thetaList0 = np.array([])
thetaList1 = np.array([])


#defining a learning rate

learningRate = 0.1


#Gradient descent function


def GradDescent(y,X,theta,learningRate, costList, thetaList0,thetaList1):

    #initializing a starting cost
    prevcost = 1e5
    while True:


        cost = findCost(y,X,theta)
        thetaList0 = np.append(thetaList0,theta[0])
        thetaList1 = np.append(thetaList1,theta[1])
        costList = np.append(costList,cost)
        

        #breaking condition defined

        if abs((cost-prevcost).item())<1e-10:
            break

        #modifying theta

        theta = theta - (learningRate)*(deltaJ(y,X,theta))
        
        prevcost = cost

    return theta,costList,thetaList0,thetaList1



#calling gradient descent

newtheta,costList,thetaList0,thetaList1 = GradDescent(y,X,theta,learningRate,costList,thetaList0,thetaList1)


#the actual theta is found using the next comment. Used this to verify precision and accuracy
#correcttheta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y)


ytest = np.matmul(Xtest,newtheta)
ytestmat = np.mat(ytest)
with open('result_1.txt','wb') as f:
    for line in ytestmat:
        np.savetxt(f, line, fmt='%.10f')



#plotting the data

from matplotlib.animation import FuncAnimation


#2d plot using simple plt

def graph2d(y,X,theta,lr):
    
    xdata = X[:,1]
    ydata = np.matmul(X,theta)

    plt.title("Learning Rate Used = " + str(lr))
    plt.xlabel("Acidity of Wine")
    plt.ylabel("Density of Wine")
    plt.scatter(xdata,y,label="Data points",color='red',marker='.')
    plt.plot(xdata,ydata,label="Regression Line",color='blue')
    plt.legend()
    plt.show()
    



#3d plot a little bit complex due to animations involved

def graph3d(y,X,thetaList0,thetaList1,costList,lr):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    
    #defining a theta dataset
    theta0 = np.array([np.linspace(-2,2,100)])
    theta1 = np.array([np.linspace(-2,2,100)])
    Theta0,Theta1 = np.meshgrid(theta0,theta1)
    n = np.shape(theta0)[1]


    #finding cost for each theta combination

    Jtheta = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            para = np.array([[Theta0[i,j]], [Theta1[i,j]]])
            Jtheta[i,j] = findCost(y,X,para)


    #creating a 3d plot for the surface
    ax.plot_surface(Theta0,Theta1,Jtheta)


    #scattering our original dataset using animations in form of red dots

    graph = ax.scatter([], [], [], marker='o', color='red')
    graph.set_alpha(1)

    #storing the values to be scattered
    xvals = []
    yvals = []
    zvals = []

    #defining the animation

    def animator(i):
        xvals.append(thetaList0[i])
        yvals.append(thetaList1[i])
        zvals.append(costList[i])
        graph._offsets3d = (xvals, yvals, zvals)
        return graph
    
    plt.title("Learning Rate Used = " + str(lr))

    #calling the FuncAnimation function
    anim = FuncAnimation(fig, animator, frames=np.arange(0, 100), interval=200, repeat_delay=1000, blit=False)
    plt.show()




def graphContour(y,X,thetaList0,thetaList1,costList,lr):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)


    #defining a theta dataset
    theta0 = np.array([np.linspace(-2,2,100)])
    theta1 = np.array([np.linspace(-2,2,100)])
    Theta0,Theta1 = np.meshgrid(theta0,theta1)
    n = np.shape(theta0)[1]


    #finding the cost for each theta combination

    Jtheta = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            para = np.array([[Theta0[i,j]], [Theta1[i,j]]])
            Jtheta[i,j] = findCost(y,X,para)


    
    #making the basic contours
    ax.contour(Theta0,Theta1,Jtheta)

    graph, = ax.plot([],[],marker='o',color='red')
    xvals = []
    yvals = []

    def initializer():
        graph.set_data([],[])
        return graph,

    def animator(i):
        xvals.append(thetaList0[i])
        yvals.append(thetaList1[i])
        graph.set_data(xvals,yvals)
        return graph,

    plt.title("Learning Rate Used = " + str(lr))
    anim = FuncAnimation(fig,animator,init_func= initializer,frames=np.arange(0,100),interval=200,repeat_delay=1000,blit = False)

    plt.show()



#calling all the required functions
print("I took 0.1 as the learning rate")

print("For learning rate",0.1, "theta is")
print(newtheta)
print("Stopping Criteria : Change in Cost function less than 1e-10")
graph2d(y,X,newtheta,learningRate)
graph3d(y,X,thetaList0,thetaList1,costList,learningRate)
graphContour(y,X,thetaList0,thetaList1,costList,learningRate)

for eta in [0.025,0.001]:
    print("For learning rate",eta,"theta is")
    costList1 = np.array([])
    thetaList01 = np.array([])
    thetaList11 = np.array([])
    newtheta1,costList1,thetaList01,thetaList11 = GradDescent(y,X,theta,eta,costList1,thetaList01,thetaList11)
    print(newtheta1)
    print("Stopping Criteria : Change in Cost function less than 1e-10")
    graphContour(y,X,thetaList01,thetaList11,costList1,eta)
