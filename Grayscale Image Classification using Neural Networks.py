from math import sqrt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import sys

trainaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/fmnist_train.csv"
testaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/fmnist_test.csv"
outputpath = ""
part = 'c'

if(len(sys.argv)!=1):
    trainaddress = sys.argv[1]
    testaddress = sys.argv[2]
    outputpath = sys.argv[3]
    part = sys.argv[4]


file = open(outputpath+"/"+part+".txt",'a')
sys.stdout = file

traindata = np.array(pd.read_csv(trainaddress,header=None))
testdata = np.array(pd.read_csv(testaddress,header = None))

#shuffle the data
np.random.shuffle(traindata)

xtrain = traindata[:,:-1]
xtest = testdata[:,:-1]

#normalized data
xtrain = xtrain/(np.max(xtrain))
xtest = xtest/(np.max(xtest))
xtrain = np.hstack([np.ones((np.shape(xtrain)[0],1)),xtrain])
xtest = np.hstack([np.ones((np.shape(xtest)[0],1)),xtest])
xvalidation = xtrain[:int((20*np.shape(xtrain)[0])/100),:]
#one hot encoded labels
labeltrain = np.eye(10)[traindata[:,-1]]
labeltest = np.eye(10)[testdata[:,-1]]
labelval = np.eye(10)[traindata[:int((20*np.shape(xtrain)[0])/100),-1]]


#used 20% data for validation in sgd

n = np.shape(xtrain)[1]
M = 100
r = 10
minibatches = [(xtrain[i:i+M,:],labeltrain[i:i+M]) for i in range(0,np.shape(xtrain)[0],M)]


def relu(val):
    return np.maximum(0.0,val)

def delrelu(val):
    val[val<=0] = 0
    val[val>0] = 1
    return val

def sigmoid(val):
    return 1/(1+np.exp(-val))

def delsigmoid(val):
    return np.multiply(val,1-val)

#number of classes are 10

def gettheta(n,layers):
    #hidden layers are represented in number of perceptrons
    theta = []
    for i in range(len(layers)+1):
        
        d0 = 0
        d1 = 0
        
        if i==0:
            d0 = n
            d1 = layers[i]
        
        elif i==len(layers):
            d0 = layers[i-1]+1
            d1 = r
            #number of classes
            #last layer has 10 perceptrons
        else:
            d0 = layers[i-1]+1
            d1 = layers[i]

        thetai = np.random.normal(0,0.1,(d0,d1))
        theta.append(thetai)

    return theta





def sig_forward(xvals, params):
    forward = []
    forward.append(xvals)

    for i in range(len(params)-1):
        forward.append(sigmoid(np.dot(forward[i],params[i])))
        forward[i+1] = np.hstack([np.ones((np.shape(forward[i+1])[0],1)),forward[i+1]])

    forward.append(sigmoid(np.dot(forward[len(params)-1],params[len(params)-1])))
    return forward


def relu_forward(xvals,params):
    forward = []
    forward.append(xvals)

    for i in range(len(params)-1):
        forward.append(relu(np.dot(forward[i],params[i])))
        forward[i+1] = np.hstack([np.ones((np.shape(forward[i+1])[0],1)),forward[i+1]])

    forward.append(sigmoid(np.dot(forward[len(params)-1],params[len(params)-1])))
    return forward

def cost_mse_sig(xval,params,y):
    forward = sig_forward(xval,params)
    return (1/(2*np.shape(xval)[0]))*np.sum((y-forward[-1])**2)

def cost_mse_relu(xval,params,y):
    forward = relu_forward(xval,params)
    return (1/(2*np.shape(xval)[0]))*np.sum((y-forward[-1])**2)

def cost_bce_sig(xval,params,y):
    forward = sig_forward(xval,params)
    return (-1/(np.shape(xval)[0]))*np.sum((y*np.log(forward[-1])) + (1-y)*np.log(1-forward[-1]))

def cost_bce_relu(xval,params,y):
    forward = relu_forward(xval,params)
    return (-1/(np.shape(xval)[0]))*np.sum((y*np.log(forward[-1])) + (1-y)*np.log(1-forward[-1]))




def sig_backward(y,params,forward,m):
    backward = [0.0]*len(forward)

    for i in range(len(forward)-1,0,-1):
        if i==len(forward)-2:
            backward[i] = np.dot(backward[i+1],np.transpose(params[i]))*delsigmoid(forward[i])
        
        elif i==len(forward)-1:
            backward[i] = ((1/m)*(y-forward[i]))*delsigmoid(forward[i])

        else:
            nobias = backward[i+1][:,1:]
            backward[i] = np.dot(nobias,np.transpose(params[i]))*delsigmoid(forward[i])

    return backward


def relu_backward(y,params,forward,m):
    backward = [0.0]*len(forward)

    for i in range(len(forward)-1,0,-1):
        if i==len(forward)-2:
            backward[i] = np.dot(backward[i+1],np.transpose(params[i]))*delrelu(forward[i])
        
        elif i==len(forward)-1:
            backward[i] = ((1/m)*(y-forward[i]))*delsigmoid(forward[i])

        else:
            nobias = backward[i+1][:,1:]
            backward[i] = np.dot(nobias,np.transpose(params[i]))*delrelu(forward[i])

    return backward


def entropy_backward_relu(y,params,forward,m):
    backward = [0.0]*len(forward)

    for i in range(len(forward)-1,0,-1):
        if i==len(forward)-2:
            backward[i] = np.dot(backward[i+1],np.transpose(params[i]))*delrelu(forward[i])
        
        elif i==len(forward)-1:
            backward[i] = ((1/m)*((y/forward[i])-((1-y)/(1-forward[i]))))*delsigmoid(forward[i])

        else:
            nobias = backward[i+1][:,1:]
            backward[i] = np.dot(nobias,np.transpose(params[i]))*delrelu(forward[i])

    return backward


def getacc_sig(xvals,params,yvals):
    prediction = sig_forward(xvals,params)[-1]
    for i in range(len(prediction)):
        maxidx = np.argmax(prediction[i])
        for j in range(10):
            if j==maxidx:
                prediction[i][j] = 1
            else:
                prediction[i][j] = 0
    accuracy = np.sum(np.multiply(prediction,yvals))
    accuracy = accuracy*100
    accuracy = accuracy/np.shape(xvals)[0]

    return accuracy


def getacc_relu(xvals,params,yvals):
    prediction = relu_forward(xvals,params)[-1]
    for i in range(len(prediction)):
        maxidx = np.argmax(prediction[i])
        for j in range(r):
            if j==maxidx:
                prediction[i][j] = 1
            else:
                prediction[i][j] = 0
    accuracy = np.sum(np.multiply(prediction,yvals))
    accuracy = accuracy*100
    accuracy = accuracy/np.shape(xvals)[0]

    return accuracy


def confusion_sig(xvals,params,yvals):
    conf = np.zeros((r,r))
    prediction = sig_forward(xvals,params)[-1]
    predidx = []
    yidx = []
    for i in range(len(prediction)):
        predidx.append(np.argmax(prediction[i]))
        yidx.append(np.argmax(yvals[i]))

    for i in range(len(yidx)):
        conf[predidx[i]][yidx[i]]+=1

    return conf


def confusion_relu(xvals,params,yvals):
    conf = np.zeros((r,r))
    prediction = relu_forward(xvals,params)[-1]
    predidx = []
    yidx = []
    for i in range(len(prediction)):
        predidx.append(np.argmax(prediction[i]))
        yidx.append(np.argmax(yvals[i]))

    for i in range(len(yidx)):
        conf[predidx[i]][yidx[i]]+=1

    return conf






if part=='a':
    #hidden layers = 1
    learning_rate = 0.1
    hidden_units = 10
    params = gettheta(n,[hidden_units])

    epoch = 1
    stopper=0
    currcost = cost_mse_sig(xvalidation,params,labelval)
    while(True):

        batchcount = 0

        for batch in minibatches:
            xvals = batch[0]
            yvals = batch[1]

            forwardvals = sig_forward(xvals,params)
            backwardvals = sig_backward(yvals,params,forwardvals,100)

            for i in range(len(params)):
                if i <len(params)-1:
                    params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                else:
                    params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

            batchcount+=1
        
        epoch+=1
        fincost = cost_mse_sig(xvalidation,params,labelval)
        if(abs(fincost-currcost)<1e-5):
            stopper+=1
        else:
            stopper=0
            
        if(stopper==10):
            break

        currcost = fincost

    trainaccuracy = getacc_sig(xtrain,params,labeltrain)
    testaccuracy = getacc_sig(xtest,params,labeltest)

    # print("Training accuracy is",trainaccuracy)
    # print("Testing accuracy is",testaccuracy)


elif part=='b':
    timetaken = []
    trainacc = []
    testacc = []
    hiddenlayerunits = [5,10,15,20,25]

    for units in hiddenlayerunits:

        sttime = time.time()
        learning_rate = 0.1
        params = gettheta(n,[units])

        epoch = 1
        stopper=0
        currcost = cost_mse_sig(xvalidation,params,labelval)
        while(True):

            batchcount = 0

            for batch in minibatches:
                xvals = batch[0]
                yvals = batch[1]

                forwardvals = sig_forward(xvals,params)
                backwardvals = sig_backward(yvals,params,forwardvals,100)

                for i in range(len(params)):
                    if i <len(params)-1:
                        params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                    else:
                        params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                batchcount+=1
            
            epoch+=1
            fincost = cost_mse_sig(xvalidation,params,labelval)
            if(abs(fincost-currcost)<1e-5):
                stopper+=1
            else:
                stopper=0
                
            if(stopper==10):
                break

            currcost = fincost

        trainaccuracy = getacc_sig(xtrain,params,labeltrain)
        testaccuracy = getacc_sig(xtest,params,labeltest)
        totaltime = time.time() - sttime
        confusionmatrix = confusion_sig(xtest,params,labeltest)

        trainacc.append(trainaccuracy)
        testacc.append(testaccuracy)
        timetaken.append(totaltime)

        print("----------Using",units,"Units-----------")
        print("Training accuracy is",trainaccuracy)
        print("Testing accuracy is",testaccuracy)
        print("Time taken is",totaltime)
        print("Confusion matrix over test set is")
        with np.printoptions(threshold=np.inf):
            print(confusionmatrix)
    
    plt.title("Accuracies vs Number of Perceptrons")
    plt.plot(hiddenlayerunits,trainacc,color='red',label='Training Accuracy')
    plt.plot(hiddenlayerunits,testacc,color='blue',label='Testing Accuracy')
    plt.ylabel("Accuracies")
    plt.xlabel("Number of Perceptrons")
    plt.legend()
    plt.savefig(outputpath+"/neural_b_accuracies.png")
    plt.clf()


    plt.title("Training Time vs Number of Perceptrons")
    plt.plot(hiddenlayerunits,timetaken,color='green')
    plt.ylabel("Time Taken")
    plt.xlabel("Number of Perceptrons")
    plt.savefig(outputpath+"/neural_b_time.png")
    plt.clf()




elif part=='c':
    timetaken = []
    trainacc = []
    testacc = []
    hiddenlayerunits = [5,10,15,20,25]

    for units in hiddenlayerunits:

        sttime = time.time()
        lr = 0.1
        params = gettheta(n,[units])

        epoch = 1
        stopper=0
        currcost = cost_mse_sig(xvalidation,params,labelval)
        while(True):
            learning_rate = lr/sqrt(epoch)
            batchcount = 0

            for batch in minibatches:
                xvals = batch[0]
                yvals = batch[1]

                forwardvals = sig_forward(xvals,params)
                backwardvals = sig_backward(yvals,params,forwardvals,100)

                for i in range(len(params)):
                    if i <len(params)-1:
                        params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                    else:
                        params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                batchcount+=1
            
            epoch+=1
            fincost = cost_mse_sig(xvalidation,params,labelval)
            if(abs(fincost-currcost)<1e-5):
                stopper+=1
            else:
                stopper=0
                
            if(stopper==10):
                break

            currcost = fincost

        trainaccuracy = getacc_sig(xtrain,params,labeltrain)
        testaccuracy = getacc_sig(xtest,params,labeltest)
        totaltime = time.time() - sttime
        confusionmatrix = confusion_sig(xtest,params,labeltest)


        trainacc.append(trainaccuracy)
        testacc.append(testaccuracy)
        timetaken.append(totaltime)

        print("----------Using",units,"Units-----------")
        print("Training accuracy is",trainaccuracy)
        print("Testing accuracy is",testaccuracy)
        print("Time taken is",totaltime)
        print("Confusion matrix over test set is")
        with np.printoptions(threshold=np.inf):
            print(confusionmatrix)

    
    plt.title("Accuracies vs Number of Perceptrons")
    plt.plot(hiddenlayerunits,trainacc,color='red',label='Training Accuracy')
    plt.plot(hiddenlayerunits,testacc,color='blue',label='Testing Accuracy')
    plt.ylabel("Accuracies")
    plt.xlabel("Number of Perceptrons")
    plt.legend()
    plt.savefig(outputpath+"/neural_c_accuracies.png")
    plt.clf()


    plt.title("Training Time vs Number of Perceptrons")
    plt.plot(hiddenlayerunits,timetaken,color='green')
    plt.ylabel("Time Taken")
    plt.xlabel("Number of Perceptrons")
    plt.savefig(outputpath+"/neural_c_time.png")
    plt.clf()


elif part=='d':
    
    activation = ["Sigmoid","ReLU"]

    for fn in activation:
        if fn=="Sigmoid":
            #hidden layers = 2
            lr = 0.1
            hidden_units = 100
            params = gettheta(n,[hidden_units,hidden_units])

            epoch = 1
            stopper=0
            currcost = cost_mse_sig(xvalidation,params,labelval)
            while(True):

                batchcount = 0
                learning_rate = lr/sqrt(epoch)
                for batch in minibatches:
                    xvals = batch[0]
                    yvals = batch[1]

                    forwardvals = sig_forward(xvals,params)
                    backwardvals = sig_backward(yvals,params,forwardvals,100)

                    for i in range(len(params)):
                        if i <len(params)-1:
                            params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                        else:
                            params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                    batchcount+=1
                
                epoch+=1
                fincost = cost_mse_sig(xvalidation,params,labelval)
                if(abs(fincost-currcost)<1e-4):
                    stopper+=1
                else:
                    stopper=0
                    
                if(stopper==10):
                    break

                currcost = fincost

            trainaccuracy = getacc_sig(xtrain,params,labeltrain)
            testaccuracy = getacc_sig(xtest,params,labeltest)
            confusionmatrix = confusion_sig(xtest,params,labeltest)

            print("Using Sigmoid Activation Function")
            print("Training accuracy is",trainaccuracy)
            print("Testing accuracy is",testaccuracy)
            print("Confusion matrix over test set is")
            with np.printoptions(threshold=np.inf):
                print(confusionmatrix)

        else:
            #hidden layers = 2
            lr = 0.1
            hidden_units = 100
            params = gettheta(n,[hidden_units,hidden_units])

            epoch = 1
            stopper=0
            currcost = cost_mse_relu(xvalidation,params,labelval)
            while(True):

                batchcount = 0
                learning_rate = lr/sqrt(epoch)
                for batch in minibatches:
                    xvals = batch[0]
                    yvals = batch[1]

                    forwardvals = relu_forward(xvals,params)
                    backwardvals = relu_backward(yvals,params,forwardvals,100)

                    for i in range(len(params)):
                        if i <len(params)-1:
                            params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                        else:
                            params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                    batchcount+=1
                
                epoch+=1
                fincost = cost_mse_relu(xvalidation,params,labelval)

                if(abs(fincost-currcost)<1e-4):
                    stopper+=1
                else:
                    stopper=0
                    
                if(stopper==10):
                    break

                currcost = fincost

            trainaccuracy = getacc_relu(xtrain,params,labeltrain)
            testaccuracy = getacc_relu(xtest,params,labeltest)
            confusionmatrix = confusion_relu(xtest,params,labeltest)

            print("Using ReLU Activation Function")
            print("Training accuracy is",trainaccuracy)
            print("Testing accuracy is",testaccuracy)
            print("Confusion matrix over test set is")
            with np.printoptions(threshold=np.inf):
                print(confusionmatrix)



elif part == 'e':
    activation = ["ReLU","Sigmoid"]
    sigtrain = []
    sigtest = []
    relutrain = []
    relutest = []
    
    hiddenlayers = [2,3,4,5]
    for fn in activation:
        for j in hiddenlayers:
            if fn=="Sigmoid":
                lr = 0.1
                hidden_units = 50
                params = gettheta(n,[hidden_units]*j)

                epoch = 1
                stopper=0
                currcost = cost_mse_sig(xvalidation,params,labelval)
                while(True):

                    batchcount = 0
                    learning_rate = lr/sqrt(epoch)
                    for batch in minibatches:
                        xvals = batch[0]
                        yvals = batch[1]

                        forwardvals = sig_forward(xvals,params)
                        backwardvals = sig_backward(yvals,params,forwardvals,100)

                        for i in range(len(params)):
                            if i <len(params)-1:
                                params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                            else:
                                params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                        batchcount+=1
                    
                    epoch+=1
                    fincost = cost_mse_sig(xvalidation,params,labelval)
                    if(abs(fincost-currcost)<1e-4):
                        stopper+=1
                    else:
                        stopper=0
                        
                    if(stopper==10):
                        break

                    currcost = fincost

                trainaccuracy = getacc_sig(xtrain,params,labeltrain)
                testaccuracy = getacc_sig(xtest,params,labeltest)

                sigtrain.append(trainaccuracy)
                sigtest.append(testaccuracy)

                print("Using Sigmoid Activation Function and number of layers =",j)
                print("Training accuracy is",trainaccuracy)
                print("Testing accuracy is",testaccuracy)

            else:
                lr = 0.1
                hidden_units = 50
                params = gettheta(n,[hidden_units]*j)

                epoch = 1
                stopper=0
                currcost = cost_mse_relu(xvalidation,params,labelval)
                while(True):

                    batchcount = 0
                    learning_rate = lr/sqrt(epoch)
                    for batch in minibatches:
                        xvals = batch[0]
                        yvals = batch[1]

                        forwardvals = relu_forward(xvals,params)
                        backwardvals = relu_backward(yvals,params,forwardvals,100)

                        for i in range(len(params)):
                            if i <len(params)-1:
                                params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                            else:
                                params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

                        batchcount+=1
                    
                    epoch+=1
                    fincost = cost_mse_relu(xvalidation,params,labelval)
                    if(abs(fincost-currcost)<1e-4):
                        stopper+=1
                    else:
                        stopper=0
                        
                    if(stopper==10):
                        break

                    currcost = fincost

                trainaccuracy = getacc_relu(xtrain,params,labeltrain)
                testaccuracy = getacc_relu(xtest,params,labeltest)

                relutrain.append(trainaccuracy)
                relutest.append(testaccuracy)


                print("Using ReLU Activation Function and number of layers =",j)
                print("Training accuracy is",trainaccuracy)
                print("Testing accuracy is",testaccuracy)


    plt.title("Training Accuracies vs Number of Layers")
    plt.plot(hiddenlayers,sigtrain,color='red',label='Sigmoid Training Accuracy')
    plt.plot(hiddenlayers,relutrain,color='blue',label='ReLU Training Accuracy')
    plt.ylabel("Accuracies")
    plt.xlabel("Number of Layers")
    plt.legend()
    plt.savefig(outputpath+"/neural_e_accuracies.png")
    plt.clf()


    plt.title("Testing Accuracies vs Number of Layers")
    plt.plot(hiddenlayers,sigtest,color='red',label='Sigmoid Testing Accuracy')
    plt.plot(hiddenlayers,relutest,color='blue',label='ReLU Testing Accuracy')
    plt.ylabel("Accuracies")
    plt.xlabel("Number of Layers")
    plt.legend()
    plt.savefig(outputpath+"/neural_e_accuracies.png")
    plt.clf()




elif part=='f':

    #max accuracy with 4 layers
    lr = 0.1
    hidden_units = 50
    params = gettheta(n,[hidden_units]*4)

    epoch = 1
    stopper=0
    currcost = cost_bce_relu(xvalidation,params,labelval)
    while(True):

        batchcount = 0
        learning_rate = lr/sqrt(epoch)
        for batch in minibatches:
            xvals = batch[0]
            yvals = batch[1]

            forwardvals = relu_forward(xvals,params)
            backwardvals = entropy_backward_relu(yvals,params,forwardvals,100)

            for i in range(len(params)):
                if i <len(params)-1:
                    params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])[:,1:]
                else:
                    params[i] += learning_rate*np.dot(np.transpose(forwardvals[i]),backwardvals[i+1])

            batchcount+=1
                
        epoch+=1
        fincost = cost_bce_relu(xvalidation,params,labelval)
        if(abs(fincost-currcost)<1e-2):
            stopper+=1
        else:
            stopper=0
                    
        if(stopper==10):
            break

        currcost = fincost

    trainaccuracy = getacc_relu(xtrain,params,labeltrain)
    testaccuracy = getacc_relu(xtest,params,labeltest)
    confusionmatrix = confusion_relu(xtest,params,labeltest)

    print("Using ReLU Activation Function and BCE Loss Function with 4 hidden layers")
    print("Training accuracy is",trainaccuracy)
    print("Testing accuracy is",testaccuracy)


elif part == 'g':
    clf = MLPClassifier(hidden_layer_sizes=(50,50,50,50),activation='relu',solver='sgd',learning_rate_init=0.1,learning_rate='adaptive',tol=1e-2)
    sttime = time.time()
    clf.fit(xtrain,labeltrain)
    endtime = time.time()

    trainaccuracy = 100*clf.score(xtrain,labeltrain)
    testaccuracy = 100*clf.score(xtest,labeltest)

    print("Using MLP Classifier with 4 hidden layers")
    print("Training accuracy is",trainaccuracy)
    print("Testing accuracy is",testaccuracy)
    print("Time Taken is", endtime-sttime)


file.close()