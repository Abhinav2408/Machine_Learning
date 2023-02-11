import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import regex as re
from scipy import sparse
from datetime import datetime
import time
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import sys
le = LabelEncoder()


trainaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/DrugsComTrain.csv"
testaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/DrugsComTest.csv"
valaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/DrugsComVal.csv"
outputpath = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3"

part = 'a'

if(len(sys.argv)!=1):
    trainaddress = sys.argv[1]
    valaddress = sys.argv[2]
    testaddress = sys.argv[3]
    outputpath = sys.argv[4]
    part = sys.argv[5]


file = open(outputpath+"/2_"+part+".txt",'a')
sys.stdout = file


swords = stopwords.words('english')
    
traindata = pd.read_csv(trainaddress)
traindata = traindata.replace(np.nan,'',regex=True)
traindata = np.array(traindata)





if part != 'g':

    for i in range(len(traindata[:,1])):
        data = traindata[i][1]
        data = data.lower()
        data = data.replace('&#039;',"'")
        data = re.sub('<br />', '', data)
        data = re.sub('\n', '', data)
        data = re.sub('\r', '', data)
        data = re.sub(r'[^\w\s]', '', data)
        z = [y for y in data.split() if y not in swords]
        traindata[i][1] = (' '.join(z))
        traindata[i][0] = traindata[i][0].lower()






    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace(np.nan,'',regex=True)
    testdata = np.array(testdata)

    for i in range(len(testdata[:,1])):
        data = testdata[i][1]
        data = data.lower()
        data = data.replace('&#039;',"'")
        data = re.sub('<br />', '', data)
        data = re.sub('\n', '', data)
        data = re.sub('\r', '', data)
        data = re.sub(r'[^\w\s]', '', data)
        z = [y for y in data.split() if y not in swords]
        testdata[i][1] = (' '.join(z))
        testdata[i][0] = testdata[i][0].lower()







    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace(np.nan,'',regex=True)
    valdata = np.array(valdata)

    for i in range(len(valdata[:,1])):
        data = valdata[i][1]
        data = data.lower()
        data = data.replace('&#039;',"'")
        data = re.sub('<br />', '', data)
        data = re.sub('\n', '', data)
        data = re.sub('\r', '', data)
        data = re.sub(r'[^\w\s]', '', data)
        z = [y for y in data.split() if y not in swords]
        valdata[i][1] = (' '.join(z))
        valdata[i][0] = valdata[i][0].lower()





    vectcondition = CountVectorizer()
    vectreview = CountVectorizer()

    traincondition = vectcondition.fit_transform(traindata[:,0])
    testcondition = vectcondition.transform(testdata[:,0])
    valcondition = vectcondition.transform(valdata[:,0])



    trainreview = vectreview.fit_transform(traindata[:,1])
    testreview = vectreview.transform(testdata[:,1])
    valreview = vectreview.transform(valdata[:,1])



    traindates = traindata[:,3]
    for i in range(len(traindates)):
        traindates[i] = re.sub(r'[^\w\s]', '', traindates[i])
    traindatematrix = np.array([traindates[i].split(' ') for i in range(len(traindates))])
    for i in range(len(traindates)):
        traindatematrix[i][0] = str(datetime.strptime(traindatematrix[i][0], '%B').month)
    traindatematrix = traindatematrix.astype(int)
    trainday = sparse.csr_matrix(traindatematrix[:,1].reshape(-1,1))
    trainmonth = sparse.csr_matrix(traindatematrix[:,0].reshape(-1,1))
    trainyear = sparse.csr_matrix(traindatematrix[:,2].reshape(-1,1))
    trainusefulcount = sparse.csr_matrix(np.array(traindata[:,4]).reshape(-1,1).astype(int))


    testdates = testdata[:,3]
    for i in range(len(testdates)):
        testdates[i] = re.sub(r'[^\w\s]', '', testdates[i])
    testdatematrix = np.array([testdates[i].split(' ') for i in range(len(testdates))])
    for i in range(len(testdates)):
        testdatematrix[i][0] = str(datetime.strptime(testdatematrix[i][0], '%B').month)
    testdatematrix = testdatematrix.astype(int)
    testday = sparse.csr_matrix(testdatematrix[:,1].reshape(-1,1))
    testmonth = sparse.csr_matrix(testdatematrix[:,0].reshape(-1,1))
    testyear = sparse.csr_matrix(testdatematrix[:,2].reshape(-1,1))
    testusefulcount = sparse.csr_matrix(np.array(testdata[:,4]).reshape(-1,1).astype(int))


    valdates = valdata[:,3]
    for i in range(len(valdates)):
        valdates[i] = re.sub(r'[^\w\s]', '', valdates[i])
    valdatematrix = np.array([valdates[i].split(' ') for i in range(len(valdates))])
    for i in range(len(valdates)):
        valdatematrix[i][0] = str(datetime.strptime(valdatematrix[i][0], '%B').month)
    valdatematrix = valdatematrix.astype(int)
    valday = sparse.csr_matrix(valdatematrix[:,1].reshape(-1,1))
    valmonth = sparse.csr_matrix(valdatematrix[:,0].reshape(-1,1))
    valyear = sparse.csr_matrix(valdatematrix[:,2].reshape(-1,1))
    valusefulcount = sparse.csr_matrix(np.array(valdata[:,4]).reshape(-1,1).astype(int))




    nettraindata = sparse.hstack((traincondition,trainreview,trainday,trainmonth,trainyear,trainusefulcount))
    labeltraindata = traindata[:,2].astype(int)


    nettestdata = sparse.hstack((testcondition,testreview,testday,testmonth,testyear,testusefulcount))
    labeltestdata = testdata[:,2].astype(int)


    netvaldata = sparse.hstack((valcondition,valreview,valday,valmonth,valyear,valusefulcount))
    labelvaldata = valdata[:,2].astype(int)


    if part=='a':
        dectree = tree.DecisionTreeClassifier()
        dectree.fit(nettraindata,labeltraindata)
        trainaccuracy = dectree.score(nettraindata,labeltraindata)
        testaccuracy = dectree.score(nettestdata,labeltestdata)
        valaccuracy = dectree.score(netvaldata,labelvaldata)
        print('Training Accuracy is',100*trainaccuracy)
        print('Validation Accuracy is',100*valaccuracy)
        print('Testing Accuracy is',100*testaccuracy)


    elif part=='b':
        params = {'max_depth' : list(range(2,15)), 'min_samples_split' : [277], 'min_samples_leaf' : list(range(10,15))}
        dectree = GridSearchCV(tree.DecisionTreeClassifier(),params)
        dectree.fit(nettraindata,labeltraindata)
        dectree = dectree.best_estimator_
        trainaccuracy = dectree.score(nettraindata,labeltraindata)
        testaccuracy = dectree.score(nettestdata,labeltestdata)
        valaccuracy = dectree.score(netvaldata,labelvaldata)
        print('Training Accuracy is',100*trainaccuracy)
        print('Validation Accuracy is',100*valaccuracy)
        print('Testing Accuracy is',100*testaccuracy)


    elif part=='c':
        clf = tree.DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(nettraindata,labeltraindata)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        depths = []
        trainaccuracy = []
        testaccuracy = []
        valaccuracy = []

        for ccp_alpha in ccp_alphas:
            dectree = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            dectree.fit(nettraindata,labeltraindata)
            trainacc = dectree.score(nettraindata,labeltraindata)
            testacc = dectree.score(nettestdata,labeltestdata)
            valacc = dectree.score(netvaldata,labelvaldata)
            
            trainaccuracy.append(trainacc)
            testaccuracy.append(testacc)
            valaccuracy.append(valacc)
            depths.append(dectree.get_depth())

        idx = np.argmax(valaccuracy)

        trainaccur = trainaccuracy[idx]
        testaccur = testaccuracy[idx]
        valaccur = valaccuracy[idx]
        print('Training Accuracy is',100*trainaccur)
        print('Validation Accuracy is',100*valaccur)
        print('Testing Accuracy is',100*testaccur)
    

        plt.plot(ccp_alphas,impurities,marker='.')
        plt.xlabel('alpha')
        plt.ylabel('total impurity')
        plt.title('Total Impurity vs alpha')
        plt.savefig(outputpath+'/2c_impurity.png')
        plt.clf()

        plt.plot(ccp_alphas,depths,marker='.')
        plt.xlabel('alpha')
        plt.ylabel('depth')
        plt.title('Depth vs alpha')
        plt.savefig(outputpath+'/2c_depth.png')
        plt.clf()

        plt.plot(ccp_alphas,trainaccuracy,color='red',label='Training accuracy')
        plt.plot(ccp_alphas,testaccuracy,color='blue',label='Testing accuracy')
        plt.plot(ccp_alphas,valaccuracy,color='green',label='Validation accuracy')
        plt.xlabel('alpha')
        plt.ylabel('accuracies')
        plt.title('Accuracies vs alpha')
        plt.legend()
        plt.savefig(outputpath+'/2c_accuracies.png')
        plt.clf()



    elif part=='d':
        params = {'n_estimators' : list(range(50,451,50)), 'min_samples_split' : [2,4,6,8,10], 'max_features' : np.arange(0.4,0.81,0.1)}
        dectree = GridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),params)
        dectree.fit(nettraindata,labeltraindata)
        dectree = dectree.best_estimator_
        trainaccuracy = dectree.score(nettraindata,labeltraindata)
        testaccuracy = dectree.score(nettestdata,labeltestdata)
        valaccuracy = dectree.score(netvaldata,labelvaldata)
        print('Training Accuracy is',100*trainaccuracy)
        print('Validation Accuracy is',100*valaccuracy)
        print('Testing Accuracy is',100*testaccuracy)
        print('Out of Bag Accuracy is', 100*dectree.oob_score_)


    elif part=='e':
        params = {'n_estimators' : list(range(50,450,50)),'subsample' : np.arange(0.4,0.81,0.1),'max_depth' : list(range(40,71,10))}
        clf = GridSearchCV(xgb.XGBClassifier(),params)
        labeltraindata = le.fit_transform(labeltraindata)
        labeltestdata = le.fit_transform(labeltestdata)
        labelvaldata = le.fit_transform(labelvaldata)
        clf.fit(nettraindata,labeltraindata)

        dectree = clf.best_estimator_

        trainaccuracy = dectree.score(nettraindata,labeltraindata)
        testaccuracy = dectree.score(nettestdata,labeltestdata)
        valaccuracy = dectree.score(netvaldata,labelvaldata)
        print('Training Accuracy is',100*trainaccuracy)
        print('Validation Accuracy is',100*valaccuracy)
        print('Testing Accuracy is',100*testaccuracy)


    elif part=='f':

        parameters = {'boosting': ['gbdt' ],'max_depth' :[10,50,100,500],'min_data_in_leaf':[15,25,50 ]}
        lgb_model = lgb.LGBMClassifier()
        clf = GridSearchCV(lgb_model, param_grid = parameters)
        nettraindata = nettraindata.astype(float)
        labeltraindata = labeltraindata.astype(float)

        nettestdata = nettestdata.astype(float)
        labeltestdata = labeltestdata.astype(float)

        netvaldata = netvaldata.astype(float)
        labelvaldata = labelvaldata.astype(float)

        clf.fit(nettraindata,labeltraindata)
        dectree = clf.best_estimator_

        trainaccuracy = dectree.score(nettraindata,labeltraindata)
        testaccuracy = dectree.score(nettestdata,labeltestdata)
        valaccuracy = dectree.score(netvaldata,labelvaldata)
        print('Training Accuracy is',100*trainaccuracy)
        print('Validation Accuracy is',100*valaccuracy)
        print('Testing Accuracy is',100*testaccuracy)


else :
    np.random.shuffle(traindata)
    traindata = np.vstack([traindata,np.random.shuffle(traindata)])
    sizes = [20000,40000,60000,80000,100000,120000,140000,160000]

    atestacc = []
    atime = []

    btestacc = []
    btime= []

    ctestacc = []
    ctime = []

    dtestacc = []
    dtime = []

    etestacc = []
    etime =  []

    ftestacc = []
    ftime = []


    for n in sizes:
        print("Using training data of size :",n)
        usedtraindata = traindata[0:n,:]
        for i in range(len(usedtraindata[:,1])):
            data = usedtraindata[i][1]
            data = data.lower()
            data = data.replace('&#039;',"'")
            data = re.sub('<br />', '', data)
            data = re.sub('\n', '', data)
            data = re.sub('\r', '', data)
            data = re.sub(r'[^\w\s]', '', data)
            z = [y for y in data.split() if y not in swords]
            usedtraindata[i][1] = (' '.join(z))
            usedtraindata[i][0] = usedtraindata[i][0].lower()






        testdata = pd.read_csv(testaddress)
        testdata = testdata.replace(np.nan,'',regex=True)
        testdata = np.array(testdata)

        for i in range(len(testdata[:,1])):
            data = testdata[i][1]
            data = data.lower()
            data = data.replace('&#039;',"'")
            data = re.sub('<br />', '', data)
            data = re.sub('\n', '', data)
            data = re.sub('\r', '', data)
            data = re.sub(r'[^\w\s]', '', data)
            z = [y for y in data.split() if y not in swords]
            testdata[i][1] = (' '.join(z))
            testdata[i][0] = testdata[i][0].lower()







        valdata = pd.read_csv(valaddress)
        valdata = valdata.replace(np.nan,'',regex=True)
        valdata = np.array(valdata)

        for i in range(len(valdata[:,1])):
            data = valdata[i][1]
            data = data.lower()
            data = data.replace('&#039;',"'")
            data = re.sub('<br />', '', data)
            data = re.sub('\n', '', data)
            data = re.sub('\r', '', data)
            data = re.sub(r'[^\w\s]', '', data)
            z = [y for y in data.split() if y not in swords]
            valdata[i][1] = (' '.join(z))
            valdata[i][0] = valdata[i][0].lower()





        vectcondition = CountVectorizer()
        vectreview = CountVectorizer()

        traincondition = vectcondition.fit_transform(usedtraindata[:,0])
        testcondition = vectcondition.transform(testdata[:,0])
        valcondition = vectcondition.transform(valdata[:,0])



        trainreview = vectreview.fit_transform(usedtraindata[:,1])
        testreview = vectreview.transform(testdata[:,1])
        valreview = vectreview.transform(valdata[:,1])



        traindates = usedtraindata[:,3]
        for i in range(len(traindates)):
            traindates[i] = re.sub(r'[^\w\s]', '', traindates[i])
        traindatematrix = np.array([traindates[i].split(' ') for i in range(len(traindates))])
        for i in range(len(traindates)):
            traindatematrix[i][0] = str(datetime.strptime(traindatematrix[i][0], '%B').month)
        traindatematrix = traindatematrix.astype(int)
        trainday = sparse.csr_matrix(traindatematrix[:,1].reshape(-1,1))
        trainmonth = sparse.csr_matrix(traindatematrix[:,0].reshape(-1,1))
        trainyear = sparse.csr_matrix(traindatematrix[:,2].reshape(-1,1))
        trainusefulcount = sparse.csr_matrix(np.array(usedtraindata[:,4]).reshape(-1,1).astype(int))


        testdates = testdata[:,3]
        for i in range(len(testdates)):
            testdates[i] = re.sub(r'[^\w\s]', '', testdates[i])
        testdatematrix = np.array([testdates[i].split(' ') for i in range(len(testdates))])
        for i in range(len(testdates)):
            testdatematrix[i][0] = str(datetime.strptime(testdatematrix[i][0], '%B').month)
        testdatematrix = testdatematrix.astype(int)
        testday = sparse.csr_matrix(testdatematrix[:,1].reshape(-1,1))
        testmonth = sparse.csr_matrix(testdatematrix[:,0].reshape(-1,1))
        testyear = sparse.csr_matrix(testdatematrix[:,2].reshape(-1,1))
        testusefulcount = sparse.csr_matrix(np.array(testdata[:,4]).reshape(-1,1).astype(int))


        valdates = valdata[:,3]
        for i in range(len(valdates)):
            valdates[i] = re.sub(r'[^\w\s]', '', valdates[i])
        valdatematrix = np.array([valdates[i].split(' ') for i in range(len(valdates))])
        for i in range(len(valdates)):
            valdatematrix[i][0] = str(datetime.strptime(valdatematrix[i][0], '%B').month)
        valdatematrix = valdatematrix.astype(int)
        valday = sparse.csr_matrix(valdatematrix[:,1].reshape(-1,1))
        valmonth = sparse.csr_matrix(valdatematrix[:,0].reshape(-1,1))
        valyear = sparse.csr_matrix(valdatematrix[:,2].reshape(-1,1))
        valusefulcount = sparse.csr_matrix(np.array(valdata[:,4]).reshape(-1,1).astype(int))




        nettraindata = sparse.hstack((traincondition,trainreview,trainday,trainmonth,trainyear,trainusefulcount))
        labeltraindata = usedtraindata[:,2].astype(int)


        nettestdata = sparse.hstack((testcondition,testreview,testday,testmonth,testyear,testusefulcount))
        labeltestdata = testdata[:,2].astype(int)


        netvaldata = sparse.hstack((valcondition,valreview,valday,valmonth,valyear,valusefulcount))
        labelvaldata = valdata[:,2].astype(int)


        for x in ['a','b','c','d','e','f']:

            if x=='a':
                print("Part a")
                sttime= time.time()
                dectree = tree.DecisionTreeClassifier()
                dectree.fit(nettraindata,labeltraindata)
                trainaccuracy = dectree.score(nettraindata,labeltraindata)
                testaccuracy = dectree.score(nettestdata,labeltestdata)
                valaccuracy = dectree.score(netvaldata,labelvaldata)
                print('Training Accuracy is',100*trainaccuracy)
                print('Validation Accuracy is',100*valaccuracy)
                print('Testing Accuracy is',100*testaccuracy)
                endtime = time.time()
                atestacc.append(testaccuracy)
                atime.append(endtime-sttime)

            elif x=='b':
                print("Part b")
                sttime = time.time()
                params = {'max_depth' : list(range(2,15)), 'min_samples_split' : [277], 'min_samples_leaf' : list(range(10,15))}
                dectree = GridSearchCV(tree.DecisionTreeClassifier(),params)
                dectree.fit(nettraindata,labeltraindata)
                dectree = dectree.best_estimator_
                trainaccuracy = dectree.score(nettraindata,labeltraindata)
                testaccuracy = dectree.score(nettestdata,labeltestdata)
                valaccuracy = dectree.score(netvaldata,labelvaldata)
                print('Training Accuracy is',100*trainaccuracy)
                print('Validation Accuracy is',100*valaccuracy)
                print('Testing Accuracy is',100*testaccuracy)
                endtime = time.time()
                btestacc.append(testaccuracy)
                btime.append(endtime-sttime)

            elif x=='c':
                print("Part c")
                sttime = time.time()
                clf = tree.DecisionTreeClassifier(random_state=0)
                path = clf.cost_complexity_pruning_path(nettraindata,labeltraindata)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities
                depths = []
                trainaccuracy = []
                testaccuracy = []
                valaccuracy = []

                for ccp_alpha in ccp_alphas:
                    dectree = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                    dectree.fit(nettraindata,labeltraindata)
                    trainacc = dectree.score(nettraindata,labeltraindata)
                    testacc = dectree.score(nettestdata,labeltestdata)
                    valacc = dectree.score(netvaldata,labelvaldata)
                    
                    trainaccuracy.append(trainacc)
                    testaccuracy.append(testacc)
                    valaccuracy.append(valacc)
                    depths.append(dectree.get_depth())

                idx = np.argmax(valaccuracy)

                trainaccur = trainaccuracy[idx]
                testaccur = testaccuracy[idx]
                valaccur = valaccuracy[idx]
                print('Training Accuracy is',100*trainaccur)
                print('Validation Accuracy is',100*valaccur)
                print('Testing Accuracy is',100*testaccur)
                endtime = time.time()
                ctestacc.append(testaccuracy)
                ctime.append(endtime-sttime)

            elif x=='d':
                print("Part d")
                sttime = time.time()
                params = {'n_estimators' : list(range(50,451,50)), 'min_samples_split' : [2,4,6,8,10], 'max_features' : np.arange(0.4,0.81,0.1)}
                dectree = GridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),params)
                dectree.fit(nettraindata,labeltraindata)
                dectree = dectree.best_estimator_
                trainaccuracy = dectree.score(nettraindata,labeltraindata)
                testaccuracy = dectree.score(nettestdata,labeltestdata)
                valaccuracy = dectree.score(netvaldata,labelvaldata)
                print('Training Accuracy is',100*trainaccuracy)
                print('Validation Accuracy is',100*valaccuracy)
                print('Testing Accuracy is',100*testaccuracy)
                print('Out of Bag Accuracy is', 100*dectree.oob_score_)
                endtime = time.time()
                dtestacc.append(testaccuracy)
                dtime.append(endtime-sttime)

            elif x=='e':
                print("Part e")
                sttime= time.time()
                params = {'n_estimators' : list(range(50,450,50)),'subsample' : np.arange(0.4,0.81,0.1),'max_depth' : list(range(40,71,10))}
                clf = GridSearchCV(xgb.XGBClassifier(),params)
                labeltraindata = le.fit_transform(labeltraindata)
                labeltestdata = le.fit_transform(labeltestdata)
                labelvaldata = le.fit_transform(labelvaldata)
                clf.fit(nettraindata,labeltraindata)

                dectree = clf.best_estimator_

                trainaccuracy = dectree.score(nettraindata,labeltraindata)
                testaccuracy = dectree.score(nettestdata,labeltestdata)
                valaccuracy = dectree.score(netvaldata,labelvaldata)
                print('Training Accuracy is',100*trainaccuracy)
                print('Validation Accuracy is',100*valaccuracy)
                print('Testing Accuracy is',100*testaccuracy)
                endtime = time.time()
                etestacc.append(testaccuracy)
                etime.append(endtime - sttime)


            elif x=='f':
                sttime = time.time()
                print("Part f")
                parameters = {'boosting': ['gbdt' ],'max_depth' :[10,50,100,500],'min_data_in_leaf':[15,25,50 ]}
                lgb_model = lgb.LGBMClassifier()
                clf = GridSearchCV(lgb_model, param_grid = parameters)
                nettraindata = nettraindata.astype(float)
                labeltraindata = labeltraindata.astype(float)

                nettestdata = nettestdata.astype(float)
                labeltestdata = labeltestdata.astype(float)

                netvaldata = netvaldata.astype(float)
                labelvaldata = labelvaldata.astype(float)

                clf.fit(nettraindata,labeltraindata)
                dectree = clf.best_estimator_

                trainaccuracy = dectree.score(nettraindata,labeltraindata)
                testaccuracy = dectree.score(nettestdata,labeltestdata)
                valaccuracy = dectree.score(netvaldata,labelvaldata)
                print('Training Accuracy is',100*trainaccuracy)
                print('Validation Accuracy is',100*valaccuracy)
                print('Testing Accuracy is',100*testaccuracy)
                endtime = time.time()
                ftestacc.append(testaccuracy)
                ftime.append(endtime-sttime)


    plt.plot(sizes,atestacc,color='red',label='Testing accuracy - A')
    plt.plot(sizes,btestacc,color='blue',label='Testing accuracy - B')
    plt.plot(sizes,ctestacc,color='green',label='Testing accuracy - C')
    plt.plot(sizes,dtestacc,color='yellow',label='Testing accuracy - D')
    plt.plot(sizes,etestacc,color='cyan',label='Testing accuracy - E')
    plt.plot(sizes,ftestacc,color='magenta',label='Testing accuracy - F')
    plt.xlabel('n')
    plt.ylabel('Accuracies')
    plt.title('Accuracies vs n')
    plt.legend()
    plt.savefig(outputpath+'/2g_accuracies.png')
    plt.clf()


    plt.plot(sizes,atime,color='red',label='Training Time - A')
    plt.plot(sizes,btime,color='blue',label='TTraining Time - B')
    plt.plot(sizes,ctime,color='green',label='Training Time - C')
    plt.plot(sizes,dtime,color='yellow',label='Training Time - D')
    plt.plot(sizes,etime,color='cyan',label='Training Time - E')
    plt.plot(sizes,ftime,color='magenta',label='Training Time - F')
    plt.xlabel('n')
    plt.ylabel('Training Time')
    plt.title('Training Time vs n')
    plt.legend()
    plt.savefig(outputpath+'/2g_time.png')
    plt.clf()


file.close()