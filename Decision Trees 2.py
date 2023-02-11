import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import graphviz
import matplotlib.pyplot as plt
import xgboost as xgb
import sys

trainaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/train1.csv"
testaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/test1.csv"
valaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3/val1.csv"
outputpath = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment3"

part = 'a'



if(len(sys.argv)!=1):
    trainaddress = sys.argv[1]
    valaddress = sys.argv[2]
    testaddress = sys.argv[3]
    outputpath = sys.argv[4]
    part = sys.argv[5]


file = open(outputpath+"/1_"+part+".txt",'a')
sys.stdout = file


if part=='a':

    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)
    traindata = traindata[~np.isnan(traindata).any(axis=1)]

    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)
    testdata = testdata[~np.isnan(testdata).any(axis=1)]

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)
    valdata = valdata[~np.isnan(valdata).any(axis=1)]


    dectree = tree.DecisionTreeClassifier()
    dectree.fit(traindata[:,:-1],traindata[:,-1])
    trainaccuracy = dectree.score(traindata[:,:-1],traindata[:,-1])
    testaccuracy = dectree.score(testdata[:,:-1],testdata[:,-1])
    valaccuracy = dectree.score(valdata[:,:-1],valdata[:,-1])
    print('Training Accuracy is',100*trainaccuracy)
    print('Validation Accuracy is',100*valaccuracy)
    print('Testing Accuracy is',100*testaccuracy)
    gdata = tree.export_graphviz(dectree, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1a")









elif part=='b':
    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)
    traindata = traindata[~np.isnan(traindata).any(axis=1)]

    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)
    testdata = testdata[~np.isnan(testdata).any(axis=1)]

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)
    valdata = valdata[~np.isnan(valdata).any(axis=1)]

    params = {'max_depth' : list(range(1,10)), 'min_samples_split' : [2,3,4], 'min_samples_leaf' : list(range(25,35))}
    dectree = GridSearchCV(tree.DecisionTreeClassifier(),params)
    dectree.fit(traindata[:,:-1],traindata[:,-1])
    dectree = dectree.best_estimator_
    trainaccuracy = dectree.score(traindata[:,:-1],traindata[:,-1])
    testaccuracy = dectree.score(testdata[:,:-1],testdata[:,-1])
    valaccuracy = dectree.score(valdata[:,:-1],valdata[:,-1])
    print('Training Accuracy is',100*trainaccuracy)
    print('Validation Accuracy is',100*valaccuracy)
    print('Testing Accuracy is',100*testaccuracy)
    gdata = tree.export_graphviz(dectree, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1b")








elif part=='c':
    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)
    traindata = traindata[~np.isnan(traindata).any(axis=1)]

    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)
    testdata = testdata[~np.isnan(testdata).any(axis=1)]

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)
    valdata = valdata[~np.isnan(valdata).any(axis=1)]
    
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(traindata[:,:-1],traindata[:,-1])
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    depths = []
    trainaccuracy = []
    testaccuracy = []
    valaccuracy = []
    for ccp_alpha in ccp_alphas:
        dectree = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        dectree.fit(traindata[:,:-1],traindata[:,-1])
        trainacc = dectree.score(traindata[:,:-1],traindata[:,-1])
        testacc = dectree.score(testdata[:,:-1],testdata[:,-1])
        valacc = dectree.score(valdata[:,:-1],valdata[:,-1])

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
    gdata = tree.export_graphviz(dectree, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1c")

    plt.plot(ccp_alphas,impurities,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('total impurity')
    plt.title('Total Impurity vs alpha')
    plt.savefig(outputpath+'/1c_impurity.png')
    plt.clf()

    plt.plot(ccp_alphas,depths,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('depth')
    plt.title('Depth vs alpha')
    plt.savefig(outputpath+'/1c_depth.png')
    plt.clf()

    plt.plot(ccp_alphas,trainaccuracy,color='red',label='Training accuracy')
    plt.plot(ccp_alphas,testaccuracy,color='blue',label='Testing accuracy')
    plt.plot(ccp_alphas,valaccuracy,color='green',label='Validation accuracy')
    plt.xlabel('alpha')
    plt.ylabel('accuracies')
    plt.title('Accuracies vs alpha')
    plt.legend()
    plt.savefig(outputpath+'/1c_accuracies.png')
    plt.clf()
    






elif part=='d':
    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)
    traindata = traindata[~np.isnan(traindata).any(axis=1)]

    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)
    testdata = testdata[~np.isnan(testdata).any(axis=1)]

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)
    valdata = valdata[~np.isnan(valdata).any(axis=1)]

    numfeatures = np.shape(traindata)[1] - 1

    params = {'n_estimators' : list(range(85,95)), 'min_samples_split' : [2,3,4], 'max_features' : ['sqrt', 'log2',None] + (list(range(2,numfeatures+1)))}
    dectree = GridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),params)
    dectree.fit(traindata[:,:-1],traindata[:,-1])
    dectree = dectree.best_estimator_
    trainaccur = dectree.score(traindata[:,:-1],traindata[:,-1])
    testaccur = dectree.score(testdata[:,:-1],testdata[:,-1])
    valaccur = dectree.score(valdata[:,:-1],valdata[:,-1])
    print('Training Accuracy is',100*trainaccur)
    print('Validation Accuracy is',100*valaccur)
    print('Testing Accuracy is',100*testaccur)
    print('Out of Bag Accuracy is', 100*dectree.oob_score_)






elif part=='e':
    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)

    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_median.fit(traindata)
    imp_mode.fit(traindata)
    
    medtraindata = imp_median.transform(traindata)
    modtraindata = imp_mode.transform(traindata)

    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)

    medtestdata = imp_median.transform(testdata)
    modtestdata = imp_mode.transform(testdata)

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)

    medvaldata = imp_median.transform(valdata)
    modvaldata = imp_mode.transform(valdata)



    print("------------Using median imputed data------------")

    print("-----part A-----")
    dectree1 = tree.DecisionTreeClassifier()
    dectree1.fit(medtraindata[:,:-1],medtraindata[:,-1])
    trainaccuracy1 = dectree1.score(medtraindata[:,:-1],medtraindata[:,-1])
    testaccuracy1 = dectree1.score(medtestdata[:,:-1],medtestdata[:,-1])
    valaccuracy1 = dectree1.score(medvaldata[:,:-1],medvaldata[:,-1])
    print('Training Accuracy is',100*trainaccuracy1)
    print('Validation Accuracy is',100*valaccuracy1)
    print('Testing Accuracy is',100*testaccuracy1)
    
    
    
    
    gdata = tree.export_graphviz(dectree1, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_a_median")


    print("-----part B-----")

    params = {'max_depth' : list(range(10,15)), 'min_samples_split' : [2,3,4], 'min_samples_leaf' : list(range(25,35))}
    dectree2 = GridSearchCV(tree.DecisionTreeClassifier(),params)
    dectree2.fit(medtraindata[:,:-1],medtraindata[:,-1])
    dectree2 = dectree2.best_estimator_
    trainaccuracy2 = dectree2.score(medtraindata[:,:-1],medtraindata[:,-1])
    testaccuracy2 = dectree2.score(medtestdata[:,:-1],medtestdata[:,-1])
    valaccuracy2 = dectree2.score(medvaldata[:,:-1],medvaldata[:,-1])
    print('Training Accuracy is',100*trainaccuracy2)
    print('Validation Accuracy is',100*valaccuracy2)
    print('Testing Accuracy is',100*testaccuracy2)
    
    
    gdata = tree.export_graphviz(dectree2, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_b_median")



    print('-----Part C-----')
    clf1 = tree.DecisionTreeClassifier(random_state=0)
    path1 = clf1.cost_complexity_pruning_path(medtraindata[:,:-1],medtraindata[:,-1])
    ccp_alphas, impurities = path1.ccp_alphas, path1.impurities

    depths = []
    trainaccuracy3 = []
    testaccuracy3 = []
    valaccuracy3 = []
    for ccp_alpha in ccp_alphas:
        dectree3 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        dectree3.fit(medtraindata[:,:-1],medtraindata[:,-1])
        trainacc = dectree3.score(medtraindata[:,:-1],medtraindata[:,-1])
        testacc = dectree3.score(medtestdata[:,:-1],medtestdata[:,-1])
        valacc = dectree3.score(medvaldata[:,:-1],medvaldata[:,-1])

        trainaccuracy3.append(trainacc)
        testaccuracy3.append(testacc)
        valaccuracy3.append(valacc)
        depths.append(dectree3.get_depth())

    idx = np.argmax(valaccuracy3)

    trainaccur = trainaccuracy3[idx]
    testaccur = testaccuracy3[idx]
    valaccur = valaccuracy3[idx]
    print('Training Accuracy is',100*trainaccur)
    print('Validation Accuracy is',100*valaccur)
    print('Testing Accuracy is',100*testaccur)



    gdata = tree.export_graphviz(dectree3, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_c_median")



    plt.plot(ccp_alphas,impurities,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('total impurity')
    plt.title('Total Impurity vs alpha in Median')
    plt.savefig(outputpath+'/1d_c_median_impurities.png')
    plt.clf()

    plt.plot(ccp_alphas,depths,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('depth')
    plt.title('Depth vs alpha in Median')
    plt.savefig(outputpath+'/1d_c_median_depths.png')
    plt.clf()

    plt.plot(ccp_alphas,trainaccuracy3,color='red',label='Training accuracy')
    plt.plot(ccp_alphas,testaccuracy3,color='blue',label='Testing accuracy')
    plt.plot(ccp_alphas,valaccuracy3,color='green',label='Validation accuracy')
    plt.xlabel('alpha')
    plt.ylabel('accuracies')
    plt.title('Accuracies vs alpha in Median')
    plt.legend()
    plt.savefig(outputpath+'/1d_c_median_accuracies.png')
    plt.clf()



    print('-----Part D-----')

    numfeatures = np.shape(medtraindata)[1] - 1

    params2 = {'n_estimators' : list(range(80,90)), 'min_samples_split' : [2,3,4], 'max_features' : ['sqrt', 'log2',None] + (list(range(2,numfeatures+1)))}
    dectree4 = GridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),params2)
    dectree4.fit(medtraindata[:,:-1],medtraindata[:,-1])
    dectree4 = dectree4.best_estimator_
    trainaccur4 = dectree4.score(medtraindata[:,:-1],medtraindata[:,-1])
    testaccur4 = dectree4.score(medtestdata[:,:-1],medtestdata[:,-1])
    valaccur4 = dectree4.score(medvaldata[:,:-1],medvaldata[:,-1])
    print('Training Accuracy is',100*trainaccur4)
    print('Validation Accuracy is',100*valaccur4)
    print('Testing Accuracy is',100*testaccur4)
    print('Out of Bag Accuracy is', 100*dectree4.oob_score_)

















    print("------------Using mode imputed data------------")

    print("-----part A-----")
    dectree1 = tree.DecisionTreeClassifier()
    dectree1.fit(modtraindata[:,:-1],modtraindata[:,-1])
    trainaccuracy1 = dectree1.score(modtraindata[:,:-1],modtraindata[:,-1])
    testaccuracy1 = dectree1.score(modtestdata[:,:-1],modtestdata[:,-1])
    valaccuracy1 = dectree1.score(modvaldata[:,:-1],modvaldata[:,-1])
    print('Training Accuracy is',100*trainaccuracy1)
    print('Validation Accuracy is',100*valaccuracy1)
    print('Testing Accuracy is',100*testaccuracy1)
    
    
    
    
    gdata = tree.export_graphviz(dectree1, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_a_mode")


    print("-----part B-----")

    params = {'max_depth' : list(range(10,15)), 'min_samples_split' : [2,3,4], 'min_samples_leaf' : list(range(25,35))}
    dectree2 = GridSearchCV(tree.DecisionTreeClassifier(),params)
    dectree2.fit(modtraindata[:,:-1],modtraindata[:,-1])
    dectree2 = dectree2.best_estimator_
    trainaccuracy2 = dectree2.score(modtraindata[:,:-1],modtraindata[:,-1])
    testaccuracy2 = dectree2.score(modtestdata[:,:-1],modtestdata[:,-1])
    valaccuracy2 = dectree2.score(modvaldata[:,:-1],modvaldata[:,-1])
    print('Training Accuracy is',100*trainaccuracy2)
    print('Validation Accuracy is',100*valaccuracy2)
    print('Testing Accuracy is',100*testaccuracy2)
    
    
    gdata = tree.export_graphviz(dectree2, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_b_mode")



    print('-----Part C-----')
    clf1 = tree.DecisionTreeClassifier(random_state=0)
    path1 = clf1.cost_complexity_pruning_path(modtraindata[:,:-1],modtraindata[:,-1])
    ccp_alphas, impurities = path1.ccp_alphas, path1.impurities

    depths = []
    trainaccuracy3 = []
    testaccuracy3 = []
    valaccuracy3 = []
    for ccp_alpha in ccp_alphas:
        dectree3 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        dectree3.fit(modtraindata[:,:-1],modtraindata[:,-1])
        trainacc = dectree3.score(modtraindata[:,:-1],modtraindata[:,-1])
        testacc = dectree3.score(modtestdata[:,:-1],modtestdata[:,-1])
        valacc = dectree3.score(modvaldata[:,:-1],modvaldata[:,-1])

        trainaccuracy3.append(trainacc)
        testaccuracy3.append(testacc)
        valaccuracy3.append(valacc)
        depths.append(dectree3.get_depth())

    idx = np.argmax(valaccuracy3)

    trainaccur = trainaccuracy3[idx]
    testaccur = testaccuracy3[idx]
    valaccur = valaccuracy3[idx]
    print('Training Accuracy is',100*trainaccur)
    print('Validation Accuracy is',100*valaccur)
    print('Testing Accuracy is',100*testaccur)



    gdata = tree.export_graphviz(dectree3, out_file=None)
    graph = graphviz.Source(gdata)
    graph.render(outputpath+"/1d_c_mode")



    plt.plot(ccp_alphas,impurities,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('total impurity')
    plt.title('Total Impurity vs alpha in Mode')
    plt.savefig(outputpath+'/1d_c_mode_impurities.png')
    plt.clf()

    plt.plot(ccp_alphas,depths,marker='.')
    plt.xlabel('alpha')
    plt.ylabel('depth')
    plt.title('Depth vs alpha in Mode')
    plt.savefig(outputpath+'/1d_c_mode_depths.png')
    plt.clf()

    plt.plot(ccp_alphas,trainaccuracy3,color='red',label='Training accuracy')
    plt.plot(ccp_alphas,testaccuracy3,color='blue',label='Testing accuracy')
    plt.plot(ccp_alphas,valaccuracy3,color='green',label='Validation accuracy')
    plt.xlabel('alpha')
    plt.ylabel('accuracies')
    plt.title('Accuracies vs alpha in Mode')
    plt.legend()
    plt.savefig(outputpath+'/1d_c_mode_accuracies.png')
    plt.clf()



    print('-----Part D-----')

    numfeatures = np.shape(modtraindata)[1] - 1

    params2 = {'n_estimators' : list(range(80,85)), 'min_samples_split' : [2,3,4], 'max_features' : ['sqrt', 'log2',None] + (list(range(2,numfeatures+1)))}
    dectree4 = GridSearchCV(RandomForestClassifier(random_state=0,oob_score=True),params2)
    dectree4.fit(modtraindata[:,:-1],modtraindata[:,-1])
    dectree4 = dectree4.best_estimator_
    trainaccur4 = dectree4.score(modtraindata[:,:-1],modtraindata[:,-1])
    testaccur4 = dectree4.score(modtestdata[:,:-1],modtestdata[:,-1])
    valaccur4 = dectree4.score(modvaldata[:,:-1],modvaldata[:,-1])
    print('Training Accuracy is',100*trainaccur4)
    print('Validation Accuracy is',100*valaccur4)
    print('Testing Accuracy is',100*testaccur4)
    print('Out of Bag Accuracy is', 100*dectree4.oob_score_)











elif part=='f':

    traindata = pd.read_csv(trainaddress)
    traindata = traindata.replace('?',np.NaN)
    traindata = np.array(traindata)
    traindata = (traindata[:,1:]).astype(float)


    testdata = pd.read_csv(testaddress)
    testdata = testdata.replace('?',np.NaN)
    testdata = np.array(testdata)
    testdata = (testdata[:,1:]).astype(float)

    valdata = pd.read_csv(valaddress)
    valdata = valdata.replace('?',np.NaN)
    valdata = np.array(valdata)
    valdata = (valdata[:,1:]).astype(float)
    
    params = {'n_estimators' : list(range(10,51,10)),'subsample' : np.arange(0.1,0.61,0.1),'max_depth' : list(range(4,11,1))}
    clf = GridSearchCV(xgb.XGBClassifier(missing=np.nan),params)
    clf.fit(traindata[:,:-1],traindata[:,-1])

    dectree = clf.best_estimator_

    trainaccur = dectree.score(traindata[:,:-1],traindata[:,-1])
    testaccur = dectree.score(testdata[:,:-1],testdata[:,-1])
    valaccur = dectree.score(valdata[:,:-1],valdata[:,-1])
    print('Training Accuracy is',100*trainaccur)
    print('Validation Accuracy is',100*valaccur)
    print('Testing Accuracy is',100*testaccur)


file.close()