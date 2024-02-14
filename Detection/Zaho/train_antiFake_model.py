#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


#absolute path & before excuting the script, I deleted all the unnecessary columns except "isFake" and 26 features
inputFile='./merg-Baseline-eau.csv'
df=pd.read_csv(inputFile)
data=df.values
data[:,1:26]=(data[:,1:26]-np.min(data[:,1:26],axis=0))/(np.max(data[:,1:26],axis=0)-np.min(data[:,1:26],axis=0))
np.random.shuffle(data)


spatial=data[:,(15,16,19,21,22)]
histogram=data[:,(1,3,4,5,6,7,8,9,11,12,14)]
frequency=data[:,23:]



clf = svm.SVC(kernel='linear', C=1)
calculist=[]
calculist.append(np.hstack((spatial,histogram,frequency)))

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

for calcu in calculist:
    X_train, X_test, y_train, y_test = train_test_split(calcu, data[:,0], test_size=0.9)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='f1')
    clf.fit(X_train,y_train)
    best_parameters=clf.best_params_
    svm = SVC(**best_parameters)
    svm.fit(X_train, y_train)
    y_true,y_pred=y_test,svm.predict(X_test)
    print(classification_report(y_true, y_pred))
