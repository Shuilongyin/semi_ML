# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:05:17 2018

@author: Administrator
"""

import sys
sys.path.append(r"C:\Users\Administrator\Desktop\Python-Real-World-Machine-Learning-master\Module 2\Chapter 5")

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import  CPLELearning_2
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sklearn.svm
from sklearn.datasets import make_blobs,make_circles
from sklearn.model_selection import train_test_split
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from imp import reload
import nlopt

warnings.filterwarnings("ignore")
reload(CPLELearning_2)
np.random.seed(1)

def auc_score(y_true, predict_proba):
    '''
    y_true: numpy.ndarray,不能是带索引的series
    '''
    false_positive_rate, recall, thresholds = roc_curve(y_true,predict_proba)
    roc_auc = auc(false_positive_rate, recall)
    return roc_auc

###########################################################################################
#生成数据
# load data
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1, 1.5]

data, label = make_blobs(n_samples=1000, n_features=5, centers=centers, cluster_std=clusters_std,random_state=0, shuffle=False)
data, label = make_circles(n_samples=1000, noise=0.08,random_state=0, shuffle=False)
pyplot.scatter(data[:, 0], data[:, 1], c=label)
pyplot.show()

#取label缺失的数据(在0中取100个，1中取400个)，有标记的数据（1,100个，0,400个）
data_unlabel_pos_idx = np.random.rand(500,)<0.9
data_unlabel_neg_idx = np.random.rand(500,)<0.9
X_unlabel_pos = data[500:][data_unlabel_pos_idx,:]
X_unlabel_neg = data[:500][data_unlabel_neg_idx,:]
y_unlabel_pos = label[500:][data_unlabel_pos_idx]
y_unlabel_neg = label[:500][data_unlabel_neg_idx]

data_unlabel = np.concatenate([X_unlabel_pos,X_unlabel_neg],axis=0)
y_unlabel = np.concatenate([y_unlabel_pos,y_unlabel_neg],axis=0)

pyplot.scatter(data_unlabel[:, 0], data_unlabel[:, 1], c=y_unlabel)
pyplot.show()

data_unlabel_train, data_unlabel_test, y_unlabel_train, y_unlabel_test = train_test_split(data_unlabel, y_unlabel, test_size=0.3, random_state=42)


X_label_pos = data[500:][~data_unlabel_pos_idx,:]
X_label_neg = data[:500][~data_unlabel_neg_idx,:]
y_label_pos = label[500:][~data_unlabel_pos_idx]
y_label_neg = label[:500][~data_unlabel_neg_idx]

data_label = np.concatenate([X_label_pos,X_label_neg],axis=0)
y_label = np.concatenate([y_label_pos,y_label_neg],axis=0)

pyplot.scatter(data_label[:, 0], data_label[:, 1], c=y_label)
pyplot.show()

data_label_train, data_label_test, y_label_train, y_label_test = train_test_split(data_label, y_label, test_size=0.3, random_state=42)

data_train_merge = np.concatenate([data_label_train,data_unlabel_train],axis=0)
y_train_merge = np.concatenate([y_label_train,y_unlabel_train],axis=0)
y_train_merge_ = copy.deepcopy(y_train_merge)
y_train_merge_[data_label_train.shape[0]:] = -1

data_test_merge = np.concatenate([data_label_test,data_unlabel_test],axis=0)
y_test_merge = np.concatenate([y_label_test,y_unlabel_test],axis=0)
y_test_merge_ = copy.deepcopy(y_test_merge)
y_test_merge_[data_label_test.shape[0]:] = -1


##############################################################################
#利用有标签的数据生成一个xgb的有监督模型
dtrain_label = xgb.DMatrix(data_label_train,y_label_train)
dtest_label = xgb.DMatrix(data_label_test,y_label_test)

param1 = {'max_depth': 2, 'eta': 0.05, 'silent': 1,'eval_metric' :'logloss','objective':'reg:logistic'}
sup_model = xgb.train(param1, dtrain_label, 100)
score_train = sup_model.predict(dtrain_label)
auc_score(y_label_train,score_train)
#0.985

score_train = sup_model.predict(xgb.DMatrix(data_train_merge))
auc_score(y_train_merge,score_train)
#0.87

score_test = sup_model.predict(dtest_label)
auc_score(y_label_test,score_test)
#0.859

score_merge = sup_model.predict(xgb.DMatrix(data_test_merge))
auc_score(y_test_merge,score_merge)
#0.89

#完全有监督模型预测得到train的概率
labelP_sup = sup_model.predict(xgb.DMatrix(data_label_train))
unlabelP_sup = sup_model.predict(xgb.DMatrix(data_unlabel_train))

labelP_sup = np.vstack((1-labelP_sup,labelP_sup)).T
unlabelP_sup = np.vstack((1-unlabelP_sup,unlabelP_sup)).T

#计算L_sup
label_ll_sup = np.average((np.vstack((1-y_label_train,y_label_train)).T*np.log(labelP_sup)).sum(axis=1)) 


#############################################################################
it = 50
q = np.random.rand(data_unlabel_train.shape[0])
unlabelP = unlabelP_sup
labelP = labelP_sup
m = data_unlabel_train.shape[0]
for i in range(it):
    #最大化ll
    print ("============================")
    print ('it:',i)
    
    ll_previous = np.average((np.vstack((1-y_label_train,y_label_train)).T*np.log(labelP)).sum(axis=1)) \
        +np.average((np.vstack((1-q,q)).T*np.log(unlabelP)).sum(axis=1)) 
    print ('ll_previous:',ll_previous)
    dtrain_merge = xgb.DMatrix(np.vstack((data_label_train,data_unlabel_train)),np.hstack((y_label_train,q)))
    param2 = {'max_depth': 2, 'eta': 0.05, 'silent': 1,'eval_metric' :'logloss','objective':'reg:logistic','verbose_eval':True}
    semi_model = xgb.train(param2, dtrain_merge, 100)
    
    labelP_ = semi_model.predict(xgb.DMatrix(data_label_train))
    unlabelP_ = semi_model.predict(xgb.DMatrix(data_unlabel_train))

    labelP_ = np.vstack((1-labelP_,labelP_)).T
    unlabelP_ = np.vstack((1-unlabelP_,unlabelP_)).T
    
    ll_last = np.average((np.vstack((1-y_label_train,y_label_train)).T*np.log(labelP_)).sum(axis=1)) \
        +np.average((np.vstack((1-q,q)).T*np.log(unlabelP_)).sum(axis=1))
    
    print ('ll_last:',ll_last)
    print ("ll_diff:",ll_last-ll_previous)
    
    print('label_train_auc:',auc_score(y_label_train,semi_model.predict(xgb.DMatrix(data_label_train))))
    print('merge_auc:',auc_score(y_test_merge,semi_model.predict(xgb.DMatrix(data_test_merge))))
    
    if i>0:
        print('unlabel_train_auc:',auc_score(q,semi_model.predict(xgb.DMatrix(data_unlabel_train))))
    
    #最小化ll_diff
    def my_fun(q_, grad=[]):
        ll_diff = 0
        if grad.size>0:
            for i in range(m):
                grad[i] = (1/m)*(np.log(unlabelP_[i,1])+np.log(unlabelP_sup[i,0])-np.log(unlabelP_[i,0])-np.log(unlabelP_sup[i,1]))
                ll_diff+=q_[i]*np.log(unlabelP_[i,1])+(1-q_[i])*np.log(unlabelP_[i,0])-(q_[i]*np.log(unlabelP_sup[i,1])+(1-q_[i])*np.log(unlabelP_sup[i,0]))
        return (1/m)*ll_diff

    opt = nlopt.opt(nlopt.LD_MMA, m)
    opt.set_lower_bounds([0 for i in range(m)])
    opt.set_upper_bounds([1 for i in range(m)])
    opt.set_min_objective(my_fun)
    opt.set_xtol_rel(1e-4)
    opt.verbose = 1
    q = opt.optimize(q.tolist())






















