"""
date: 2018-11-29

to: self-training
"""

from sklearn.base import BaseEstimator
import sklearn.metrics
#import sys
import numpy
import numpy as np
#from sklearn.linear_model import LogisticRegression as LR
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_curve, auc
import copy
import pandas as pd

def auc_score(y_true, predict_proba):
    '''
    y_true: numpy.ndarray,不能是带索引的series
    '''
    false_positive_rate, recall, thresholds = roc_curve(y_true,predict_proba)
    roc_auc = auc(false_positive_rate, recall)
    return roc_auc

class SelfLearningModel(BaseEstimator):
    """
    self-training的简版框架；

    基础模型需要是一些类sklearn的模型，主要是train和predict的方法等等；

    self-training的一些资料： 如：http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf

    Parameters
    ----------
    basemodel : 基模型；
    max_iter : int,最大迭代次数；
    prob_threshold_pos : float, 即将unlabeled的样本加入训练样本的阈值(正例)；
    prob_threshold_neg : float, 即将unlabeled的样本加入训练样本的阈值（负例）;
    unlabeled_sample_weight: float, 当无标签样本加入训练时，给予的权重；
    top_n：int,即每次加入1标签的样本数
    NPratio：float,即每次加入1标签样本与0标签样本数之比，即标签不均衡时需调整
    de_ratio：float,即这次加入的样本数与前次的比值，即加入的样本数在逐渐衰减。
    """

    def __init__(self, basemodel, max_iter = 200, prob_threshold_pos = 0.8, prob_threshold_neg=0.8, unlabeled_sample_weight=0.8\
                 ,top_n=10,NPratio=1,de_ratio=0.9,predictors=[],dev_data=None,stopping_t=30):
        self.model = basemodel
        self.max_iter = max_iter
        self.prob_threshold_pos = prob_threshold_pos
        self.prob_threshold_neg = prob_threshold_neg
        self.unlabeled_sample_weight = unlabeled_sample_weight
        self.top_n=top_n
        self.NPratio = NPratio
        self.de_ratio = de_ratio
        self.predictors = predictors
        self.dev_data = dev_data
        self.stopping_t = stopping_t

    def fit(self, df):
        """
        Basemodel的train方法；
        df:DataFrame,dep='y',无标签样本的y=-1

        Returns
        -------
        self : returns an instance of self.
        """
        X = df[self.predictors]
        y_ = df['y']
        y = copy.deepcopy(y_)
        unlabeledX = X.loc[y==-1, :] #取无标签的变量
        labeledX = X.loc[y!=-1, :] #取有标签的变量
        labeledy = y[y!=-1] #取有标签的y
        sample_weight_ = np.array([1.0]*(X.shape[0]))
        sample_weight_[y==-1] = self.unlabeled_sample_weight
        
        self.model.fit(labeledX.values, labeledy.values) #先将有标签的样本进行训练
        unlabeledy = self.predict(unlabeledX.values) #对无标签的样本进行标签预测
        unlabeledprob = self.predict_proba(unlabeledX.values) #对无标签的样本进行概率进行预测
        
        pos_prob = pd.Series(unlabeledprob[:, 1],index=unlabeledX.index)
        neg_prob = pd.Series(unlabeledprob[:, 0],index=unlabeledX.index)
        
        pos_top = pos_prob.sort_values(ascending=False).head(int(self.top_n))
        uidx_pos = pos_top.index
        neg_top = neg_prob.sort_values(ascending=False).head(int((self.top_n/self.NPratio)))
        uidx_neg = neg_top.index
        print("pos_top:min_prob=%f,mean_prob=%f"%(pos_top.min(),pos_top.mean()))
        print("neg_top:min_prob=%f,mean_prob=%f"%(neg_top.min(),neg_top.mean()))
        
        df_pos = df.loc[uidx_pos,:]
        df_neg = df.loc[uidx_neg,:]
        # print("pos_top: ss:%i, y_10:%i, y_30:%i"%(df_pos[df_pos['y_10'].isin([0,1])].shape[0],df_pos[df_pos['y_10'].isin([0,1])]['y_10'].sum(),df_pos[df_pos['y_10'].isin([0,1])]['y_30'].sum()))
        # print("neg_top: ss:%i, y_10:%i, y_30:%i"%(df_neg[df_neg['y_10'].isin([0,1])].shape[0],df_neg[df_neg['y_10'].isin([0,1])]['y_10'].sum(),df_neg[df_neg['y_10'].isin([0,1])]['y_30'].sum()))
        
        #uidx_pos = pos_prob[pos_prob > self.prob_threshold_pos].index #unlabeled判断为正例的样本
        #uidx_neg = neg_prob[neg_prob > self.prob_threshold_neg].index #unlabeled判断为负例的样本
        uidx = np.hstack((uidx_pos, uidx_neg))
        self.uidx_pos = {}
        self.uidx_pos[0] = uidx_pos
        
        self.uidx_neg = {}
        self.uidx_neg[0] = uidx_neg
        self.auc = {}
        #re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0
        print ('iter: %i, n_pos: %i, n_neg: %i.'%(i, uidx_pos.shape[0],uidx_neg.shape[0])) #组合              
        print('uidx num: ',uidx.shape[0])
        
        #dev_auc
        # max_auc_10 = auc_score(self.dev_data[self.dev_data['y_10'].isin([0,1])]['y_10'], self.predict_proba(self.dev_data[self.dev_data['y_10'].isin([0,1])][self.predictors].values)[:,1])
        # max_auc_30 = auc_score(self.dev_data[self.dev_data['y_30'].isin([0,1])]['y_30'], self.predict_proba(self.dev_data[self.dev_data['y_30'].isin([0,1])][self.predictors].values)[:,1])
        # stop_t = 0
        while (len(uidx)!= 0) and i < self.max_iter :
            #当样本不满足阈值或达到迭代阈值时，停止迭代
            
            #部分U重新分配
            y[uidx_pos] = 1
            y[uidx_neg] = 0
            unlabeledX = X.loc[y==-1, :]
            labeledX = X.loc[y!=-1, :]
            labeledy = y[y!=-1]

            #训练新的数据的样本，并加上样本的weight
            self.model.fit(labeledX.values, labeledy.values,sample_weight = sample_weight_[y!=-1])
            unlabeledprob = self.predict_proba(unlabeledX.values)
            pos_prob = pd.Series(unlabeledprob[:, 1],index=unlabeledX.index)
            neg_prob = pd.Series(unlabeledprob[:, 0],index=unlabeledX.index)

            pos_top = pos_prob.sort_values(ascending=False).head(int(self.top_n*(self.de_ratio**(i+1))))
            uidx_pos = pos_top.index
            neg_top = neg_prob.sort_values(ascending=False).head(int((self.top_n/self.NPratio)*(self.de_ratio**(i+1))))
            uidx_neg = neg_top.index
            print("pos_top:min_prob=%f,mean_prob=%f"%(pos_top.min(),pos_top.mean()))
            print("neg_top:min_prob=%f,mean_prob=%f"%(neg_top.min(),neg_top.mean()))

            df_pos = df.loc[uidx_pos,:]
            df_neg = df.loc[uidx_neg,:]
            # print("pos_top: ss:%i, y_10:%i, y_30:%i"%(df_pos[df_pos['y_10'].isin([0,1])].shape[0],df_pos[df_pos['y_10'].isin([0,1])]['y_10'].sum(),df_pos[df_pos['y_10'].isin([0,1])]['y_30'].sum()))
            # print("neg_top: ss:%i, y_10:%i, y_30:%i"%(df_neg[df_neg['y_10'].isin([0,1])].shape[0],df_neg[df_neg['y_10'].isin([0,1])]['y_10'].sum(),df_neg[df_neg['y_10'].isin([0,1])]['y_30'].sum()))

#            uidx_pos = pos_prob[pos_prob > self.prob_threshold_pos].index #unlabeled判断为正例的样本
#            uidx_neg = neg_prob[neg_prob > self.prob_threshold_neg].index #unlabeled判断为负例的样本
            uidx = np.hstack((uidx_pos, uidx_neg))
            i += 1
            print ('iter: %i, n_pos: %i, n_neg: %i.'%(i, uidx_pos.shape[0],uidx_neg.shape[0])) #组合              
            print('uidx num: ',uidx.shape[0])
            
            self.uidx_pos[i] = uidx_pos
            self.uidx_neg[i] = uidx_neg
            # auc_10 = auc_score(self.dev_data[self.dev_data['y_10'].isin([0,1])]['y_10'], self.predict_proba(self.dev_data[self.dev_data['y_10'].isin([0,1])][self.predictors].values)[:,1])
            # auc_30 = auc_score(self.dev_data[self.dev_data['y_30'].isin([0,1])]['y_30'], self.predict_proba(self.dev_data[self.dev_data['y_30'].isin([0,1])][self.predictors].values)[:,1])
            # if auc_10>max_auc_10:
                # max_auc_10 = auc_10
                # stop_t=0
            # if auc_30>max_auc_30:
                # max_auc_30 = auc_30
                # stop_t=0
            # else:
                # stop_t+=1
            # print ('auc_10: %f'%auc_10)
            # print ('auc_30: %f'%auc_30)
            # self.auc[i] = [auc_10,auc_30]
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def auc_score(self, X, y):
        false_positive_rate, recall, thresholds = roc_curve(y,self.predict_proba(X))
        return auc(false_positive_rate, recall)


