class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

from sklearn.base import BaseEstimator
import numpy
import sklearn.metrics
from sklearn.linear_model import LogisticRegression as LR
import nlopt
import scipy.stats

class CPLELearningModel(BaseEstimator):
    """
    Contrastive Pessimistic Likelihood Estimation framework for semi-supervised 
    learning, based on (Loog, 2015). This implementation contains two 
    significant differences to (Loog, 2015):
    - the discriminative likelihood p(y|X), instead of the generative 
    likelihood p(X), is used for optimization
    - apart from `pessimism' (the assumption that the true labels of the 
    unlabeled instances are as adversarial to the likelihood as possible), the 
    optimization objective also tries to increase the likelihood on the labeled
    examples

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then uses global optimization to 
    find (soft) label hypotheses for the unlabeled examples in a pessimistic  
    fashion (such that the model log likelihood on the unlabeled data is as  
    small as possible, but the log likelihood on the labeled data is as high 
    as possible)

    See Loog, Marco. "Contrastive Pessimistic Likelihood Estimation for 
    Semi-Supervised Classification." arXiv preprint arXiv:1503.00269 (2015).
    http://arxiv.org/pdf/1503.00269

    Attributes
    ----------
    basemodel : BaseEstimator instance
        Base classifier to be trained on the partially supervised data

    pessimistic : boolean, optional (default=True)
        Whether the label hypotheses for the unlabeled instances should be
        pessimistic (i.e. minimize log likelihood) or optimistic (i.e. 
        maximize log likelihood).
        Pessimistic label hypotheses ensure safety (i.e. the semi-supervised
        solution will not be worse than a model trained on the purely 
        supervised instances)
        
    predict_from_probabilities : boolean, optional (default=False)
        The prediction is calculated from the probabilities if this is True 
        (1 if more likely than the mean predicted probability or 0 otherwise).
        If it is false, the normal base model predictions are used.
        This only affects the predict function. Warning: only set to true if 
        predict will be called with a substantial number of data points
        
    use_sample_weighting : boolean, optional (default=True)
        Whether to use sample weights (soft labels) for the unlabeled instances.
        Setting this to False allows the use of base classifiers which do not
        support sample weights (but might slow down the optimization)

    max_iter : int, optional (default=3000)
        Maximum number of iterations
        
    verbose : int, optional (default=1)
        Enable verbose output (1 shows progress, 2 shows the detailed log 
        likelihood at every iteration).

    """
    
    def __init__(self, basemodel, pessimistic=True, predict_from_probabilities = False, use_sample_weighting = True, max_iter=3000, verbose = 1):
        self.model = basemodel
        self.pessimistic = pessimistic
        self.predict_from_probabilities = predict_from_probabilities
        self.use_sample_weighting = use_sample_weighting
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.it = 0 # iteration counter 循环次数计数
        self.noimprovementsince = 0 # log likelihood hasn't improved since this number of iterations
        self.maxnoimprovementsince = 3 # threshold for iterations without improvements (convergence is assumed when this is reached) xx次迭代没有提升就停止
        
        self.buffersize = 200
        # buffer for the last few discriminative likelihoods (used to check for convergence)
        self.lastdls = [0]*self.buffersize
        
        # best discriminative likelihood and corresponding soft labels; updated during training
        self.bestdl = numpy.infty #初始最佳log-loss值
        self.bestlbls = [] #最佳的无标签的样本0-1概率
        
        # unique id
        self.id = str(chr(numpy.random.randint(26)+97))+str(chr(numpy.random.randint(26)+97))

    def discriminative_likelihood(self, model, labeledData, labelP_sup,unlabelP_sup,labeledy = None, unlabeledData = None, unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        """
        model:训练有无标签样本的模型
        labeledData:array-like, 有标签的X变量；
        labeledy：array-like, 有标签样本的标签；
        unlabeledData：array-like, 无标签的X变量；
        unlabeledWeights：array-like, 无标签样本的初始0-1概率；
        unlabeledlambda：float, 无标签样本的log-l的加权
        gradient：
        alpha：float, 学习率
        
        """
        unlabeledy = (unlabeledWeights[:, 0]<0.5)*1 #将无标签的概率转成0,1分类标签
        uweights = numpy.copy(unlabeledWeights[:, 0]) # large prob. for k=0 instances, small prob. for k=1 instances 复制
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence 生成权重，即1取1的概率，0取0的概率
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights))#与有标签的样本的权重stack起来
        labels = numpy.hstack((labeledy, unlabeledy)) #与有标签的样本标签stack起来
        
        #计算在有标签样本训练的模型下的log-likelihoods
        label_log_likeli_sup = -sklearn.metrics.log_loss(labeledy, labelP_sup)
        unlabel_log_likeli_sup = numpy.average((unlabeledWeights*numpy.log(unlabelP_sup)).sum(axis=1))
        
        # fit model on supervised data 训练无标签和有标签的样本（分有无样本权重训练）
        if self.use_sample_weighting:
            model.fit(numpy.vstack((labeledData, unlabeledData)), labels, sample_weight=weights)
        else:
            model.fit(numpy.vstack((labeledData, unlabeledData)), labels)
        
        # probability of labeled data 预测有标签样本的0-1概率
        P = model.predict_proba(labeledData)
        
        try:
            # labeled discriminative log likelihood
            labeledDL = -sklearn.metrics.log_loss(labeledy, P) #计算有标签样本的log-likelihoods
        except Exception as e:
            print(e)
            P = model.predict_proba(labeledData)

        # probability of unlabeled data
        unlabeledP = model.predict_proba(unlabeledData)  #预测无标签样本的0-1概率
           
        try:
            # unlabeled discriminative log likelihood
            eps = 1e-15
            unlabeledP = numpy.clip(unlabeledP, eps, 1 - eps) #将概率限制在0-1开区间
            unlabeledDL = numpy.average((unlabeledWeights*numpy.log(unlabeledP)).sum(axis=1)) #计算无标签样本的log likelihoods
        except Exception as e:
            print(e)
            unlabeledP = model.predict_proba(unlabeledData)
        
        #计算总样本的log-loss（为了min）
        if self.pessimistic:
            # pessimistic: minimize the difference between unlabeled and labeled discriminative likelihood (assume worst case for unknown true labels)
            dl = ( unlabeledDL + labeledDL-(label_log_likeli_sup+unlabel_log_likeli_sup)) 
        else: 
            # optimistic: minimize negative total discriminative likelihood (i.e. maximize likelihood) 
            dl =  (unlabeledDL + labeledDL+label_log_likeli_sup+unlabel_log_likeli_sup)
        
        return dl
        
    def discriminative_likelihood_objective(self, model, labeledData, labelP_sup,unlabelP_sup,labeledy = None, unlabeledData = None, unlabeledWeights = None, unlabeledlambda = 1, gradient=[], alpha = 0.01):
        """
        model:训练有无标签样本的模型
        labeledData:array-like, 有标签的X变量；
        labeledy：array-like, 有标签样本的标签；
        unlabeledData：array-like, 无标签的X变量；
        unlabeledWeights：array-like, 无标签样本的初始0-1概率；
        unlabeledlambda：float, 无标签样本的log-l的加权
        gradient：
        alpha：float, 学习率
        """
        if self.it == 0:
            self.lastdls = [0]*self.buffersize
        
        dl = self.discriminative_likelihood(model, labeledData, labelP_sup,unlabelP_sup,labeledy, unlabeledData, unlabeledWeights, unlabeledlambda, gradient, alpha) #此时即认为p(u,x|sita)已知，需要求最佳的q
        
        self.it += 1 #迭代次数+1
        self.lastdls[numpy.mod(self.it, len(self.lastdls))] = dl #即只保留最近200次迭代的log-loss
        
        if numpy.mod(self.it, self.buffersize) == 0: # or True: 即每隔200，检验一下log-loss有没有收敛
            #print (self.lastdls)
            improvement = numpy.mean((self.lastdls[int(len(self.lastdls)/2):])) - numpy.mean((self.lastdls[:int(len(self.lastdls)/2)])) #前100与后100 log-loss的均值差值
            # ttest - test for hypothesis that the likelihoods have not changed (i.e. there has been no improvement, and we are close to convergence) 
            _, prob = scipy.stats.ttest_ind(self.lastdls[int(len(self.lastdls)/2):], self.lastdls[:int(len(self.lastdls)/2)]) #均值的t检验
            
            # if improvement is not certain accoring to t-test...如果t检验不显著且log-loss没有降低，则noimprovement
            noimprovement = prob > 0.1 and numpy.mean(self.lastdls[int(len(self.lastdls)/2):]) < numpy.mean(self.lastdls[:int(len(self.lastdls)/2)])
            if noimprovement:
                self.noimprovementsince += 1 #noimprovement累积
                if self.noimprovementsince >= self.maxnoimprovementsince: #若noimprovementsince累积到e-stopping，则迭代停止
                    # no improvement since a while - converged; exit
                    self.noimprovementsince = 0
                    raise Exception(" converged.") # we need to raise an exception to get NLopt to stop before exceeding the iteration budget
            else:
                self.noimprovementsince = 0 #否则noimprovementsince每次归零，即e-stopping需重新累积
            
            if self.verbose == 2:
                print(self.id,self.it, dl, numpy.mean(self.lastdls), improvement, round(prob, 3), (prob < 0.1))
            elif self.verbose:
                sys.stdout.write(('.' if self.pessimistic else '.') if not noimprovement else 'n')
                      
        if dl < self.bestdl: #如果log-loss比最佳的要小，则替换
            self.bestdl = dl
            self.bestlbls = numpy.copy(unlabeledWeights[:, 0]) #替换无标签的0-1概率
                        
        return dl
    
    def fit(self, X, y, iterators=1): # -1 for unlabeled
        unlabeledX = X[y==-1, :] #取无标签样本的X变量
        labeledX = X[y!=-1, :] #取有标签样本的X变量
        labeledy = y[y!=-1] #取有标签样本的标签
        
        M = unlabeledX.shape[0] #无标签样本的样本数
        
        # train on labeled data
        self.model.fit(labeledX, labeledy) #有标签样本的训练

        unlabeledy = self.predict(unlabeledX) #对无标签样本的标签预测
        
        labelP_sup = self.model.predict_proba(labeledX)
        unlabelP_sup = self.model.predict_proba(unlabeledX)
        lblinit = numpy.random.random(len(unlabeledy)) #初始化
        
        #re-train, labeling unlabeled instances pessimistically
        
        # pessimistic soft labels ('weights') q for unlabelled points, q=P(k=0|Xu)，即q是标签为0的概率
        f = lambda softlabels, grad=[]: self.discriminative_likelihood_objective(self.model, labeledX, labelP_sup,unlabelP_sup,labeledy=labeledy, unlabeledData=unlabeledX, unlabeledWeights=numpy.vstack((softlabels, 1-softlabels)).T, gradient=grad) #- supLL
        
        try:
            self.it = 0 #迭代次数
            opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, M) #设定优化算法和优化变量维数
            opt.set_lower_bounds(numpy.zeros(M))#设定优化变量的下界
            opt.set_upper_bounds(numpy.ones(M)) #设定优化变量的上界
            opt.set_min_objective(f) #设定最小化的目标
            opt.set_maxeval(self.max_iter) #设定迭代次数
            self.bestsoftlbl = opt.optimize(lblinit) #以xx为起点开始优化，取得最佳值
            print(" max_iter exceeded.")
        except Exception as e:
            print(e)
            self.bestsoftlbl = self.bestlbls        
            
        if numpy.any(self.bestsoftlbl != self.bestlbls): #如果有迭代，则无标签最新概率替换
            self.bestsoftlbl = self.bestlbls
        ll = f(self.bestsoftlbl) #计算log-loss

        unlabeledy = (self.bestsoftlbl<0.5)*1 #生成无标签样本的0-1标签
        uweights = numpy.copy(self.bestsoftlbl) # large prob. for k=0 instances, small prob. for k=1 instances ，无标签样本权重更新
        uweights[unlabeledy==1] = 1-uweights[unlabeledy==1] # subtract from 1 for k=1 instances to reflect confidence 
        weights = numpy.hstack((numpy.ones(len(labeledy)), uweights)) #组合有无标签样本权重
        labels = numpy.hstack((labeledy, unlabeledy)) #组合有无标签样本的标签
        #根据无标签最新更新的标签，重新训练（所以q和sita是只迭代了一次？不是迭代了一次，而是每次q都改变，所以标签再分配，但是ll不应该小于0，否则效果会变差）
        if self.use_sample_weighting:
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels, sample_weight=weights)
        else:
            self.model.fit(numpy.vstack((labeledX, unlabeledX)), labels)
        
        if self.verbose > 1:
            print("number of non-one soft labels: ", numpy.sum(self.bestsoftlbl != 1), ", balance:", numpy.sum(self.bestsoftlbl<0.5), " / ", len(self.bestsoftlbl))
            print("current likelihood: ", ll)
        
        if not getattr(self.model, "predict_proba", None):
            # Platt scaling
            self.plattlr = LR()
            preds = self.model.predict(labeledX)
            self.plattlr.fit( preds.reshape( -1, 1 ), labeledy )
            
        return self
        
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        
        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            return self.plattlr.predict_proba(preds.reshape( -1, 1 ))
        
    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        
        if self.predict_from_probabilities:
            P = self.predict_proba(X)
            return (P[:, 0]<numpy.average(P[:, 0]))
        else:
            return self.model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return sklearn.metrics.accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
