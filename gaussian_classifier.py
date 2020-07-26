#
# Version 0.9  (HS 09/03/2020)
#

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# ================================================
#                   TASK 1_1
# ================================================

def task1_1(X, Y):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Variables to save
    #  S : D-by-D covariance matrix (np.double) to save as 't1_S.mat'
    #  R : D-by-D correlation matrix (np.double) to save as 't1_R.mat'
    S = myCov(X)
    R = myCor(X,S)

    scipy.io.savemat('t1_S.mat', mdict={'S': S})
    scipy.io.savemat('t1_R.mat', mdict={'R': R})

def myMean(X):
    # Input:
    #  X : N-by-D data matrix (np.double)
    # Output:
    #  Mean : D-by-1 mean vector (np.double)
    N = X.shape[0]
    return (1/N)*X.sum(axis=0)

def myCov(X):
    # Input:
    #  X : N-by-D data matrix (np.double)
    # Output
    #  S : D-by-D covariance matrix (np.double) 
    N = X.shape[0]
    D = X.shape[1]
    mean = myMean(X)
    S = np.array((1/N)*(np.dot((X-mean).T,(X-mean))),dtype=np.double)
    return S

def myCor(X,S):
    # Input:
    #  X : N-by-D data matrix (np.double)
    #  S : D-by-D covariance matrix (np.double) 
    # Output
    #  R : D-by-D correlation matrix (np.double) 
    D = X.shape[1]
    R = np.zeros((D,D), dtype=np.double)
    for i in range(D):
        for j in range(D):
            R[i][j] = S[i][j]/(S[i][i]*S[j][j])**(0.5)
    return R

# ================================================
#                   TASK 1_3
# ================================================
def task1_3(Cov):
    # Input:
    #  Cov : D-by-D covariance matrix (np.double)
    # Variales to save:
    #  EVecs : D-by-D matrix of column vectors of eigen vectors (np.double)  
    #  EVals : D-by-1 vector of eigen values (np.double)  
    #  Cumvar : D-by-1 vector of cumulative variance (np.double)  
    #  MinDims : 4-by-1 vector (np.int32)  
   
    EVals, EVecs = np.linalg.eig(Cov)
    sorted_indices = np.argsort(EVals)[::-1]
    EVals = EVals[sorted_indices]
    EVecs = EVecs[:,sorted_indices]
 
    # Making sure first element of each eigenvector is non-negative
    D = len(EVecs)
    for i in range(D):
        if EVecs[:,i][0] < 0:
            EVecs[:,i] = -1*EVecs[:,i]

    Cumvar = np.zeros(D, dtype=np.double)
    for i in range(D):
        Cumvar[i] = np.sum(EVals[:i+1])

    Cumvar_percent = Cumvar *(100/Cumvar[D-1])
    MinDims = minDim(Cumvar_percent)
    
    scipy.io.savemat('t1_EVecs.mat', mdict={'EVecs': EVecs})
    scipy.io.savemat('t1_EVals.mat', mdict={'EVals': EVals})
    scipy.io.savemat('t1_Cumvar.mat', mdict={'Cumvar': Cumvar})
    scipy.io.savemat('t1_MinDims.mat', mdict={'MinDims': MinDims})

def minDim(cum_percent):
    # Input:
    #  cum_percent : D-by-D cumuative variance in percentages matrix (np.double)
    #  Output: 
    #  MinDims : 4-by-1 vector (np.int32)  
    val1 = next(x for x, val in enumerate(cum_percent) if val > 70) + 1
    val2 = next(x for x, val in enumerate(cum_percent) if val > 80) + 1
    val3 = next(x for x, val in enumerate(cum_percent) if val > 90) + 1 
    val4 = next(x for x, val in enumerate(cum_percent) if val > 95) + 1
    return np.array([val1,val2,val3,val4],dtype=np.int32)

# ================================================
#                   TASK 1_4
# ================================================
def task1_mgc_cv(X, Y, CovKind, epsilon, Kfolds):
    # Input:
    #  X : N-by-D matrix of feature vectors (np.double)
    #  Y : N-by-1 label vector (np.int32)
    #  CovKind : scalar (np.int32)
    #  epsilon : scalar (np.double)
    #  Kfolds  : scalar (np.int32)
    #
    # Variables to save
    #  PMap   : N-by-1 vector of partition numbers (np.int32)
    #  Ms     : C-by-D matrix of mean vectors (np.double)
    #  Covs   : C-by-D-by-D array of covariance matrices (np.double)
    #  CM     : C-by-C confusion matrix (np.double)

    C = len(np.unique(Y))
    PMap = classwise_partition(Y,Kfolds)    
    part,counts = np.unique(PMap,return_counts=True) 
    scipy.io.savemat('t1_mgc_'+str(Kfolds)+'cv_PMap.mat', mdict={'PMap':PMap})
    final_CM = np.zeros((C,C),dtype=np.double)

    for i in range(Kfolds):
        p = np.where(PMap==i+1)[0]
        test_x = np.take(X,p,axis=0)
        test_y = np.take(Y,p)
        train_x = np.delete(X,p,axis=0) 
        train_y = np.delete(Y,p)
        Ms = class_mean(train_x,train_y)
        Covs = class_cov(train_x,train_y,Ms,CovKind,epsilon) 
        CM = gaussian_classifier(Ms,Covs, test_x,test_y)
        final_CM += CM * (1.0/test_x.shape[0])
        fold = i+1 
    
        scipy.io.savemat('t1_mgc_'+str(Kfolds)+'cv'+str(fold)+'_Ms.mat', mdict={'Ms':Ms})
        scipy.io.savemat('t1_mgc_'+str(Kfolds)+'cv'+str(fold)+'_ck'+str(CovKind)+'_Covs.mat', mdict={'Covs': Covs})
        scipy.io.savemat('t1_mgc_'+str(Kfolds)+'cv'+str(fold)+'_ck'+str(CovKind)+'_CM.mat', mdict={'CM':CM})
        
    # Average confusion matrix
    final_CM = final_CM *(1.0/Kfolds)
    L = Kfolds + 1
    scipy.io.savemat('t1_mgc_'+str(Kfolds)+'cv'+str(L)+'_ck'+str(CovKind)+'_CM.mat', mdict={'CM':final_CM});
    
def classwise_partition(Y,k):
    # Input:
    #  Y : N-by-1 label vector (np.int32)
    #  k  : Kfolds scalar (np.int32)
    #
    # Output:
    # mappings: N-by-1 kfold labels (np.int32)
    classes,counts = np.unique(Y,return_counts=True)
    class_count = dict(zip(classes, counts))
    partitions = [[] for i in range(k)]
   
    for c in classes:
        Nc = class_count[c]
        Mc = int(np.floor(Nc / float(k)))
        indices = np.where(Y==c)[0]
        for i in range(k-1):
            partitions[i].extend(indices[:Mc])
            indices = indices[Mc:]
        partitions[k-1].extend(indices) # last partiton gets remaining items

    N = Y.shape[0]
    mappings = np.zeros(N,dtype=np.int32)
    for p in range(len(partitions)):
        for index in partitions[p]:
            mappings[index] = p+1
    return mappings 

def class_mean(X,Y):
    # Input:
    #  X : N-by-D matrix of feature vectors (np.double)
    #  Y : N-by-1 label vector (np.int32)
    # Output:
    # C-by-D class means vector (np.double)
    classes = np.unique(Y)
    mean_matrix = []
    for i in classes:
        indices = np.where(Y==i)[0]
        class_items = np.take(X,indices, axis=0)
        n = class_items.shape[0]
        c_mean = (1/n)*class_items.sum(axis=0)
        mean_matrix.append(c_mean.tolist())

    return np.array(mean_matrix,dtype=np.double)

def class_cov(X,Y,M,mode,E):
    # Input:
    #  X : N-by-D matrix of training feature vectors (np.double)
    #  Y : N-by-1 training label vector (np.int32)
    #  M : C-by-D matrix of mean vectors (np.double)
    #  mode : (CovKind) scalar (np.int32)
    #  E : (epsilon) scalar (np.double)
    #  Ouptut:
    #  cov_matrix   : C-by-D-by-D array of covariance matrices (np.double)
   
    # Covariance Calculation using MLE
    classes = np.unique(Y)
    cov_list = []
    for c in classes:
        indices = np.where(Y==c)[0]
        class_items = np.take(X,indices,axis=0)
        Nc = class_items.shape[0]
        Mc = M[c-1]
        Sc = (1/Nc)*(np.dot((class_items-Mc).T,(class_items-Mc)))
        cov_list.append(Sc)

    cov_matrix = np.array(cov_list,dtype=np.double)
    reg_matrix = reg_ep(cov_matrix,E)
    if mode == 1:
        return reg_matrix
    elif mode == 2:
        diag = np.copy(reg_matrix)
        for c in classes:
            diag[c-1] = np.diag(np.diag(diag[c-1]))
        return diag
    else:
        k = len(classes)
        d = cov_matrix.shape[1]
        S = (1/k)*(np.sum(cov_matrix,axis=0))
        shared_var = np.zeros((k,d,d),dtype=np.double)
        for i in range(k):
            shared_var[i] = S
        return reg_ep(shared_var,E)

def reg_ep(cov_matrix,e):
    # Input:
    # cov_matrix: C-by-D-by-D array of covariance matrices (np.double)
    # e : (epsilon) scalar (np.double)
    # Output:
    # reg_matrix: C-by-D-by-D array of regularized covariance matrices (np.double)
    # Performs regularization for each class cov matrix
    C = cov_matrix.shape[0]
    D = cov_matrix.shape[1]
    e_matrix = e * np.eye(D,dtype=np.double)
    copy_matrix = np.copy(cov_matrix)
    for i in range(C):
        copy_matrix[i] = cov_matrix[i] + e_matrix
    return copy_matrix

def gaussian_classifier(c_means, covs, testX, testY):
    # Input: 
    # c_means : C-by-D matrix of mean vectors (np.double)
    # covs    : C-by-D-by-D array of covariance matrices (np.double)
    # testX   : P-by-D matrix of feature vectors (np.double)
    # testY   : P-by-1 N-by-1 label vector (np.int32)
    # Ouput :
    # CM      : C-by-C confusion matrix (np.double)
    P = testX.shape[0]
    C = c_means.shape[0]
    prior_prob = float(1/C)
    classifier_output = []
    for i in range(P):
        x = testX[i]
        class_probs = [] 
        for c in range(C):
            # LL(x|m,S) = log(P(x|m,s))
            LL = log_likelihood(x,c_means[c],covs[c])
            # p is proportional to the log prosterior prob.
            p = LL + np.log(prior_prob)
            class_probs.append(p)
        classifier_output.append(class_probs.index(max(class_probs))+1) 
    return confusion_matrix(testY,np.asarray(classifier_output),C)

def log_likelihood(x,m,S):
    # x : D-by-1 feature vectors (np.double)
    # m : D-by-1 mean vector (np.double)
    # S : D-by-D array of covariance matrixs (np.double)
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    temp =np.dot(np.dot((x-m).T,inv),(x-m))
    ll = -(0.5*temp)-(0.5*np.log(det))
    return ll

def confusion_matrix(trueY,predictedY,c_labels):
    # trueY : P-by-1 label vector (np.int32)
    # predictedY : P-by-1 N-by-1 label vector (np.int32)
    # c_labels : number of classes (int)
    # Where P = number of samples in test set
    cm = np.zeros((c_labels,c_labels), dtype = np.double)
    for true_class, predicted_class in zip(trueY,predictedY):
        # Subtract 1 in index because class labels are between 1 and 10
        cm[true_class-1][predicted_class-1] += 1
    return cm

