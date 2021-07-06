# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:48:57 2021

"""

# code for computing logistic regression
import numpy
import scipy.optimize

class logRegClass():
    def __init__(self,DTR,LTR):
        self.DTR =DTR
        self.LTR =LTR
        # self.l = l
        
    def logreg_obj(self,v):
        w,b = v[0:-1],v[-1]
        # we use numpylog1p to compute with more stability log(1+x)
        # we do the 2d equation
        first = numpy.square(numpy.linalg.norm(w)) *0.5 *self.l
        #normalizer = self.l * (w * w).sum() / 2

        zi = 2*self.LTR-1
        exp1 = -zi * (w.T.dot(self.DTR)+b)
        second = numpy.log1p(numpy.exp(exp1)).mean()

        jwb =first + second
        return jwb
    
    def setlamb(self,lamb):
        self.l=lamb
        
def KFoldValidationLogisticRegression(D,L,lambdaChosen):
    K = 8 
    # N = int(D.shape[1]/K)


    N = int(D.shape[1]/K)       

    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])
    for i in range(K):

        idxTest = indexes[i*N:(i+1)*N]

        if i > 0:
            idxTrainLeft = indexes[0:i*N]
        elif (i+1) < K:
            idxTrainRight = indexes[(i+1)*N:]

        if i == 0:
            idxTrain = idxTrainRight
        elif i == K-1:
            idxTrain = idxTrainLeft
        else:
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
        
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        l=lambdaChosen[i]
        logisticRegression(DTR,LTR, DTE, LTE,i,l)


def logisticRegression(DTrain,LTrain,DTest,LTest,i,l):
    #logistic regression
     # loaded data

    #---- binary logistic regression
    print ('\n')
    print("--------partition"+str(i)+"-------")
    
    DTR =DTrain
    LTR= LTrain

    DTE =DTest
    LTE = LTest

    #v(shape (d+1,) where d is the dimensionality of the space)
    # lambda l is a hyperparameter, so we can choose many values and see which is the optmimum
    logRegFunc = logRegClass(DTR,LTR)
    #lamToTest = numpy.array([8e-4,8.3e-4,8.6e-4])
    #for l in lamToTest:
    print(l)
    logRegFunc.setlamb(l)

    v=numpy.zeros(DTR.shape[0]+1)
    # jwb=logreg_obj(v,DTR,LTR,l)
    x,fx,d= scipy.optimize.fmin_l_bfgs_b(func = logRegFunc.logreg_obj,x0=v,approx_grad=True, factr=5000, maxfun=20000)
    print(x)
    print(fx)
    w,b = x[0:-1],x[-1]
    s = w.T.dot(DTE) + b

    # #predicted labels array, if> 0 then 1
    LP = s>0
    accuracy= (LTE == LP).mean()
    errorRate = (1-accuracy)*100
    print("error %.8f for function with lambda %f " % (errorRate,l))