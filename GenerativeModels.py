# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:10:43 2021

"""

import numpy
import scipy
from scipy import special


def getMu(D):
    mu = []
    for i in range(D.shape[0]):
        mui = (1/(D[i].size))*sum(D[i])
        mu.append(mui)
    return mu

def colvec(vec):
  return vec.reshape(vec.size,1)

def getSigma(D,mu):
    mu = colvec(mu)
    C=0
    for i in range (D.shape[1]):
        C=C+numpy.dot(D[:,i:i+1]-mu,(D[:,i:i+1] - mu ).T)

    C= C/float(D.shape[1])
    return C

def getSigmaI(D,mu):
    mu = colvec(mu)
    C=0
    for i in range (D.shape[1]):
        C=C+numpy.dot(D[:,i:i+1]-mu,(D[:,i:i+1] - mu ).T)

    C= numpy.multiply(C/float(D.shape[1]),numpy.identity(D.shape[0]))
    return C

def logpdf_GAU_ND(x, mu, C):
    x=colvec(x)
    lastTerm = 0.5*(numpy.dot(numpy.dot(((x-mu).T),(numpy.linalg.inv(C))),(x-mu)))
    det = numpy.linalg.slogdet(C)
    det = numpy.log(abs(det[1]))
    #ricalcolo determinante 
    det = numpy.log(numpy.linalg.det(C))
    m=x.size
    return (-m*0.5*numpy.log(numpy.pi*2)) - (0.5*det) - lastTerm



def MVG_classifier(DTR,LTR,DTE, LTE):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
   # D2 = DTR[:, LTR==2]
    
    mu0 = colvec(numpy.matrix(getMu(D0)))
    mu1 = colvec(numpy.matrix(getMu(D1)))
    #mu2 = colvec(numpy.matrix(getMu(D2)))
    
    sigma0 = getSigma(D0,mu0)
    sigma1 = getSigma(D1,mu1)
    #sigma2 = getSigma(D2,mu2)
   
    m_c = []
    s_c = []
    
    m_c.append(mu0)
    m_c.append(mu1)
    #m_c.append(mu2)
    s_c.append(sigma0)
    s_c.append(sigma1)
    #s_c.append(sigma2)
    
    S = numpy.zeros((2, DTE.shape[1]))
    
    for i in range(2):
       for j, sample in enumerate(DTE.T):
          S[i, j] = numpy.exp(logpdf_GAU_ND(sample, m_c[i], s_c[i]))
          
    SJoint = 1/2*S
    SSum = SJoint.sum(axis=0)
    SPost = SJoint/SSum
 
    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size
    return Predicted, DTE.shape[1]

def MVG_log(DTR,LTR,DTE, LTE):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
   # D2 = DTR[:, LTR==2]
    
    mu0 = colvec(numpy.matrix(getMu(D0)))
    mu1 = colvec(numpy.matrix(getMu(D1)))
    #mu2 = colvec(numpy.matrix(getMu(D2)))
    
    sigma0 = getSigma(D0,mu0)
    sigma1 = getSigma(D1,mu1)
    #sigma2 = getSigma(D2,mu2)
   
    m_c = []
    s_c = []
    
    m_c.append(mu0)
    m_c.append(mu1)
    #m_c.append(mu2)
    s_c.append(sigma0)
    s_c.append(sigma1)
    #s_c.append(sigma2)

    S = numpy.zeros((2, DTE.shape[1]))

    for i in range(2):
        for j, sample in enumerate(DTE.T):
            S[i, j] = logpdf_GAU_ND(sample, m_c[i], s_c[i])

    SJoint = numpy.log(1/2) + S
    SSum = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size

    return Predicted, DTE.shape[1]

def NaiveBayesGaussianClassifier(DTR, LTR, DTE, LTE):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    #D2 = DTR[:, LTR==2]
    
    mu0 = colvec(numpy.matrix(getMu(D0)))
    mu1 = colvec(numpy.matrix(getMu(D1)))
    #mu2 = colvec(numpy.matrix(getMu(D2)))
    
    sigma0 = getSigmaI(D0,mu0)
    sigma1 = getSigmaI(D1,mu1)
    #sigma2 = getSigmaI(D2,mu2)
   
    m_c = []
    s_c = []
    
    m_c.append(mu0)
    m_c.append(mu1)
   # m_c.append(mu2)
    s_c.append(sigma0)
    s_c.append(sigma1)
    #s_c.append(sigma2)

    S = numpy.zeros((2, DTE.shape[1]))

    for i in range(2):
        for j, sample in enumerate(DTE.T):
            S[i, j] = logpdf_GAU_ND(sample, m_c[i], s_c[i])

    SJoint = numpy.log(1/2) + S
    SSum = special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size

    return Predicted, DTE.shape[1]

def TiedCovarianceGaussianClassifier(DTR, LTR, DTE, LTE):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    # D2 = DTR[:, LTR==2]
    
    mu0 = colvec(numpy.matrix(getMu(D0)))
    mu1 = colvec(numpy.matrix(getMu(D1)))
    # mu2 = colvec(numpy.matrix(getMu(D2)))
    
    sigma0 = getSigmaI(D0,mu0)
    sigma1 = getSigmaI(D1,mu1)
    # sigma2 = getSigmaI(D2,mu2)
   
    m_c = []
    
    m_c.append(mu0)
    m_c.append(mu1)
    # m_c.append(mu2)
    
    SStar = (sigma0*D0.shape[1]+sigma1*D1.shape[1])/DTR.shape[1]

    S = numpy.zeros((2, DTE.shape[1]))

    for i in range(2):
        for j, sample in enumerate(DTE.T):
            S[i, j] = logpdf_GAU_ND(sample, m_c[i], SStar)

    SJoint = numpy.log(1/2) + S
    SSum = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SSum

    Predictions = SPost.argmax(axis=0) == LTE
    Predicted = Predictions.sum()
    NotPredicted = Predictions.size - Predicted
    acc = Predicted/Predictions.size

    return Predicted, DTE.shape[1]

def KFoldValidation(D,L):
    fileResults = open('Resultsfile.txt','w')
    fileResults.writelines('k \t mvg \t naivebayes \t tiedCov' '\n')
    K = 8 
    # N = int(D.shape[1]/K)
    classifiers = [(MVG_log, "Multivariate Gaussian Classifier"),(NaiveBayesGaussianClassifier, "Naive Bayes"),(TiedCovarianceGaussianClassifier, "Tied Covariance")]

    N = int(D.shape[1]/K)
    fileResults.writelines(str(K)+ ' \t ')         

    for j, (c, cstring) in enumerate(classifiers):
        nWrongPrediction = 0
        numpy.random.seed(j)
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
            nCorrectPrediction, nSamples = c(DTR,LTR, DTE, LTE)
            nWrongPrediction += nSamples - nCorrectPrediction


        errorRate = nWrongPrediction/D.shape[1]
        accuracy = 1 - errorRate
        fileResults.writelines(str(round(accuracy*100, 1)) + '\t')  


        print(f"{cstring} results:\nAccuracy: {round(accuracy*100, 1)}%\nError rate: {round(errorRate*100, 1)}%\n") 

    fileResults.close()

# kfold for different models
def KFoldValidationGenerativeModels(D,L):
    K = 8 
    # N = int(D.shape[1]/K)
    fileResults = open('ResultsfileGenerativeModelPCA.txt','w')
    fileResults.writelines('partition \t mvg \t mvglog \t naive \t tied \n')

    N = int(D.shape[1]/K)       

    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])

    # stored ac0curacies
    mvgPrecList=[]
    mvgLogPrecList=[]
    NaivePrecList=[]
    tiedPrecList=[]

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

        predictedMVG,shapeMVG = MVG_classifier(DTR,LTR,DTE,LTE)
        mvgPrec=predictedMVG/shapeMVG*100
        mvgPrecList.append(mvgPrec)


        predictedMVGLog,shapeMVGLog = MVG_log(DTR,LTR,DTE,LTE)
        mvgLogPrec=predictedMVGLog/shapeMVGLog*100
        mvgLogPrecList.append(mvgLogPrec)


        predictedNaive,shapeNaive = NaiveBayesGaussianClassifier(DTR,LTR,DTE,LTE)
        NaivePrec=predictedNaive/shapeNaive*100
        NaivePrecList.append(NaivePrec)


        predictedTied,shapeTied = TiedCovarianceGaussianClassifier(DTR,LTR,DTE,LTE)
        tiedPrec=predictedTied/shapeTied*100
        tiedPrecList.append(tiedPrec)

       
        fileResults.writelines(str(K)+" \t "+str(mvgPrec)+ " \t" + str(mvgLogPrec) +"\t"+ str(NaivePrec) +" \t "+str(tiedPrec) + "\n")
   
    # compute the mean
    fileResults.writelines("means \t " + str(numpy.mean(mvgPrecList))+" \t "+ str(numpy.mean(mvgLogPrecList))+" \t "+str(numpy.mean(NaivePrecList))+" \t "+str(numpy.mean(tiedPrecList))+" \n ")
    fileResults.close()

