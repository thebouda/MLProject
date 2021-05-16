# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:57:27 2021

@author: simo6
"""

# from naivebayesMVG import tiedCov
import numpy
import scipy
import matplotlib.pyplot as plt
from scipy import special


def mcol(v):
    return v.reshape((v.size,1))

int_to_label = {
    0 : 'fixed acidity',
    1 :'volatile acidity',
    2 : 'citric acid',
    3 : 'residual sugar',
    4 : 'chlorides',
    5 : 'free sulfur dioxide',
    6 : 'total sulfur dioxide',
    7 : 'density',
    8 : 'pH',
    9 : 'sulphates',
    10 : 'alcohol'
    }

int_to_label_quality = {
    0:'low_quality',
    1:'high_quality'
    }

def load(fname):
    DList = []
    labelsListQuality = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = mcol(numpy.array(attrs,dtype=numpy.float32))
                quality = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsListQuality.append(quality)
            except:
                pass
    return numpy.hstack(DList),numpy.array(labelsListQuality,dtype=numpy.int32)

def PCAfunct(D,L):
    mu = D.mean(1) #media
    DC = D - mu.reshape((mu.size, 1)) #centrare la matrice
    DCT = DC.T #trasposta
    Ctmp=(1/DC.shape[1])*DC
    C=numpy.dot(Ctmp,DCT) #calcolo la covarianza
    s, U = numpy.linalg.eigh(C) #autovalori e autovettori
    P = U[:, ::-1][:, 0:2]
    DP = numpy.dot(P.T, D)
    plot_scatter(DP, L)
    
def plot_scatter(D,L):  
    
    for i in range(2):
        to_plot = D[:, L==i]
        plotted = plt.scatter(to_plot[0,:], to_plot[1,:]) # Plot Magic
        plotted.set_label(int_to_label_quality[i])
    plt.legend()
    plt.show()
    
def getMu(D):
    mu = []
    for i in range(D.shape[0]):
        mui = (1/(D[i].size))*sum(D[i])
        mu.append(mui)
    return mu

def colvec(vec):
  return vec.reshape(vec.size,1)

def getSigma(D,mu):
    sigma = []
    mu = colvec(mu)
    C=0
    for i in range (D.shape[1]):
        C=C+numpy.dot(D[:,i:i+1]-mu,(D[:,i:i+1] - mu ).T)

    C= C/float(D.shape[1])
    return C

def getSigmaI(D,mu):
    sigma = []
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
    


if __name__ == '__main__':
    DTR,LTR = load ('./Data/Train.txt')
    DTE,LTE = load('./Data/Test.txt')
    
    # D = numpy.hstack((DTR,DTE))
    D=DTR
    # L = numpy.hstack((LTR,LTE))
    L=LTR
    
    #PCAfunct(DTR,LTR)
    
    # predicted,shape = MVG_classifier(DTR,LTR,DTE,LTE)
    # print(predicted/shape)
    
    # predicted,shape = MVG_log(DTR,LTR,DTE,LTE)
    # print(predicted/shape)
    
    # predicted,shape = NaiveBayesGaussianClassifier(DTR,LTR,DTE,LTE)
    # print(predicted/shape)
    
    # predicted,shape = TiedCovarianceGaussianClassifier(DTR,LTR,DTE,LTE)
    # print(predicted/shape)
    
    fileResults = open('Resultsfile.txt','w')
    fileResults.writelines('k \t mvg \t naivebayes \t tiedCov' '\n')

    K = 40 #30    ddalle 2 alle k = 120 
    # N = int(D.shape[1]/K)
    classifiers = [(MVG_log, "Multivariate Gaussian Classifier"),(NaiveBayesGaussianClassifier, "Naive Bayes"),(TiedCovarianceGaussianClassifier, "Tied Covariance")]
    for kappa in range(2,K):
        N = int(D.shape[1]/kappa)
        fileResults.writelines(str(kappa)+ ' \t ')         

        for j, (c, cstring) in enumerate(classifiers):
            nWrongPrediction = 0
            numpy.random.seed(j)
            indexes = numpy.random.permutation(D.shape[1])
            for i in range(kappa):
        
                idxTest = indexes[i*N:(i+1)*N]
        
                if i > 0:
                    idxTrainLeft = indexes[0:i*N]
                elif (i+1) < kappa:
                    idxTrainRight = indexes[(i+1)*N:]
        
                if i == 0:
                    idxTrain = idxTrainRight
                elif i == kappa-1:
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
        fileResults.writelines('\n')

    fileResults.close()
    
    
# analisi dei risultati 
# pensiamo perche possono essere diversi per tiedcov etc...
# confusion matrix