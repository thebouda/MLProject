# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:57:27 2021

@author: simo6
"""

# from naivebayesMVG import tiedCov
import numpy
from PCA import PCAfunct
from GenerativeModels import MVG_classifier,MVG_log,NaiveBayesGaussianClassifier,TiedCovarianceGaussianClassifier,KFoldValidation
from confusionMatrix import KFoldValidationConfusionMatrix
from logisticRegression import KFoldValidationLogisticRegression

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




    

    


if __name__ == '__main__':
    DTR,LTR = load ('./Data/Train.txt')
    DTE,LTE = load('./Data/Test.txt')
    
    # D = numpy.hstack((DTR,DTE))
    D=DTR
    # L = numpy.hstack((LTR,LTE))
    L=LTR
    
    DTR_PCA,DTE_PCA = PCAfunct(DTR,LTR,DTE)
    DTE = DTE_PCA
    DTR = DTR_PCA    
    lambdaVector = [0.1e-4,1e-4, 2e-4, 4e-4, 6e-4, 10e-4, 40e-4, 100e-4]
    
# =============================================================================
#     predicted,shape = MVG_classifier(DTR,LTR,DTE,LTE)
#     print(predicted/shape)
#     
#     predicted,shape = MVG_log(DTR,LTR,DTE,LTE)
#     print(predicted/shape)
#     
#     predicted,shape = NaiveBayesGaussianClassifier(DTR,LTR,DTE,LTE)
#     print(predicted/shape)
#     
#     predicted,shape = TiedCovarianceGaussianClassifier(DTR,LTR,DTE,LTE)
#     print(predicted/shape)
# =============================================================================

    #KFoldValidation(D,L)
    #KFoldValidation(DTR,L) #PCA
    #KFoldValidationConfusionMatrix(D,L)
    #KFoldValidationConfusionMatrix(DTR,L) #PCA
    #KFoldValidationLogisticRegression(D,L,lambdaVector)
    KFoldValidationLogisticRegression(DTR,L,lambdaVector) #PCA

# analisi dei risultati 
# pensiamo perche possono essere diversi per tiedcov etc...
# confusion matrix

