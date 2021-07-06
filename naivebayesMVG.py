import sys
import numpy
import scipy.linalg 
import matplotlib.pyplot as plt
import matplotlib


# file imports
from dataRepresentation import load


def colvec(vec):
  return vec.reshape(numpy.size(vec,0),1)

def vrow(vec):
  return vec.reshape(1,numpy.size(vec,0))


# compute the variances for each class
def computeVarMed(d,l):
  var1=[]
  mu1=[]
  # for the number of classes
  for i in range(len(set(l))): 
      d0=d[:,l==i]
      mu,var=covMatrix(d0)   
      mu1.insert(i,mu)
      var1.insert(i,var)
  
  return (mu1[0],var1[0]),(mu1[1],var1[1])

#compute covariance matrix 
def covMatrix(D):
    mu= D.mean(1) # mu is a 1d array rather than a col vector as done Previously
    mu = colvec(mu)
    C=0
    for i in range (D.shape[1]):
        C=C+numpy.dot(D[:,i:i+1]-mu,(D[:,i:i+1] - mu ).T)
    C= C/float(D.shape[1])
    
    DC= D-mu
    Cdot = numpy.dot(DC,DC.T)/numpy.size(D,1)

    return mu, C


# computes the log of the probability density function
def logpdf_GAU_ND(x,mu,C):
    
    inversedC= numpy.linalg.inv(C)
    _,detC=numpy.linalg.slogdet(C)
    centeredX=x-mu
    MatDo=0.5*numpy.dot(numpy.dot(centeredX.T,inversedC),centeredX)
    m=x[:,0].size
    logn= -m*0.5*numpy.log(numpy.pi*2)-0.5*detC- MatDo
    
    return numpy.diag(logn)
def calculatePriors(D,L,c):
    return numpy.size(D[:,L == c] )/numpy.size(D)

# funciton that calculates the mvg
def mvg(DTrain,LTrain,DTest,LTest):

  # compute var and mu for each class 
  (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)

  # priors of classes 0 and 1
  # prior0 = calculatePriors(DTrain,LTrain,0)
  prior0 =0.5
  prior1 =1- prior0
    
  likeVar0=logpdf_GAU_ND(DTest,mu0,var0)*prior0 # by assigning priors it does not ameliorate
  likeVar1=logpdf_GAU_ND(DTest,mu1,var1)*prior1
  
  S=numpy.vstack((likeVar0,likeVar1)) 
  SJoint =S # we have to multply with the prior probs
  Predic= numpy.argmax(SJoint,axis=0)
  Acc= LTest ==Predic # we get the values that are the same as we expect
    
  # count how many are true to get the accuracy
  AccuracyPred= sum(Acc==True)/numpy.size(Predic)
  return AccuracyPred


def naivebayes(DTrain,LTrain,DTest,LTest):
  # calculate variances
  (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)

  #diagonalize variances
  var0Diag = numpy.multiply(numpy.identity(11),var0)
  var1Diag =numpy.multiply(numpy.identity(11),var1)

  # prior0 = calculatePriors(DTrain,LTrain,0)
  prior0 =0.5
  prior1 =1- prior0

  likeVar1=logpdf_GAU_ND(DTest,mu0,var0Diag)*prior0
  likeVar2=logpdf_GAU_ND(DTest,mu1,var1Diag)*prior1

  SMatrix=numpy.vstack((likeVar1,likeVar2)) #S matrix
  SJoint=SMatrix # because we already mutiplied by the prior

  # we et the max value in order to predict of the prorbabilities calcualted 
  predic= numpy.argmax(SJoint,axis=0)
  acc= LTest ==predic # we get the values that are the same as we expect
  # count how many are true to get the accuracy
  accuracyPred= sum(acc==True)/numpy.size(predic)
  #err=1-AccuracyPre

  return accuracyPred

def secMethodTiedCov(var1,var2,D,L):
    tiedVar=0
    newVars= [var1,var2]
    for cClass in range(numpy.size(list(set(L)))): #with set we get the distinct values, we need it for the number of classes
        tiedVar= tiedVar + newVars[cClass]*numpy.size(D[:,L==cClass],1) # Nc*Variance for that class
    tiedVar = tiedVar / numpy.size(D,1)
    return tiedVar

def tiedCov(DTrain,LTrain,DTest,LTest):

  (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)
  defVar= secMethodTiedCov(var0,var1,DTrain,LTrain)

  # prior0 = calculatePriors(DTrain,LTrain,0)
  prior0 =0.5
  prior1 =1- prior0

  likeVar1=logpdf_GAU_ND(DTest,mu0,defVar)*prior1
  likeVar2=logpdf_GAU_ND(DTest,mu1,defVar) *prior0

  S=numpy.vstack((likeVar1,likeVar2)) #S matrix
  # SJoint=S*1/2
  SJoint=S
  predic= numpy.argmax(SJoint,axis=0)
  acc= LTest==predic # we get the values that are the same as we expect
  # count how many are true to get the accuracy
  accuracyPred= sum(acc==True)/numpy.size(predic)

  return accuracyPred

if  __name__ == '__main__':
    # loaded data
    DTrain,LTrain =load('Data/Train.txt')
    DTest,LTest =load('Data/Test.txt')

    accMVG = mvg(DTrain,LTrain,DTest,LTest)
    accNaiveBayes = naivebayes(DTrain,LTrain,DTest,LTest)
    accTiedCov = tiedCov(DTrain,LTrain,DTest,LTest)
    #INTERESTING HOW IF we take 0.5 insteade of Nc/N we get better results
    #it seems that asigning 0.5 probability is fair 

    print(accMVG)
    print(accNaiveBayes)
    print(accTiedCov)





