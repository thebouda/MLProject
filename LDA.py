"""
Created on Thu May  6


"""

#%% import libraries
import sys
import numpy
import scipy.linalg 
import matplotlib.pyplot as plt
import matplotlib


# file imports
from dataRepresentation import load

"""         start of code       """

#
def colvec(vec):
  return vec.reshape(numpy.size(vec,0),1)

def vrow(vec):
  return vec.reshape(1,numpy.size(vec,0))


# calculate the covariance matrix and return the mean
def covMatrix(D,L):
    mu= D.mean(1) # mu is a 1d array rather than a col vector as done Previously
    mu = colvec(mu)
    C=0
    for i in range (D.shape[1]):
        C=C+numpy.dot(D[:,i:i+1]-mu,(D[:,i:i+1] - mu ).T)
    C= C/float(D.shape[1])
    
    DC= D-mu
    Cdot = numpy.dot(DC,DC.T)/numpy.size(D,1)

    return mu, C

# calculate the lda
def ldaCalcu(mu,d,l):
  #calculate the between class
  sb=0
  for c in range( numpy.size(set(l)) ):
    di=d[:,l==c] # we select the class
    mui = di.mean(1)
    sb= sb + numpy.size(di,1)*numpy.dot(colvec(mui)-mu ,( colvec(mui)-mu).T)
  sb=sb/numpy.size(d,1)

  # within class covariance
  sw=0
  swc=0
  for c in range (numpy.size(set(l))):
    di=d[:,l==c]
    mui=colvec(di.mean(1))
    for i in range (di.shape[1]-1):
      swc=swc + numpy.dot(colvec(di[:,i])-mui,(colvec(di[:,i])-mui).T)    
    sw= sw+swc
    swc=0 # reset for the next one
  sw=sw/d.shape[1]

  return sb,sw

# get the direction of the lda
def directionsLDA(sb,sw,D):
  s,U=scipy.linalg.eigh(sb,sw)
  w=U[:,::-1]
  dp = numpy.dot(w.T,D)
  return dp
# extracting the values of the wines


 

# diagonalization of the matrix
def diagonalization(sw,sb,D):
  U, s, _ = numpy.linalg.svd(sw)
  P1 = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)
  sbt=numpy.dot(numpy.dot(P1,sb),P1.T)
  s2,P2 = numpy.linalg.eigh(sbt)
  P2 = P2[:,::-1] # reorder
  w =numpy.dot(P1.T,P2) # transformation matrix
  allTrans = numpy.dot(w.T,D)

  return allTrans



def drawings(D,L):
  theCarac={
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
  
  theAtt={
    0:'low_quality',
    1:'high_quality'
  }
  l0 = (L==0)
  l1 = (L==1)

  D0 = D[:,l0] 
  D1 = D[:,l1]
  
  # we represent the matrix of caracters
  fig,axs =plt.subplots(len(theCarac),len(theCarac))
  fig,axs =plt.subplots()

  for carac in theCarac:
    for secondCarac  in theCarac:
      if (secondCarac >carac):

        plt.figure()
        plt.scatter(D0[carac,:],D0[secondCarac,:], label=theAtt[0])
        plt.scatter(D1[carac,:],D1[secondCarac,:], label=theAtt[1])
        plt.legend()
        plt.title('x='+ theCarac[carac] + ' y= '+ theCarac[secondCarac] )
        plt.show()
        fileName = 'x='+ theCarac[carac] + ' y= '+ theCarac[secondCarac] +'SECONDLDA'
        plt.savefig(fileName) # save the files
  

if __name__ == '__main__':
    
  # load the data
  DTrain,LTrain =load('Data/Train.txt')

  #compute the calculations
  # get the covariance matrix
  muTrain, Ctrain =covMatrix(DTrain,LTrain)

  # get the lda values
  sbTrain,swTrain=ldaCalcu(muTrain,DTrain,LTrain)



  ldaDirections=directionsLDA(sbTrain,swTrain,DTrain)

  # get the representation with the lda
  drawings(ldaDirections,LTrain)
  # get the representation without the lda
  drawings(DTrain,LTrain)

  transformedMatrix=diagonalization(swTrain,sbTrain,DTrain)

  # draw results
  drawings(transformedMatrix,LTrain)

