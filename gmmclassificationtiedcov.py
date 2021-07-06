import numpy
import scipy.optimize
# from GMM_load import load_gmm
import matplotlib.pyplot as plt
# import sklearn.datasets as skl

from gmmclassificationgeneralfunctions import gmmValues, computeClassifications,mcol,computeNewCovar,SplitGMM,logpdf_GAU_ND,logpdf_GMM,Estep


# similar to the em algorithm  but different constrint for the covarinace
def TiedEMalgorithm(X, gmm):
    # flag is used to exit the if llr1-llr2 < threshold in this case 1e-6
    flag = True
    
    while(flag):
      
        (logdens, S) = logpdf_GMM(X, gmm)
        LLR1 = numpy.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = TiedMstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        LLR2 = numpy.sum(logdens)/X.shape[1]
        if (LLR2-LLR1 < 10**(-6)):
            flag = False
    return gmm

def TiedMstep(X, S, posterior):
    psi = 0.01

    Zg = numpy.sum(posterior, axis=1)  # 3
  
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum

    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * numpy.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum

    mu = Fg / Zg
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))

    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu

    tsum = numpy.zeros((cov.shape[1], cov.shape[2]))
    for g in range(S.shape[0]):
        tsum += Zg[g]*cov[g]
        
    for g in range(S.shape[0]):
        cov[g] = 1/X.shape[1] * tsum
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, mcol(s)*U.T)

    w = Zg/numpy.sum(Zg)
    return (w, mu, cov)


def LBTiedCov(x,gmm,alpha,iterations,minEigen):
    psi = minEigen
    gmmDiag = [(gmm[0][0],gmm[0][1],computeNewCovar(gmm[0][2],psi))]

    for i in range(iterations):
      
        gmmDiag = SplitGMM(gmmDiag,alpha)
        gmmDiag = TiedEMalgorithm(x,gmmDiag)

    return gmmDiag

def CovMatrixTied(gmmDiag,sigma):
    gmm = []
    for i in range(len(gmmDiag)):
        gmm.append((gmmDiag[i][0],gmmDiag[i][1],sigma))
    return gmm


def KFoldValidationTiedGMMCovariance(D,L,alpha,minEigen,gmms):
    K = 8 

    N = int(D.shape[1]/K)       

    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1])
    errorRates=[]

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

        # separate from classes
        
        DTR0 =DTR[:,LTR ==0] 
        DTR1 = DTR[:,LTR == 1]
   
        # here start the classification
        # Dtotal=[DTR0,DTR1,DTR2,DTR3,DTR4, DTR5, DTR6, DTR7, DTR8, DTR9, DTR10]
        
        Dtotal = [DTR0,DTR1]
        errors = computeClassifications(gmms,Dtotal,LBTiedCov,minEigen,alpha,DTE,LTE)
        errorRates.append(errors)
        print(errorRates)
    npErrors = numpy.array(errorRates)

    # emean precision for 1 gmms
    pre1= npErrors[:,0][:,1].sum()/len(npErrors[:,0][:,1])
    # emean precision for 2 gmms
    pre2= npErrors[:,1][:,1].sum()/len(npErrors[:,0][:,1])

    # emean precision for 4 gmms
    pre4= npErrors[:,2][:,1].sum()/len(npErrors[:,0][:,1])

    # emean precision for 8 gmms
    pre8= npErrors[:,3][:,1].sum()/len(npErrors[:,0][:,1])

    # emean precision for 16 gmms
    pre16= npErrors[:,4][:,1].sum()/len(npErrors[:,0][:,1])
    
    print("error full cov:\n")
    print("gmm1: "+ str(pre1))
    print("gmm2: "+ str(pre2))
    print("gmm4: "+ str(pre4))
    print("gmm8: "+ str(pre8))
    print("gmm16: "+ str(pre16))

    