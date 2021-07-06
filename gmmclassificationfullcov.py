from os import error
import numpy
import scipy.optimize
# from GMM_load import load_gmm
import matplotlib.pyplot as plt
# import sklearn.datasets as skl



from gmmclassificationgeneralfunctions import gmmValues, computeClassifications,mcol,computeNewCovar,SplitGMM,logpdf_GAU_ND,logpdf_GMM,Estep

def split_db_2to1(D, L, seed=0):
    # Get an integer nTrain representing 2/3 of the dataset dimension
    nTrain = int(D.shape[1]*2.0/3.0)
    # Generate a random seed
    numpy.random.seed(seed)
   
    idx = numpy.random.permutation(D.shape[1])
    # In idxTrain we select only the first 100 elements of idx
    idxTrain = idx[0:nTrain]
    # In idxEval we select only the last 50 elements of idx
    idxEval = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxEval]
    LTR = L[idxTrain]
    LTE = L[idxEval]
    return (DTR, LTR), (DTE, LTE)


# def ComputeEstep (s,logdensity): # computes the responsabilities
#     return numpy.exp(s-logdensity)



def EMAlgo(X, gmm):
    
    flag = True
  
    while(flag):
        # log  densities 
        (logdens, S) = logpdf_GMM(X, gmm)
        # average loglikelihood
        LLR1 = numpy.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        # responsabilities
        posterior = Estep(logdens, S)

        # ------ M-step ----------
        (w_gt_1, mu_gt_1, sigma_gt_1) = ComputeMstep(X, S, posterior)

        # update model params
        gmm =gmmValues(mu_gt_1,sigma_gt_1,w_gt_1,gmm)

        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        LLR2 = numpy.sum(logdens)/X.shape[1]

        if (LLR2-LLR1 < 10**(-6)):
            flag = False
    return gmm

def ComputeMstep(X, S, posterior):
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

    # constraint for the covariance
    for g in range(S.shape[0]):
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, mcol(s)*U.T)

    w = Zg/numpy.sum(Zg)

    return (w, mu, cov)

def LBGAlgo(x, gmm, alpha,iter,minEigen):
   
    gmm1 =  [(gmm[0][0],gmm[0][1],computeNewCovar(gmm[0][2],minEigen))]
    gmm1 = EMAlgo(x,gmm1)
    for i in range(iter):
        gmm1 = SplitGMM(gmm1,alpha)       
        gmm1 = EMAlgo(x,gmm1)

    return gmm1    


def KFoldValidationFullGMMCovariance(D,L,alpha,minEigen,gmms):
    K = 8 
    # N = int(D.shape[1]/K)


    N = int(D.shape[1]/K)       

    nWrongPrediction = 0
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
        errors = computeClassifications(gmms,Dtotal,LBGAlgo,minEigen,alpha,DTE,LTE)
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




    

