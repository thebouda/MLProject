import numpy
import matplotlib.pyplot as plt

from gmmclassificationgeneralfunctions import gmmValues, computeClassifications,mcol,computeNewCovar,SplitGMM,logpdf_GMM,Estep

# ----  diagonal covariances
def DiagoSigma(gmm):
    gmmDiag = []
    for i in range (len(gmm)):
        Sigma_g = gmm[i][2]
        Sigma_g = Sigma_g * numpy.eye(Sigma_g.shape[0])
        gmmDiag.append((gmm[i][0],gmm[i][1],Sigma_g))
    return gmmDiag


def LBDiagVariance(x,gmm,alpha,iterations,minEigen):
    psi = minEigen
    gmmDiag = [(gmm[0][0],gmm[0][1],computeNewCovar(gmm[0][2],psi))]
    gmmDiag = EMDiagAlgorithm(x,gmmDiag)
    for i in range(iterations):
        gmmDiag = SplitGMM(gmmDiag,alpha)
        gmmDiag = EMDiagAlgorithm(x,gmmDiag)
    return gmmDiag



def DiagMstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
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
    # print(Sg)
    mu = Fg / Zg
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    for g in range(S.shape[0]):
        cov[g] = cov[g] * numpy.eye(cov[g].shape[0])
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/numpy.sum(Zg)
    # print(w)
    return (w, mu, cov)

def EMDiagAlgorithm(x,gmm):
    diffLLR =10
    threshold = 1e-6
    flag = True
    while flag :
        (logdens4, s4) = logpdf_GMM(x,gmm)
      
        oldLLR = numpy.sum(logdens4)/len(logdens4)

        #  E step
        responsabilities = Estep(logdens4, s4)

        # m step
        
        (w, mu, cov) = DiagMstep(x, s4, responsabilities)


        newGMM = gmmValues(mu,cov,w,gmm)

        (newlogdens4, s4) = logpdf_GMM(x,newGMM)
        newLLR = numpy.sum(newlogdens4)/len(newlogdens4)

        diffLLR = newLLR-oldLLR
        gmm =newGMM
        delta = abs(diffLLR) <  threshold
        if delta:
            flag = False
          
            
    finalGMM = newGMM
  
    return finalGMM



def KFoldValidationDiagGMMCovariance(D,L,alpha,minEigen,gmms):
    K = 8 


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
   
        
        Dtotal = [DTR0,DTR1]
        errors = computeClassifications(gmms,Dtotal,LBDiagVariance,minEigen,alpha,DTE,LTE)
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

