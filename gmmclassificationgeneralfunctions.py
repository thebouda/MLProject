import numpy
import scipy.optimize
# from GMM_load import load_gmm
import matplotlib.pyplot as plt
# import sklearn.datasets as skl

def computeClassifications(gmmTry,Dtotal,Algorithm,minEigen,alpha,DTE,LTE):
    
    errorRateVec = []
    for i in range(gmmTry):
        marginalLike = []

        for j in range(len(Dtotal)): # for each class
            mu_1 = numpy.mean(Dtotal[j],axis=1).reshape((Dtotal[j].shape[0], 1))

            C_1 = numpy.cov(Dtotal[j]).reshape((Dtotal[j].shape[0], Dtotal[j].shape[0]))
            C_1= computeNewCovar(C_1,minEigen)
            gmm_1 = [(1,mu_1,C_1)]
            

            gmmLBG = Algorithm(Dtotal[j],gmm_1,alpha,i,minEigen)
            marginalLike.append(logpdf_GMM(DTE,gmmLBG)[0])
        stackedLike = numpy.vstack((marginalLike[0],marginalLike[1]))


        predictions = numpy.argmax(stackedLike,axis = 0)
        correctpredictions= numpy.array(predictions == LTE).sum()
        errorRate = 100 - correctpredictions *100 /LTE.size
        errorRateTup = (i,errorRate)
        errorRateVec.append(errorRateTup)
        print("error Rate for "+ str((i)*(i)) +"gmms \n")
        print(str(errorRate) + "\n")
    return errorRateVec

# function that computes one model of gmm, for a given gmm and data
def computeOneModelGmmClassification(gmms, Dtotal,Algorithm,minEigen,alpha,DTE,LTE):

    i = gmms
    marginalLike = []

    for j in range(len(Dtotal)): # for each class
        mu_1 = numpy.mean(Dtotal[j],axis=1).reshape((Dtotal[j].shape[0], 1))

        C_1 = numpy.cov(Dtotal[j]).reshape((Dtotal[j].shape[0], Dtotal[j].shape[0]))
        C_1= computeNewCovar(C_1,minEigen)
        gmm_1 = [(1,mu_1,C_1)]
        

        gmmLBG = Algorithm(Dtotal[j],gmm_1,alpha,i,minEigen)
        marginalLike.append(logpdf_GMM(DTE,gmmLBG)[0])
    stackedLike = numpy.vstack((marginalLike[0],marginalLike[1]))


    predictions = numpy.argmax(stackedLike,axis = 0)
    correctpredictions= numpy.array(predictions == LTE).sum()
    errorRate = 100 - correctpredictions *100 /LTE.size
    print("error Rate ")
    print(str(errorRate) + "\n")

    return errorRate




def gmmValues(mu,sigma,w,gmm):
    newGMM = []
    for g in range(len(gmm)):
        MUG = numpy.reshape(mu[:,g],(mu.shape[0],1)) # instead of (4,) we have 4,1 no errors in the future
        newGMM.append((w[g],MUG,sigma[g]))
    return newGMM

def mcol(v):
    return v.reshape((v.size, 1))


def computeNewCovar(cov,minPSI):
    psi = minPSI
    initialCovTrans,s1, _  =numpy.linalg.svd(cov)
    s1[s1<psi] = psi
    covNewInitial = numpy.dot(initialCovTrans, mcol(s1)*initialCovTrans.T)
    return covNewInitial

    
def SplitGMM(gmm,alpha):
    newGMM = []
    
    for i in range(len(gmm)):
        Sigma_g = gmm[i][2]
        U, s, Vh = numpy.linalg.svd(Sigma_g) # sigma being the variance everytime
        d = U[:, 0:1] * s[0]**0.5 * alpha # apha is a factor that we use
        # calculate new mean and new weight
        mu1 = gmm[i][1] - d 
        mu2 = gmm[i][1] + d
        newGMM.append((gmm[i][0]*0.5,mu1,gmm[i][2]))
        newGMM.append((gmm[i][0]*0.5,mu2,gmm[i][2]))

    return newGMM

def logpdf_GMM(X,gmm):
   
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        S[i,:] = logpdf_GAU_ND(X,gmm[i][1], gmm[i][2])
        S[i,:] += numpy.log(gmm[i][0])

    logdens = scipy.special.logsumexp(S,axis = 0)

    return (logdens.T,S)

def logpdf_GAU_ND(x, mu, sigma):
    return numpy.diag(-(x.shape[0]/2)*numpy.log(2*numpy.pi)-(1/2)*(numpy.linalg.slogdet(sigma)[1])-(1/2)*numpy.dot(numpy.dot((x-mu).T,numpy.linalg.inv(sigma)),(x-mu)))


def GAU_logpdf(x, mu, var):
    # computes log density and returns 1-dim array
    return (-0.5*numpy.log(2*numpy.pi))-0.5*numpy.log(var)-(((x-mu)**2)/(2*var))


def Estep(logdens, S):
    return numpy.exp(S-logdens.reshape(1, logdens.size))
