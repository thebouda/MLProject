import numpy
from numpy.core.fromnumeric import reshape
import scipy.optimize




def mCol(v):
    return v.reshape((v.size, 1))

# primal that computes the primal objective
def primal (z,w,x,c):
    firstEl=numpy.square(numpy.linalg.norm(w)) *0.5 
    #first way of calculating
    # secondEl= z[:,None].T.dot((w[:,None].T.dot(x)).T)
    secondEl=z*w[:,None].T.dot(x) #1*66
    ones =1-secondEl
    zeros = numpy.zeros(shape=ones.shape)
    sec= numpy.sum(c*numpy.maximum(ones,zeros))
    J=firstEl+sec
    return J


# class in order to compute the primal 

class svmPrimal():

    def __init__(self,k,x,l):
        auxVec =numpy.ones(shape=(numpy.size(l),))
        self.k= auxVec* k # vector of ones multiplied by k, value that we choose
        self.d_Hat=numpy.vstack((x,self.k)) # 
        self.l=l # labels 
    

    def computation(self,alfa):

        # we get the values of 0 to be labeled as -1
        z= 2*self.l -1 
        z = mCol(z)
        # first way of computing H
        G_hat = numpy.dot(self.d_Hat.T,self.d_Hat)

        H_hat= z * z.T * G_hat
       
        # none turn into (1,66) instead of (,66)
        theone = numpy.ones(shape=(self.d_Hat.shape[1],))

        
        jAlfa= 0.5*alfa[:,None].T.dot(H_hat.dot(alfa[:,None])) - alfa[:,None].T.dot(theone[:,None])
        
        # grad = H_hat.dot(alfa[:,None]) - theone[:,None]

        # return jAlfa, grad.flatten()
        return jAlfa

    def grad(self,alfa):
        z= 2*self.l -1 
        z = mCol(z)
        # first way of computing H
        G_hat = numpy.dot(self.d_Hat.T,self.d_Hat)

        H_hat= z * z.T * G_hat
        theone = numpy.ones(shape=(self.d_Hat.shape[1],))

        grad = H_hat.dot(alfa[:,None]) - theone[:,None]

        return grad.flatten()


# function that computes the svm solution, prints it and saves it in a file
def svm(DTR,LTR,DTE,LTE,c):
    k=1

    svmJ=svmPrimal(k,DTR,LTR)
    
    v=numpy.zeros(DTR.shape[1])    
    boundsL= [(0,c) for i in range(DTR.shape[1])]
        
    x,fx,d= scipy.optimize.fmin_l_bfgs_b(func = svmJ.computation,fprime= svmJ.grad,x0=v,approx_grad=False,bounds = boundsL, factr=1000)

    # x is alfa
    # fx i j(alfa)
    z= 2*LTR -1 

    
    extendTest=numpy.ones(shape=(numpy.size(DTE[1,:])))*k
    xExtendedTest=numpy.vstack((DTE,extendTest))

    extendTrain=numpy.ones(shape=(numpy.size(DTR[1,:]))) *k
    xExtendedTrain=numpy.vstack((DTR,extendTrain))

    w_hat_star=numpy.sum(mCol(x) * mCol(z) * xExtendedTrain.T,axis =0)
    # w_star = w_hat_star[0:-1] 
    # b_star = w_hat_star[-1] 


    primalObj= primal(z,w_hat_star,xExtendedTrain,c)
    dualObj= fx
    dualityGap = primalObj +dualObj
    
    classPred= w_hat_star[:,None].T.dot(xExtendedTest)
    acc= ((classPred > 0) == LTE).mean()
    errorRate = (1-acc)*100

    print(" c " + str(c))
    print("errorRate: %f" %(errorRate))
    print("duality gap %f" %(dualityGap))
    print("primal : "+str(primalObj))
    print("dualObj: "+ str(dualObj))
    print("\n")

    return errorRate,dualityGap,primalObj,c



# function that computes the kfold validation for the svm
def KFoldValidationSVM(D,L,C):
    K = 8 
    # N = int(D.shape[1]/K)
    resultsSVM = open('Resultsfile1.txt','w')
    resultsSVM.writelines('K \t set \t c \t error Rate \t duality gap \t primal \n')

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
        c=C[i]
        errorRate,dualityGap,primalW,c = svm(DTR,LTR,DTE,LTE,c)
        resultsSVM.writelines(str(K)+" \t "+str(i)+ " \t" + str(c) +"\t"+ str(numpy.around(errorRate,3)) +" \t "+str(numpy.around(dualityGap[0,0],4)) +" \t "+str(numpy.around(primalW,5))+ "\n")
    resultsSVM.close()