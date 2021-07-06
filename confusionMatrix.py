import numpy


def colvec(vec):
  return vec.reshape(numpy.size(vec,0),1)


# function that return the llr
def naivebayesLLR(DTrain,LTrain,DTest,LTest,prior0):
        # calculate variances
    (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)

    #diagonalize variances
    var0Diag = numpy.multiply(numpy.identity(DTrain.shape[0]),var0)
    var1Diag =numpy.multiply(numpy.identity(DTrain.shape[0]),var1)

    # prior0 = calculatePriors(DTrain,LTrain,0)
    prior0 =0.5
    prior1 =1- prior0

    likeVar1=logpdf_GAU_ND(DTest,mu0,var0Diag)*prior0
    likeVar2=logpdf_GAU_ND(DTest,mu1,var1Diag)*prior1

    LikeVar =likeVar2-likeVar1
    llrNaive =LikeVar
    # SMatrix=numpy.vstack((likeVar1,likeVar2)) #S matrix
    # SJoint=SMatrix.sum(axis= 0) # because we already mutiplied by the prior

    return llrNaive

# computes the log likelihood ratio 
def mvgLLR(DTrain,LTrain,DTest,LTest,prior0):

  # compute var and mu for each class 
  (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)

  # priors of classes 0 and 1
  # prior0 = calculatePriors(DTrain,LTrain,0)
  #prior0 =0.5
  prior1 =1- prior0
    
  likeVar0=logpdf_GAU_ND(DTest,mu0,var0)*prior0 # by assigning priors it does not ameliorate
  likeVar1=logpdf_GAU_ND(DTest,mu1,var1)*prior1
  
  llrMVG=likeVar1-likeVar0

  return llrMVG

# compuute the llr for the tied cov
def tiedCovLLR(DTrain,LTrain,DTest,LTest,prior0):
  (mu0,var0),(mu1,var1)= computeVarMed(DTrain,LTrain)
  defVar= secMethodTiedCov(var0,var1,DTrain,LTrain)

  # prior0 = calculatePriors(DTrain,LTrain,0)
 # prior0 =0.5
  prior1 =1- prior0

  likeVar1=logpdf_GAU_ND(DTest,mu0,defVar)*prior1
  likeVar2=logpdf_GAU_ND(DTest,mu1,defVar) *prior0

  tiedLLR = likeVar2-likeVar1
  
  return tiedLLR


def secMethodTiedCov(var1,var2,D,L):
    tiedVar=0
    newVars= [var1,var2]
    for cClass in range(numpy.size(list(set(L)))): #with set we get the distinct values, we need it for the number of classes
        tiedVar= tiedVar + newVars[cClass]*numpy.size(D[:,L==cClass],1) # Nc*Variance for that class
    tiedVar = tiedVar / numpy.size(D,1)
    return tiedVar

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
    
def DCF(bayesMatrix,prior1,cfn,cfp): # optimal bayes matrix and pi1,cost false negative, cost false positive
  fnr =bayesMatrix[0,1]/(bayesMatrix[0,1]+bayesMatrix[1,1]) #false negative rate
  fpr =bayesMatrix[1,0]/(bayesMatrix[1,0]+bayesMatrix[0,0]) # false positive rate

  dcf = prior1 * cfn *fnr +(1-prior1)*cfp*fpr
  bDummy= min(prior1*cfn,(1-prior1)*cfp)
  dcfNorma =dcf/bDummy
  return dcf,dcfNorma

def calculatePredictionsBinary (L,Predic):
  # predicttion where 0 was predicted 0 
  class00Pred = getIfSameValue(numpy.where(L==0)[0],numpy.where(Predic==0)[0])
  # prediction where 0 was predicted 1 and 2
  class01Pred = getIfSameValue(numpy.where(L==0)[0],numpy.where(Predic==1)[0])

  # predicttion where 1 was predicted 2 
  class10Pred = getIfSameValue(numpy.where(L==1)[0],numpy.where(Predic==0)[0])
  class11Pred = getIfSameValue(numpy.where(L==1)[0],numpy.where(Predic==1)[0])
 
  return class00Pred,class01Pred,class10Pred,class11Pred

# returns the number of times a L = predictions
def getIfSameValue(L,predictions):
  sumed=0
  for elem in L:
    if elem in predictions:
      sumed+=1
  return sumed

def computeOptimalBayes(posterior1,CFN,CFP,llr,labels):
  # array of predictions
  predictions =[]
  for likeli in llr:
    thresholdLog = -numpy.log(posterior1*CFN/((1-posterior1)*CFP))
    comparison = likeli - thresholdLog
    if comparison > 0 :
      predic=1
    else:
      predic=0

    predictions.append(predic)
  npPredictions = numpy.array(predictions)
  # we get the predictions
  class00Pred,class01Pred,class10Pred,class11Pred=calculatePredictionsBinary(labels,npPredictions)
  return numpy.array([[class00Pred,class10Pred],
  [class01Pred,class11Pred]])


def KFoldValidationConfusionMatrix(D,L):
    K = 8 
    # N = int(D.shape[1]/K)


    N = int(D.shape[1]/K)       

    nWrongPrediction = 0
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
        confusionMatrix(DTR,LTR, DTE, LTE,i)

      






def confusionMatrix(DTrain,LTrain,DTest,LTest,i):
   
    # what have we consider as positive ? positive good wine negative bad wine
    # we need c mu and 
    #fileNameConfMatrix = "confMatrixResults"+ str(i) + ".txt" 
    
    priorNB = 0.5 

    # get the llr of the naive bayes model
    nbLLR=naivebayesLLR(DTrain,LTrain,DTest,LTest,priorNB)
    multiLLR =mvgLLR(DTrain,LTrain,DTest,LTest,priorNB)
    tiedLLR =tiedCovLLR(DTrain,LTrain,DTest,LTest,priorNB)


    #fileConfMatrix= open(fileNameConfMatrix,'w')
      
    # cfnRange = range()
    CFN =1
    CFP =2 # 2 scelta


  
    print ('\n')
    print("--------partition"+str(i)+"-------")
    
    # optimal bayes
    # naive bayes
    obmNB =  computeOptimalBayes(priorNB,CFN,CFP,nbLLR,LTest)
    print('naive bayes')
    print(obmNB )
    print('\n')

    # fileConfMatrix.writelines('naive bayes \t')
    # fileConfMatrix.writelines(obmNB)

    # mvg
    obmMVG =  computeOptimalBayes(priorNB,CFN,CFP,multiLLR,LTest)
    print('mvg')
    print(obmMVG )
    print('\n')

    # mvg
    obmTied =  computeOptimalBayes(priorNB,CFN,CFP,tiedLLR,LTest)
    print('tied cov')
    print(obmTied)



    #------- evaluation-------
    dcfNB,dcfNormaNB = DCF(obmNB,priorNB,CFN,CFP)
    print('naive bayes')
    print(dcfNB)
    print(dcfNormaNB )
    print('\n')

    #mvg
    dcfNB,dcfNormaNB = DCF(obmMVG,priorNB,CFN,CFP)
    print('mvg')
    print(dcfNB)
    print(dcfNormaNB)
    print('\n')

    #tied cov
    dcfTied,dcfNormaTied = DCF(obmTied,priorNB,CFN,CFP)
    print('Tied Cov')
    print(dcfTied)
    print(dcfNormaTied)
    print('\n')





