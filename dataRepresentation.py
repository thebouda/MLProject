import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib

from functions import labels, load

if __name__ == '__main__':

    DTrain,LTrain =load('Data/Train.txt')
    DTest,LTest =load('Data/Test.txt')
    # figure,axes  =  plt.subplots(nrows=4, ncols=3)

    # j=0
    # q=0
    # for i in range(DTrain.shape[0]): # 11 values, 12 subplots 3 *4
    #     if (j >2):
    #         j=0
    #         q+=1
    #     if (q >3):
    #         q=0
    #     axes[q,j].hist(DTrain[i,:],bins = 10, density = True,label =labels(i))
    #     axes[q,j].legend(str(i))
    #     j +=1
    # plt.savefig('../histograms.pdf')    
    # plt.show()

    # calculate the correlation betwween variables
    # using the coeficient of pearson
    corr=numpy.zeros((11,11))
    varsToCompare =[]
    for i in range(DTrain.shape[0]): # 11 values, 12 subplots 3 *4
        for j in range(DTrain.shape[0]):
            varsToCompare = [DTrain[j,:],DTrain[i,:]]
            corr[i,j] = numpy.cov(varsToCompare)[0,0]/(numpy.sqrt(numpy.var(DTrain[i,:])) *numpy.sqrt( numpy.var(DTrain[i,:])))
    
    corr2= numpy.corrcoef(DTrain)
    print(corr)
    plt.imshow(corr2,cmap='hot')
    plt.show()


    #for individual printing
    
    # for i in range(DTrain.shape[0]): # 11 values, 12 subplots 3 *4
    #     plt.figure()
    #     plt.hist(DTrain[i,:],bins = 10, density = True, alpha = 0.4,label =labels(i)) 
    #     plt.legend(labels(i))
    #     plt.savefig('../hist_%s.pdf' % labels(i))
    #     # Show range of values 
    #     print('%s' % labels(i))
    #     print('Range of values: max: {%.4f}, min: {%.4f}' % (float (numpy.max(DTrain[i,:])),float (numpy.min(DTrain[i,:]))))
    # plt.show()

