import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib


def labels(i):
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
    return int_to_label[i]

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
def mcol(v):
    return v.reshape((v.size,1))


if __name__ == '__main__':

    DTrain,LTrain =load('Data/Train.txt')
    DTest,LTest =load('Data/Test.txt')
    figure,axes  =  plt.subplots(nrows=4, ncols=3)

    j=0
    q=0
    # represent the 11 histograms for each variable
    for i in range(DTrain.shape[0]): # 11 values, 12 subplots 3 *4
        if (j >2):
            j=0
            q+=1
        if (q >3):
            q=0
        axes[q,j].hist(DTrain[i,:],bins = 10, density = True,label =labels(i))
        axes[q,j].legend(str(i))
        j +=1
    plt.savefig('../histograms.pdf')    
    plt.show()


    # calculate the correlation and plot it
    # using the coeficient of pearson
    corr2=numpy.round(numpy.corrcoef(DTrain),3)
    print(corr2)
    plt.imshow(corr2, cmap='RdBu', vmin=-1, vmax=1)
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

