
import numpy
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size,1))

# returns the label
def labels(label):
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
    return int_to_label[label]

# returns the quality label
def labelsQuality(quality):
    int_to_label_quality = {
        0:'low_quality',
        1:'high_quality'
        }
    return 

# loads the data and returns Data and labels 
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

# function that computes the PCA, and draws the scatter plot
def PCAfunct(D,L):
    mu = D.mean(1) #media
    DC = D - mu.reshape((mu.size, 1)) #centrare la matrice
    DCT = DC.T #trasposta
    Ctmp=(1/DC.shape[1])*DC
    C=numpy.dot(Ctmp,DCT) #calcolo la covarianza
    s, U = numpy.linalg.eigh(C) #autovalori e autovettori
    P = U[:, ::-1][:, 0:2]
    DP = numpy.dot(P.T, D)
    plot_scatter(DP, L)

# function that plots the scatter
def plot_scatter(D,L):  
    for i in range(2):
        to_plot = D[:, L==i]
        plotted = plt.scatter(to_plot[0,:], to_plot[1,:]) # Plot Magic
        plotted.set_label(labelsQuality(i))
    plt.legend()
    plt.show()





    