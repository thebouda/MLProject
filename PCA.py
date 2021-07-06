# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:04:35 2021

"""

import numpy
import matplotlib.pyplot as plt

int_to_label_quality = {
    0:'low_quality',
    1:'high_quality'
    }

def plot_scatter(D,L):  
    
    for i in range(2):
        to_plot = D[:, L==i]
        plotted = plt.scatter(to_plot[0,:], to_plot[1,:]) # Plot Magic
        plotted.set_label(int_to_label_quality[i])
    plt.legend()
    plt.show()
    
def PCAfunct(D,L,DT):
    mu = D.mean(1) #media
    DC = D - mu.reshape((mu.size, 1)) #centrare la matrice
    DCT = DC.T #trasposta
    Ctmp=(1/DC.shape[1])*DC
    C=numpy.dot(Ctmp,DCT) #calcolo la covarianza
    S, U = numpy.linalg.eigh(C) #autovalori e autovettori
    explained_variance_ratio_ = S / numpy.sum(S)
    reverse_explained_variance = explained_variance_ratio_[::-1]
    P = U[:, ::-1][:, 0:5]
    DP = numpy.dot(P.T, D)
    DTE = numpy.dot(P.T,DT)
    plot_scatter(DP, L)
    
    #drow cumulative expleined variance
    plt.plot(numpy.cumsum(reverse_explained_variance))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('cumulative_explained_variance')
    return DP,DTE 