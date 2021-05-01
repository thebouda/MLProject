# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:57:27 2021

@author: simo6
"""

import numpy

def mcol(v):
    return v.reshape((v.size,1))

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

int_to_label_quality = {
    0:'low_quality',
    1:'high_quality'
    }

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


if __name__ == '__main__':
    D,L = load ('./Data/Train.txt')
  