#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Vincent <vincent.wangworks@gmail.com>
#
# Distributed under terms of the MIT license.

"""
This file define the structure and the value of own data, which prepare for neural network
"""
import numpy as np

def defLabel(images):
    # initial all label of point are zero
    labels = np.repeat(0,len(images))
    # if point inside the circle which radius equal to 1, it's label is 1
    labels[np.square(images).sum(1)<=1] = 1
    # the point which in the circle and in the first and third quadrand have label 2
    labels[np.multiply(images[:,0], images[:,1])>=0] = 2
    # convert to dual format
    dual_labels = np.repeat(0,len(labels)*len(np.unique(labels))).reshape(len(labels),len(np.unique(labels)))
    dual_labels[range(len(labels)),labels] = 1
    return dual_labels
        
class train():
    np.random.seed(1)
    images = np.random.randn(5000,2)
    labels = defLabel(images)
    
class validation():
    np.random.seed(2)
    images = np.random.randn(500,2)
    labels = defLabel(images)           

class test():
    np.random.seed(3)
    images = np.random.randn(800,2)
    labels = defLabel(images)           
