#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Vincent <Vincent@Vincents-MacBook-Air.local>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def calDerivative(seq):
    """TODO: to calculate the 1st derivative of sequence
    :seq : the sequence need to calculate the 1st derivative
    :returns: the result 
    """
    tmp  = seq[1:]
    tmp.append(seq[-1])
    derivative = np.divide(np.subtract(tmp, seq), seq)
    derivative[-1] = derivative[-2] # delete the last one, which have wrong data
    derivative = moving_average(derivative)
    return derivative

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#pkl_file = open('learning_rate.pkl','rb')
pkl_file = open('experiment_6_2_learning_rate.pkl','rb')
data = list()
node1 = 0
node2 = 50
#while True:
for item in range(5):
    try:
        data = pickle.load(pkl_file)
        avg_loss = data['avg_loss'][node1:node2]
        val_acc = data['validation_accuracy'][node1:node2]
        # plot the avg_loss
        p1 = plt.subplot(221)
        p1.set_title('average loss '+str(node1)+'-'+str(node2)+' in different learning rate')
        p1.plot(range(len(avg_loss)), avg_loss, label=str(data['learning_rate']))
        # plot the 1st derivative of avg_loss
        p2 = plt.subplot(222)
        derivative = calDerivative(avg_loss)
        p2.set_title('1st derivative of average loss '+str(node1)+'-'+str(node2)+' in different learning rate')
        p2.plot(range(len(derivative)), derivative, label=str(data['learning_rate']))
        # plot the validation accuracy
        p3 = plt.subplot(223)
        p3.set_title('validation accuracy '+str(node1)+'-'+str(node2)+' in different learning rate')
        p3.plot(range(len(val_acc)), val_acc, label=str(data['learning_rate']))
        # plot the 1st derivative of validation accuracy
        p4 = plt.subplot(224)
        derivative = calDerivative(val_acc)
        p4.set_title('1st derivative of validation accuracy '+str(node1)+'-'+str(node2)+' in different learning rate')
        p4.plot(range(len(derivative)), derivative, label=str(data['learning_rate']))
    except:
        #Exception("EOF")
        break

p1.legend()
plt.show()

