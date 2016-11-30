#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Vincent <Vincent@Vincents-MacBook-Air.local>
#
# Distributed under terms of the MIT license.

"""

"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bar", type=argparse.FileType('rb'), default=None, nargs='+',
            help="The .pkl data file you want to plot. You could include several file once, the program will give the average")
    parser.add_argument("-p","--parameter", type=str, default=None,
            help="The parameter name you test, e.g -p 'learning_rate' ")
    args = parser.parse_args()
    #print(args.__dict__)

    # open file and load data
    tmp_data = list()
    for item in range(len(args.bar)):
        tmp_data.append(withdraw(str(args.bar[item].name))) 
    # average the tmp_data
    data = avgData(tmp_data)
    arg = args.parameter

    # plot data seperately
    #for item in tmp_data:
        #for i in item:
            #plot_origin(i,arg)

    # plot the data averagely
    for item in data:
        plot_origin(item, arg)
    plt.show()

    # plot derivative of data
    max_loss_d = 0
    min_vadAcc_d = 0
    for item in data:
        tmp_max = np.abs(np.max(calDerivative(item['avg_loss'])))
        tmp_min = np.min(calDerivative(item['validation_accuracy']))
        if tmp_max > max_loss_d:
            max_loss_d = tmp_max
        if tmp_min < min_vadAcc_d:
            min_vadAcc_d = tmp_min
    for item in data:
        plot_derivative(item, arg, max_loss_d, min_vadAcc_d)
    plt.show()

def withdraw(filename):
    """
    filename : must be a string
    return   : the list of data
    """
    # open file
    pkl_file = open(filename, 'rb')
    data = list()
    while True:
        try:
            data.append(pickle.load(pkl_file))
        except:
            #Exception("EOF")
            break
    return data

def plot_origin(data, arg=None):
    '''TODO: plot the avg_loss, validation_accuracy vs epoch
    data     : should be dict
    arg      : the name of paramenter which show in plot
    '''

    # hold the variable
    avg_loss = data['avg_loss']
    validation_accuracy = data['validation_accuracy']
    #avg_loss_derivative = calDerivative(avg_loss)
    #validation_accuracy_derivative = calDerivative(validation_accuracy)

    node1 = len(avg_loss)
    # plot the average loss
    p1 = plt.subplot(221)
    p1.set_title('Loss VS epochs in different '+str(arg))
    p1.set_ylabel('Loss')
    p1.plot(range(len(avg_loss[:node1])), avg_loss[:node1], label=str(arg)+':'+str(data[str(arg)]))
    p1.grid(True)
    p1.legend()

    #p2 = plt.subplot(222)
    #p2.set_title('Loss VS 100-end epoch in different '+str(arg))
    #p2.set_ylabel('Loss')
    #p2.plot(range(node1, len(avg_loss)), avg_loss[node1:])
    #p2.grid(True)

    #log the average loss and plot
    p2 = plt.subplot(222)
    p2.set_title('Log scale of Loss')
    p2.set_ylabel('Log(Loss)')
    p2.plot(range(len(avg_loss)), np.log(avg_loss))
    p2.grid(True)
    
    node1 = len(validation_accuracy)
    # plot the validation accuracy
    p3 = plt.subplot(223)
    p3.set_title('Validation Accuracy VS epochs in different '+str(arg))
    p3.set_xlabel('Num_Epoch')
    p3.set_ylabel('ValidationAccuracy')
    p3.plot(range(len(validation_accuracy[:node1])), validation_accuracy[:node1])
    p3.grid(True)

    #p4 = plt.subplot(224)
    #p4.set_title('Validation Accuracy VS 100-end epoch in different '+str(arg))
    #p4.set_xlabel('Num_Epoch')
    #p4.set_ylabel('ValidationAccuracy')
    #p4.plot(range(node1,len(validation_accuracy)), validation_accuracy[node1:])
    #p4.grid(True)

    # calculate -log(1-validationAccuracy) and plot
    p4 = plt.subplot(224)
    p4.set_title('Log scale of ValidAccuracy')
    p4.set_xlabel('Num_Epoch')
    p4.set_ylabel('-Log(1-ValidationAccuracy)')
    p4.plot(range(len(validation_accuracy)), -np.log(np.subtract(1, validation_accuracy)))
    p4.grid(True)

def plot_derivative(data, arg=None, max_loss_d=0, min_vadAcc_d=0):
    '''TODO: plot the derivative of avg_loss and validation_accuracy
    data     : should be dict
    arg      : the name of paramenter which show in plot
    '''

    # hold the variable
    avg_loss = data['avg_loss']
    validation_accuracy = data['validation_accuracy']
    avg_loss_derivative = calDerivative(avg_loss)
    validation_accuracy_derivative = calDerivative(validation_accuracy)

    node1 = len(avg_loss_derivative)
    # plot the 1 derivative of average loss
    p1 = plt.subplot(221)
    p1.set_title('1st derivative of Loss VS epochs in different '+str(arg))
    p1.set_ylabel('1st derivative of Loss')
    p1.plot(range(len(avg_loss_derivative[:node1])), avg_loss_derivative[:node1])
    p1.grid(True)

    #p2 = plt.subplot(222)
    #p2.set_title('1st derivative of Loss VS 100-end epoch in different '+str(arg))
    #p2.set_ylabel('1st derivative of Loss')
    #p2.plot(range(node1, len(avg_loss_derivative)), avg_loss_derivative[node1:])
    #p2.grid(True)
    
    #log scale of the 1st derivative of average loss and plot
    p2 = plt.subplot(222)
    p2.set_title('Log scale of 1st derivative of Loss')
    p2.set_ylabel('-Log(-dLoss)')
    p2.plot(range(len(avg_loss_derivative)), -np.log(np.subtract(2*max_loss_d, avg_loss_derivative)))
    #p2.plot(range(len(avg_loss_derivative)), -np.log(np.subtract(0.001,avg_loss_derivative)))
    p2.grid(True)

    node1 = len(validation_accuracy_derivative)
    # plot the validation accuracy
    p3 = plt.subplot(223)
    p3.set_title('1st derivative of Validation Accuracy VS epochs in different '+str(arg))
    p3.set_xlabel('Num_Epoch')
    p3.set_ylabel('1st derivative of ValidationAccuracy')
    p3.plot(range(len(validation_accuracy_derivative[:node1])), validation_accuracy_derivative[:node1], label=str(arg)+':'+str(data[str(arg)]))
    p3.grid(True)
    p3.legend()

    #p4 = plt.subplot(224)
    #p4.set_title('1st derivative of Validation Accuracy VS 100-end epoch in different '+str(arg))
    #p4.set_xlabel('Num_Epoch')
    #p4.set_ylabel('1st derivative of ValidationAccuracy')
    #p4.plot(range(node1,len(validation_accuracy_derivative)), validation_accuracy_derivative[node1:])
    #p4.grid(True)

    # log scale of the 1st derivative of validation accuracy
    p4 = plt.subplot(224)
    p4.set_title('Log scale of 1st derivative of ValidAccuracy')
    p4.set_xlabel('Num_Epoch')
    p4.set_ylabel('Log(dValidationAccuracy)')
    p4.plot(range(len(validation_accuracy_derivative[:node1])), np.log(np.subtract(validation_accuracy_derivative[:node1], 2*min_vadAcc_d)))
    #p4.plot(range(len(validation_accuracy_derivative)), np.log(np.subtract(validation_accuracy_derivative,-0.001)))
    p4.grid(True)

def calDerivative(seq):
    """TODO: to calculate the 1st derivative of sequence
    :seq : the sequence need to calculate the 1st derivative
    :returns: the result 
    """
    tmp  = seq[1:]
    tmp = np.append(tmp, seq[-1])
    derivative = np.divide(np.subtract(tmp, seq), seq)
    derivative[-1] = derivative[-2] # delete the last one, which have wrong data
    #derivative = moving_average(derivative,49)
    return derivative

def moving_average(a, n=3):
    """TODO: using moving average to smooth the sequence
    :a : the sequence
    :n : the step of moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def avgData(tmp_data):
    """TODO: to average the avg_loss, validation_accuracy and time of several file
    :tmp_data : the list of data file
    :return : average all file's data into one data file
    """
    data = tmp_data[0]
    if len(tmp_data)!=1:
        # loop the tmp_data
        for i in range(1, len(tmp_data)):
            # loop the tmp_data[i]
            for j in range(len(tmp_data[i])):
                data[j]['avg_loss'] = np.add(data[j]['avg_loss'], tmp_data[i][j]['avg_loss'])
                data[j]['validation_accuracy'] = np.add(data[j]['validation_accuracy'], tmp_data[i][j]['validation_accuracy'])
                data[j]['time'] = data[j]['time'] + tmp_data[i][j]['time']
        # loop the data, calculate average
        for item in data:
            item['avg_loss'] /= len(tmp_data)
            item['validation_accuracy'] /= len(tmp_data)
            item['time'] /= len(tmp_data)
    return data

if __name__=="__main__":
    main()

