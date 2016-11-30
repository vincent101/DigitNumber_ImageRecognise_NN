#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Vincent <Vincent@Vincents-MacBook-Air.local>
#
# Distributed under terms of the MIT license.

"""
A simple Neural Network to recognise images
"""
from __future__ import division, print_function, absolute_import  
import os
import argparse
import time
import pickle

import tensorflow as tf
import numpy as np

# Global Default Value
DISPLAY_STEP = 100
SEED = 1
NOISE_PROB = 0
SIZE_HIDDEN_LAYERS = (256,)
ACTIVATION_FUNCTION = ("tf.nn.relu",)
PRED_ACTIVATION_FUNCTION = "ArgMax"
OPTIMISER = "GradientDescent" # "Momentum"
LOSS_FUNCTION = "CrossEntropy" # or "MSE"
LEARNING_RATE = 0.05
TRAINING_EPOCHS = 1001
BATCH_SIZE = 256
KEEP_PROB = 1         # for dropout, if keep probability equal 1 means no dropout

# import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("DataSet/MNIST_data/", one_hot=True)
trX, trY, vaX, vaY, teX, teY = mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, mnist.test.images, mnist.test.labels

# import own dataset
#import DataSet.generateData
#trX, trY, vaX, vaY, teX, teY = generateData.train.images, generateData.train.labels, generateData.validation.images, generateData.validation.labels, generateData.test.images, generateData.test.labels

_, size_X = trX.shape
_, size_Y = trY.shape

class objects():
    """Docstring for parameter. """
    # the paramters normally dont change
    display_step = DISPLAY_STEP 
    # the paramters used to be test
    seed = 0
    size_hidden_layers = 0
    activation_function = ''
    pred_activation_function = ''
    optimiser = ''
    loss_function = ''
    learning_rate = 0
    training_epochs = 0
    batch_size = 0
    keep_prob = 0
    # restore the result
    avg_loss = []
    validation_accuracy = []
    time = 0.

    def __init__(self, seed=SEED, size_hidden_layers=SIZE_HIDDEN_LAYERS, activation_function=ACTIVATION_FUNCTION, 
            pred_activation_function=PRED_ACTIVATION_FUNCTION, optimiser=OPTIMISER, loss_function = LOSS_FUNCTION,
            learning_rate=LEARNING_RATE, training_epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, keep_prob=KEEP_PROB):
        self.seed = seed
        self.size_hidden_layers = size_hidden_layers
        self.activation_function = activation_function
        self.pred_activation_function = pred_activation_function
        self.optimiser = optimiser
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.keep_prob = keep_prob

    def speaker(self):
        print('Training Parameters:', '\n',
                'seed: ', self.seed, '\n',
                'size_hidden_layers:', self.size_hidden_layers, '\n',
                'activation_function:', self.activation_function, '\n',
                'pred_activation_function:', self.pred_activation_function, '\n',
                'optimiser:', self.optimiser, '\n', 
                'loss function:', self.loss_function, '\n',
                'learning_rate:', '%.5f\n'%self.learning_rate, 
                'batch_size: ', '%d\n'%self.batch_size,
                'training_epochs:', '%d\n'%self.training_epochs,
                'keep_prob: ', '%.2f'%self.keep_prob)

def main():
    """TODO: Import MNIST data and parameters for train

    """

    # add parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('bar', type=argparse.FileType('wb'), default='data/data.pkl',
            help="The file which restore the data.")
    # parameters control hidden layers
    parser.add_argument('--seed', default=SEED,
            help="e.g. --seed '[1,2]' or '1' ")
    parser.add_argument('--size_hidden_layers', default=SIZE_HIDDEN_LAYERS,
            help="e.g. --size_hidden_layer '[(256,),(128,)]' or '(256,)' ")
    parser.add_argument('--activation_function', default=ACTIVATION_FUNCTION,
            help="e.g. --activation_function '[(\"tf.nn.relu\",),(\"tf.sigmoid\",)]' or '(\"tf.nn.relu\",)'")
    parser.add_argument('--pred_activation_function', default=PRED_ACTIVATION_FUNCTION,
            help="e.g. --pred_activation_function '[\"ArgMax\",None]' or \"ArgMax\" ")
    parser.add_argument('--optimiser', default=OPTIMISER,
            help="e.g. --optimiser '[\"GradientDescent\", \"Momentum\"]' or \"GradientDescent\" ")
    parser.add_argument('--loss_function', default=LOSS_FUNCTION,
            help="e.g. --loss_function '[\"CrossEntropy\", \"MSE\"]' or \"CrossEntropy\" ")
    # parameters control learning
    parser.add_argument('--learning_rate', default=LEARNING_RATE,
            help="e.g. --learning_rate '[0.1, 0.05]' or '0.1' ")
    parser.add_argument('--training_epochs', default=TRAINING_EPOCHS,
            help="e.g. --training_epoch '[3, 10]' or '3' ")
    parser.add_argument('--batch_size', default=BATCH_SIZE,
            help="e.g. --batch_size '[128, 256]' or '128' , batch_size=0 means no batch.")
    parser.add_argument('--keep_prob', default=KEEP_PROB,
            help="e.g. --keep_prob '[0.5, 0.8]' or '0.5' , keep_prob=0 means no dropout.")
    # noise the training labels
    parser.add_argument('--noise_prob', default=NOISE_PROB, type=float,
            help="e.g. --noise_prob '0.05' , noise_prob=0 means without noise in training set.")
    
    # check if there are some parameter need to be test
    args = parser.parse_args()
    # open output_file
    output_file = open(str(args.bar.name), 'wb')
    #output_file = open('data/data.pkl', 'wb')

    # call train
    noise_labels(args.noise_prob)
    flag = 0
    for i in args.__dict__:
        # try to remove the ' ' of each value
        try:
            args.__dict__[i] = eval(args.__dict__[i])
        except:
            continue
        # check if the values in the list
        if type(args.__dict__[i]) == list:
            # if there are, use new parameter gourp in several times
            flag = 1
            for j in args.__dict__[i]:
                arg = args
                arg.__dict__[i] = j
                # try to remove the ' ' of each value
                for k in arg.__dict__:
                    try:
                        arg.__dict__[k] = eval(arg.__dict__[k])
                    except:
                        continue
                obj = objects(seed=arg.seed, size_hidden_layers=arg.size_hidden_layers, activation_function=arg.activation_function,
                        pred_activation_function=arg.pred_activation_function, optimiser=arg.optimiser, loss_function = arg.loss_function,
                        learning_rate=arg.learning_rate, training_epochs=arg.training_epochs, batch_size=arg.batch_size, keep_prob=arg.keep_prob)
                obj.speaker()
                train(obj)
                restore_res(obj, output_file)

    # if no parameter need to be test
    if flag == 0:
        arg = args
        obj = objects(seed=arg.seed, size_hidden_layers=arg.size_hidden_layers, activation_function=arg.activation_function,
                pred_activation_function=arg.pred_activation_function, optimiser=arg.optimiser, loss_function=arg.loss_function,
                learning_rate=arg.learning_rate, training_epochs=arg.training_epochs, batch_size=arg.batch_size)
        obj.speaker()
        train(obj)
        restore_res(obj, output_file)

    # close output_file
    output_file.close()

    return 0

# build the function used for train model, give loss and plot
def train(obj):
    """TODO: Docstring for train.
    :obj: parameters

    """
    # hold the input
    X = tf.placeholder(tf.float32, [None, size_X])
    Y = tf.placeholder(tf.float32, [None, size_Y])
    keep_prob = tf.placeholder("float32")

    # check the type of parameters
    obj.learning_rate = float(obj.learning_rate)
    obj.training_epochs = int(obj.training_epochs)
    obj.batch_size = int(obj.batch_size)
    # convert the type of some paremeter
    if type(obj.size_hidden_layers) == str:
        obj.size_hidden_layers = eval(obj.size_hidden_layers)
    if type(obj.activation_function) == str:
        obj.activation_function = eval(obj.activation_function)
    if len(obj.size_hidden_layers) == len(obj.activation_function):
        num_hidden_layers = len(obj.size_hidden_layers)
    else:
        num_hidden_layers = min(len(obj.activation_function),len(obj.size_hidden_layers))
    
    # initial weights and bias, set seed = SEED
    # the weight and bias of 1 hidden layer
    weights = {
        'hl_0' : tf.Variable(tf.random_normal([size_X, obj.size_hidden_layers[0]], stddev=0.01, seed=obj.seed)),
        'pred' : tf.Variable(tf.random_normal([obj.size_hidden_layers[num_hidden_layers-1], size_Y], stddev=0.01, seed=obj.seed)),
            }
    bias = {
        'hl_0' : tf.Variable(tf.zeros([1, obj.size_hidden_layers[0]])+0.1),
        'pred' : tf.Variable(tf.zeros([1, size_Y])+0.1),
            }
    
    # add the 1st hidden layer
    exec("hl_0 = add_layer(X, weights['hl_0'], bias['hl_0'], activation_function="+obj.activation_function[0]+", keep_prob=keep_prob)")
    # add other hidden layers with its weight and bias
    #hl_1 = add_layer(hl_0, weights['hl_1'], bias['hl_1'], activation_function=obj.activation_function[1])
    for item in range(1,num_hidden_layers):
        tmp_1, tmp = 'hl_'+str(item-1), 'hl_'+str(item)
        exec("weights['"+tmp+"']=tf.Variable(tf.random_normal([obj.size_hidden_layers[item-1], obj.size_hidden_layers[item]], stddev=0.01, seed=obj.seed))")
        exec("bias['"+tmp+"']=tf.Variable(tf.zeros([1, obj.size_hidden_layers[item]])+0.1)")
        exec(tmp+"= add_layer("+tmp_1+", weights['"+tmp+"'], bias['"+tmp+"'], activation_function="+obj.activation_function[item]+", keep_prob=keep_prob)")
    # add the prediction layer
    #pred = add_layer(hl_1, weights['pred'], bias['pred'], activation_function = None)
    exec("pred = add_layer(hl_"+str(num_hidden_layers-1)+", weights['pred'], bias['pred'], activation_function=None)")
    
    # loss function
    if obj.loss_function == "CrossEntropy": 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y)) # classfication
    elif obj.loss_function == "MSE":
        loss = tf.reduce_mean(tf.square(tf.sub(pred, Y))) # MSE
    else:
        print("Wrong Input in Loss Function Name.\n")
    # train with optimiser to minimise the loss
    if obj.optimiser == "GradientDescent":
        train_op = tf.train.GradientDescentOptimizer(obj.learning_rate).minimize(loss)
    elif obj.optimiser == "Momentum":
        train_op = tf.train.MomentumOptimizer(obj.learning_rate, 0.5).minimize(loss)
    else:
        print("Wrong Input in Optimiser Name.\n")
    # classify the pred
    if obj.pred_activation_function == "ArgMax":
        predict_op = tf.argmax(pred, 1)
    
    # initialize the Variable
    init = tf.initialize_all_variables()
    # start to train
    print("Training Start:")
    with tf.Session() as sess:
        all_start = time.time()
        sess.run(init)
        # training cycle
        avg_loss = [None]*obj.training_epochs
        validation_acc = [None]*obj.training_epochs
        if obj.batch_size == 0:
            obj.batch_size = len(trX)
        num_batch = int(len(trX)/obj.batch_size)
        for epoch in range(obj.training_epochs):
            avg_loss[epoch] = 0.
            # compute by batch
            for i in range(num_batch):
                # run train op and loss op to get loss value
                start, end = obj.batch_size*i, obj.batch_size*(i+1)
                _, l = sess.run([train_op, loss], feed_dict={X: trX[start:end], Y: trY[start:end], keep_prob: obj.keep_prob})
                # compute average loss in each epoch
                avg_loss[epoch] += l / num_batch
            validation_acc[epoch] = np.mean(np.argmax(vaY, axis=1) == sess.run(predict_op, feed_dict={X:vaX, keep_prob:1.}))
            # report
            if epoch % obj.display_step == 0:
                # report validation acc
                print(" Epoch:", "%04d, " % (epoch+1), 
                        "Loss:", "{:.9f}, ".format(avg_loss[epoch]),
                        "Validation Accuracy:", "{:.9f}".format(validation_acc[epoch]))
        test_acc = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X:teX, keep_prob:1.}))
        all_end = time.time()
        time_cost = all_end - all_start
        print("Optimization Finished! ",
                "Test accuracy:","{:.9f} ".format(test_acc),
                "Time cost: {:.3f}\n".format(time_cost))
    # retore result into object
    obj.avg_loss = avg_loss
    obj.validation_accuracy = validation_acc
    obj.time = time_cost


# build the layer add function
def add_layer(x, W, b, activation_function=None, keep_prob=False):
    """TODO: function to add a layer 
    :x: input data
    :activation_function: the activation function using in added layer, default is None means linear function
    :returns: output data

    """
    Wx_plus_b = tf.add(tf.matmul(x, W), b)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    if keep_prob != False:
        outputs = tf.nn.dropout(outputs, keep_prob)
    return outputs

def restore_res(obj, output_file):
    """TODO: Docstring for restore_res.

    :obj: the object
    :returns: 0

    """
    pickle.dump(obj.__dict__, output_file)
    return 0

def noise_labels(noise_prob):
    """TODO: Docstring for noise.

    :noise_prob: TODO
    :returns: TODO

    """
    if noise_prob != 0:
        print('Labels have been noised.')
        trY[np.int32(np.random.sample(int(len(trY)*noise_prob))*len(trY))] = trY[np.int32(np.random.sample(int(len(trY)*noise_prob))*len(trY))]
    return 0

if __name__=="__main__":
    main()

