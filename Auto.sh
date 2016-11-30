#! /bin/sh
#
# main.sh
# Copyright (C) 2016 Vincent <vincent.wangworks@gmail.com>
#
# Distributed under terms of the MIT license.
#

# Installation for openSUSE 13.2 (Harlequin) (x86_64)
# install and use conda
wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh
conda create -n tensorflow python=2.7
source activate tensorflow
# use pip
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install --upgrade argparse
pip install --upgrade pickle
pip install --upgrade numpy
pip install --upgrade matplotlib
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL

# code for experiment 1
python ANN.py Result/experiment_1_num_hidden_layers_displayed_by_size_hidden_layers/d1.pkl --size_hidden_layers '[(128,),(128,128,)]' --seed '1'
python ANN.py Result/experiment_1_num_hidden_layers_displayed_by_size_hidden_layers/d2.pkl --size_hidden_layers '[(128,),(128,128,)]' --seed '2'
python ANN.py Result/experiment_1_num_hidden_layers_displayed_by_size_hidden_layers/d3.pkl --size_hidden_layers '[(128,),(128,128,)]' --seed '3'

# code for experiment 2
python ANN.py Result/experiment_2_size_hidden_layers/d1.pkl --size_hidden_layers '[(128,),(512,)]' --seed '1'
python ANN.py Result/experiment_2_size_hidden_layers/d2.pkl --size_hidden_layers '[(128,),(512,)]' --seed '2'
python ANN.py Result/experiment_2_size_hidden_layers/d3.pkl --size_hidden_layers '[(128,),(512,)]' --seed '3'

# code for experiment 3
python ANN.py Result/experiment_3_activation_function/d1.pkl --activation_function '[("tf.nn.relu",),("tf.sigmoid",)]' --seed '1'
python ANN.py Result/experiment_3_activation_function/d2.pkl --activation_function '[("tf.nn.relu",),("tf.sigmoid",)]' --seed '2'
python ANN.py Result/experiment_3_activation_function/d3.pkl --activation_function '[("tf.nn.relu",),("tf.sigmoid",)]' --seed '3'

# code for experiment 4
python ANN.py Result/experiment_4_optimiser/	d1.pkl --optimiser '["GradientDescent", "Momentum"]' --seed '1'
python ANN.py Result/experiment_4_optimiser/	d2.pkl --optimiser '["GradientDescent", "Momentum"]' --seed '2'
python ANN.py Result/experiment_4_optimiser/	d3.pkl --optimiser '["GradientDescent", "Momentum"]' --seed '3'

# code for experiment 5
python ANN.py Result/experiment_5_learning_rate/d1.pkl --learning_rate '[0.1, 0.05]' --seed '1'
python ANN.py Result/experiment_5_learning_rate/d2.pkl --learning_rate '[0.1, 0.05]' --seed '2'
python ANN.py Result/experiment_5_learning_rate/d3.pkl --learning_rate '[0.1, 0.05]' --seed '3'

# code for experiment 6
python ANN.py Result/experiment_6_dropout/d1.pkl --keep_prob '[0, 0.5]' --seed '1'
python ANN.py Result/experiment_6_dropout/d2.pkl --keep_prob '[0, 0.5]' --seed '2'
python ANN.py Result/experiment_6_dropout/d3.pkl --keep_prob '[0, 0.5]' --seed '3'

# code for experiment 7
python ANN.py Result/experiment_7_loss_function/d1.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '1'
python ANN.py Result/experiment_7_loss_function/d2.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '2'
python ANN.py Result/experiment_7_loss_function/d3.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '3'
python ANN.py Result/experiment_7_loss_function/d1_noise.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '1' --noise_prob '0.05'
python ANN.py Result/experiment_7_loss_function/d2_noise.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '2' --noise_prob '0.05'
python ANN.py Result/experiment_7_loss_function/d3_noise.pkl --loss_function '["CrossEntropy", "MSE"]' --seed '3' --noise_prob '0.05'


