#Neural Network in Image Recognition

This program is used for experiments of dissertation.
Please see the detail on the 4th section of Report.

## Easy use
The Auto.sh provides a bash file to install the packages and run all the experiments automatically.
Please make sure that you have been permitted to write in system.
For department server, openSUSE 13.2 (Harlequin) (x86_64). Run by
$ ./Auto.sh
The results of experimetns run by Auto.sh is saved on the Result/, each experiment result will be saved on a single folder. There is an auto.sh file in each single folder used to plot the result automatically. Run by
$ cd Result/experiment_Name
$ ./auto.sh

## The installation of program for OSX and UNIX like system:

1. In case the permission problem, we suggest to use Anaconda. Installed following https://www.continuum.io/downloads 
$ wget https://repo.continuum.io/archive/Anaconda2-4.1.1-MacOSX-x86_64.sh
$ bash Anaconda2-4.1.1-MacOSX-x86_64.sh
$ conda create -n tensorflow python=2.7
source activate tensorflow

2. Install pip tools, following https://pip.pypa.io/en/stable/installing/
$ wget https://bootstrap.pypa.io/get-pip.py
$ python get-pip.py

3. Check and install packages we need
$ pip install --upgrade argparse
$ pip install --upgrade pickle
$ pip install --upgrade numpy
$ pip install --upgrade matplotlib
$ export TF_BINARY_URL= #the source is depended on platform from https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation
$ pip install --upgrade $TF_BINARY_URL

## The usage of program is given below:

1. To get help list by
$ python ANN.py -h

2. To run the program by
$ python ANN.py result_address      %The program would restore the result date at result_address

3. To add more parameters by
$ python ANN.py result_address --ParameterName 'ParameterValues'
e.g. 
$ python ANN.py data.pkl --learning_rate '[0.1, 0.05, 0.02]'

4. Show plot by
$ python show.py result_address -p 'ParameterName'
e.g.
$ python show.py data.pkl -p 'learning_rate'

