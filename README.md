

This repository contains code to train state-of-the-art cardiac segmentation networks as described in this
paper: [An Exploration of 2D and 3D Deep Learning
Techniques for Cardiac MR Image Segmentation](https://arxiv.org/abs/1709.04496). The modified 
U-Net architecture achieved the **3rd overall rank** at the MICCAI 2017 [ACDC Cardiac segmentation challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html). 

Authors:
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))
- Lisa M. Koch ([email](mailto:lisa.koch@inf.ethz.ch))


## Requirements 

- Python 3.4 (only tested with 3.4.3)
- Tensorflow >= 1.0 (tested with 1.1.0, and 1.2.0)
- The remainder of the requirements are given in `requirements.txt`


## Getting the code

Clone the repository by typing

``` git clone https://github.com/baumgach/acdc_segmenter.git ```


## Installing required Python packages

Create an environment with Python 3.4. If you use virutalenv it 
might be necessary to first upgrade pip (``` pip install --upgrade pip ```).

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

Then, install tensorflow:

``` pip install tensorflow==1.2 ```
or
``` pip install tensorflow-gpu==1.2 ```

depending if you are setting up your GPU environment or CPU environment. The code was also
tested with tensorflow 1.1 if for some reason you prefer that version. Tensorflow 1.3 is currently causing
trouble on our local machines, so we couldn't test this version yet. 

WARNING: Installing tensorflow before the requirements.txt will lead to weird errors while compiling `scikit-image` in `pip install -r requirements`. Make sure you install tensorflow *after* the requirements. 
If you run into errors anyways try running `pip install --upgrade cython` and then reruning `pip install -r requirements.txt`. 


## Download the ACDC challenge data

If you don't have access to the data already you can sign up and download it from this [webpage](http://acdc.creatis.insa-lyon.fr/#challenges).

The cardiac segmentation challenge and the data is described in detail [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html).


## Running the code locally

Open the `config/system.py` and edit all the paths there to match your system.

Next, open `train.py` and, at the top of the file, select the experiment you want to run (or simply use the default).

To train a model simpy run:

``` python train.py ```

WARNING: When you run the code on CPU, you need around 12 GB of RAM. Make sure your system is up to the task. If not you can try reducing the batch size, or simplifying the network. 

In `system.py`, a log directory was defined. By default it is called `acdc_logdir`. You can start a tensorboard
session in order to monitor the training of the network(s) by typing the following in a shell with your virtualenv
activated

``` tensorboard --logdir=acdc_logdir --port 8008 ```

Then, navigate to [127.0.0.1:8008](http://127.0.0.1:8008) in your browser to open tensorboard.

At any point during the training, or after, you can evaluate your model by typing the following:

``` python evaluate acdc_logdir/unet3D_bn_modified_wxent ```

where you have to adapt the line to match your experiment. Note that, the path must be given relative to your
working directory. Giving the full path will not work.


## Known issues

- If `pip install -r requirements.txt` fails while compiling `scikit-image`, try the following:
    - Make sure you install `tensorflow` _after_ the `requirements.txt`
    - If that didn't solve the issue, try running `pip install --upgrade cython` seperately, and then run `pip install -r requirements.txt` again.
     
- There seems to be an issue compiling scikit-image when using Python 3.5. If this is happening make sure you are using Python 3.4. 

