

This repository contains code to train state-of-the-art cardiac segmentation networks as described in this
paper: [An Exploration of 2D and 3D Deep Learning
Techniques for Cardiac MR Image Segmentation](https://arxiv.org/abs/1709.04496). The modified 
U-Net architecture achieved the **3rd overall rank** at the MICCAI 2017 [ACDC Cardiac segmentation challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html). 

Authors:
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))
- Lisa M. Koch ([email](mailto:lisa.koch@inf.ethz.ch))

If you find this code helpful in your research please cite the following paper:

```
@article{baumgartner2017exploration,
  title={An Exploration of {2D} and {3D} Deep Learning Techniques for Cardiac {MR} Image Segmentation},
  author={Baumgartner, Christian F and Koch, Lisa M and Pollefeys, Marc and Konukoglu, Ender},
  journal={arXiv preprint arXiv:1709.04496},
  year={2017}
}
```

## Pre-trained weights

Pretrained weights for the best performing method can be found here: https://git.ee.ethz.ch/baumgach/acdc_pretrained_weights

It was not possible to upload all experiments due to size limits. However, the pretrained weights for the other experiments can be requested via email from baumgartner@vision.ee.ethz.ch. 

## Requirements 

- Python 3.5 (tested with Python 3.5.3)
- Tensorflow (tested with tensorflow 1.12)
- The package requirements are given in `requirements.txt`

## Getting the code

Clone the repository by typing

``` git clone https://github.com/baumgach/acdc_segmenter.git ```


## Installing required Python packages

Create an environment with Python 3.5. If you use virutalenv it 
might be necessary to first upgrade pip (``` pip install --upgrade pip ```).

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

The tensorflow packages are not part of the requirements file because you may want to toggle the CPU and GPU version. For the GPU version type

``` pip install tensorflow-gpu==1.12```

If you want to use the CPU version use the following command. 

``` pip install tensorflow==1.12```

If you want to go back and forth between GPU and CPU it probably makes sense to make two separate environments which are identical except
for the tensorflow version. 

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

Then, navigate to [localhost:8008](localhost:8008) in your browser to open tensorboard.

At any point during the training, or after, you can evaluate your model by typing the following:

``` python evaluate acdc_logdir/unet3D_bn_modified_wxent ```

where you have to adapt the line to match your experiment. Note that, the path must be given relative to your
working directory. Giving the full path will not work.


## Running the code on the ETH CVL (Biwi) GPU infrastructure:

Instructions for setting everything up to run this code on the Biwi GPU infrastructure can be found [here](https://git.ee.ethz.ch/baumgach/biwi_tensorflow_setup_instructions).

Don't forget to change the `at_biwi` option in `config/system.py`! 

## Known issues

None at the moment