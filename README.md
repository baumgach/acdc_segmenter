
This repository contains code to train state-of-the-art cardiac segmentation networks as described in this
conference submission: [An Exploration of 2D and 3D Deep Learning
Techniques for Cardiac MR Image Segmentation](http://www.vision.ee.ethz.ch/~baumgach/papers/ACDC_challenge_paper.pdf).

Authors:
- Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))
- Lisa M. Koch ([email](mailto:lisa.koch@inf.ethz.ch))


## Requirements 

- Python 3.4 (only tested with 3.4.3)
- Tensorflow >= 1.0 (only tested with 1.1.0)
- The remainder of the requirements are given in `requirements.txt`


## System setup

If you are running this code at the ETH Computer Vision Lab (CVL), make sure you have set up your system
as is required.

Specifically, setup your python using `pyenv` as described [here](https://computing.ee.ethz.ch/Programming/Languages/Python). Since the code requires Python 3.4 (ideally 3.4.3) make sure you choose that version in the `pyenv install`, and `pyenv global` steps. If you have previously installed another Python version using `pyenv` simply repeat the last three steps for version 3.4.3. 

Setup your CUDA and `tensorflow` as described [here](https://docs.google.com/document/d/1UXhXkqn20v_jC3CzSvdgvgED2iuXIKsgcW16GtVYBu8/edit#heading=h.p5485wgdj33x). The code was tested with CUDA 8.0.44, and CuDNN 5.1.10. *Do not* pip install tensorflow yet. This will be done after setting up the virtualenv and installing the `requirements.txt`. 

The code was tested using `tensorflow-1.1.0`. There were a lot of breaks with the release tensorflow 1.x, so the code is unlikely to 
work with earlier versions. However, it might work with version 1.2 (untested).

To run the code on the Biwi GPU clusters you will also need to setup the environment for that. All infos can be found [here](https://wiki.vision.ee.ethz.ch/itet/gridengine?s[]=gpu). If you are a not a permanent member of the CVL (Biwi) group, you need to email [Alex Locher](mailto:alocher@vision.ee.ethz.ch) so he can create a grid engine account for you. 

You will also need to include some settings into your `.bashrc`. Simply paste the following lines into your shell

```
echo "# settings for BIWI cluster" >> ~/.bashrc
echo "source /home/sgeadmin/BIWICELL/common/settings.sh" >> ~/.bashrc
```

## Getting the code

Simply clone the repository by typing

``` git clone git@git.ee.ethz.ch:baumgach/acdc_segmenter_internal.git ```

If this is your first time using the D-ITET gitlab server, you will need to [setup an SSH key](https://git.ee.ethz.ch/help/ssh/README.md) first.  

## Setting up virtualenv

For convenience it is often best to set up seperate virtualenv's for usage on GPU and usage on CPU. This code requires Python 3.x
and was only tested with Python 3.4.3. It is not backwards compatible with Python 2.x. In order to set up the virtual 
environment appropriately, run the following command:

``` virtualenv -p python3.4 name-of-your-env ```

After setting up and activating the virtualenv, it might be necessary to first upgrade pip.

``` pip install --upgrade pip ```

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

Lastly, install tensorflow:

``` pip install tensorflow==1.1.0 ```
or
``` pip install tensorflow-gpu==1.1.0 ```

depending if you are setting up your GPU environment or CPU environment

WARNING: Installing tensorflow before the requirements.txt will lead to weird errors while compiling `scikit-image` in `pip install -r requirements`. Make sure you install tensorflow *after* the requirements. 

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

``` tensorboard --logdir=acdc_logdir --port 6006 ```

Then, navigate to [127.0.0.1:6006](http://127.0.0.1:6006) in your browser to open tensorboard.

At any point during the training, or after, you can evaluate your model by typing the following:

``` python evaluate acdc_logdir/unet3D_bn_modified_wxent ```

where you have to adapt the line to match your experiment. Note that, the path must be given relative to your
working directory. Giving the full path will not work.


## Running the code on the GPU clusters

The code can be sent to the Biwi GPU Clusters (gridengine) by using the scripts `train_on_host.sh` and `evaluate_on_host.sh`. Before
using those functions make sure you adjust all the paths in both files to match your local system.

A job can be submitted as follows:

``` qsub train_on_host.sh ```

or

``` qsub evaluate_on_host.sh acdc_logdir/unet3D_bn_modified_wxent ```

The progress of the training can still be monitored in tensorboard exactly in the same way, as when running the code
locally.

If you need to see the console output as well, for example, for debugging you can find it in the gridengine log file which
will be written to the folder specified in the `train_on_host.sh` and `evaluate_on_host.sh` files.

First check for the most recently changed log files using

``` ls -ltr /scratch_net/your-machine/path-to-your-logfolder ```

Then you can watch the output of the file by using `tail`:

``` tail -f /scratch_net/your-machine/path-to-your-logfolder/logfile ```

Other useful commands for dealing with the GPU cluster are:
- `qstat`: Show your current jobs. `w` means waiting, `r` means running.
- `qdel -j JOBID`: Delete job with JOBID
- `qdel -u USERNAME`: Delete all jobs belonging to USERNAME

The full documentation can be found [here](https://wiki.vision.ee.ethz.ch/itet/gridengine).

WARNING: Do not write the log files to your home directory because it will be slow and slow down the network for others.
Instead write the files on your local hard drive via `/scratch_net/computer-name`.


## Known issues

- If `pip install -r requirements.txt` fails while compiling `scikit-image`, try the following:
    - Make sure you install `tensorflow` _after_ the `requirements.txt`
    - If that didn't solve the issue, try running `pip install --upgrade cython` seperately, and then run `pip install -r requirements.txt` again. 