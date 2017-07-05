#!/bin/bash
#
# Script to send job to BIWI clusters using qsub.
# Usage: qsub run_on_host.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments
# and the code to the local equivalents of your machines.
# Author: Christian F. Baumgartner (c.f.baumgartner@gmail.com)
#
## SGE Variables:
#
## otherwise the default shell would be used
#$ -S /bin/bash
#
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue
#$ -l h_rt=06:00:00

## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=40G

# Host and gpu settings
#$ -l gpu
##$ -l hostname=biwirender11 #bmicgpu01   # Comment in or out to force a specific machine

## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferably on your scratch
#$ -o /scratch_net/bmicdl03/logs/acdc/
#
## send mail on job's end and abort
#$ -m a

## multi-thread
##$ -pe multicore 2
## LOCAL PATHS

CUDA_HOME=/scratch_net/bmicdl03/libs/cuda-8.0-bmic
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# for pyenv
export PATH="/home/baumgach/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# not sure if the next two lines are required
source /home/sgeadmin/BIWICELL/common/settings.sh
export CUDA_VISIBLE_DEVICES=$SGE_GPU

source /scratch_net/bmicdl03/code/python/environments/tensorflow-gpu/bin/activate

## EXECUTION OF PYTHON CODE:
python /scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/train3d.py

echo "Hostname was: `hostname`"
echo "Reached end of job file."


