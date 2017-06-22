import nibabel as nib
import numpy as np
import cv2
import os
import glob
import logging

def makefolder(folder):
    """
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    """
    if not os.path.exists(folder):
        #print " - Creating output directory..."
        os.makedirs(folder)
        return True
    return False

def load_nii(img_path):

    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)

def get_best_model_checkpoint_path(folder, name):

    iteration_nums = []
    for file in glob.glob(os.path.join(folder, '%s*.meta' % name)):

        file_base, postfix_and_number, rest = file.split('.')[0:3]
        it_num = int(postfix_and_number.split('-')[-1])

        iteration_nums.append(it_num)

    latest_iteration = np.max(iteration_nums)

    return os.path.join(folder, name + '-' + str(latest_iteration))

def log_and_print(text):
    print(text)
    logging.info(text)


