import nibabel as nib
import numpy as np
import cv2
import os

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

