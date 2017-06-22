import nibabel as nib
import numpy as np
import cv2
import os
import glob
import logging
from skimage import measure

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

def get_latest_model_checkpoint_path(folder, name):

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


def post_process_prediction_3D(img):
    """
    Hook for some possible image postprocessing.

    - keep only largest connected component for each structure

    :param img:
    :return: processed image
    """

    out_img = np.zeros(img.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = img == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img
