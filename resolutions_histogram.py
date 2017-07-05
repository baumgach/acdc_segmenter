import utils
import image_utils

import os
import glob

from skimage.transform import rescale #(image, scale, order=1, mode=None, cval=0, clip=True, preserve_range=False, multichannel=None)
from skimage.io import imshow

import matplotlib.pyplot as plt

import numpy as np
import cv2

import h5py



def run(base_path):

    pixel_size_list = []
    for folder in os.listdir(base_path):

        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):


            print('------------------------')
            print(folder)

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = int(folder.lstrip('patient'))


            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                print(file)

                file_img = file

                img_dat = utils.load_nii(file_img)

                pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2], img_dat[2].structarr['pixdim'][3])
                print('pixel size:')
                print(pixel_size)

                pixel_size_list.append(pixel_size)

    return pixel_size_list




if __name__ == '__main__':

    base_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    pixel_size_list = run(base_path)

    pixel_size_array = np.asarray(pixel_size_list)

    print(pixel_size_array.shape)

    print('IN PLANE')
    in_plane = pixel_size_array[:,0]
    unique = np.unique(in_plane)
    print(unique)
    print(np.histogram(in_plane, bins=len(unique)))

    plt.hist(in_plane, bins=len(unique))
    plt.show()

    print('THROUGH PLANE')
    through_plane = pixel_size_array[:,2]
    unique = np.unique(through_plane)
    print(unique)
    print(np.histogram(through_plane, bins=len(unique)))

    plt.hist(through_plane, bins=len(unique))
    plt.show()




