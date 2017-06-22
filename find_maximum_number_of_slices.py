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


class DataMakerACDC:


    def __init__(self, out_file_name, base_path, n_x, n_y):
        self.base_path = base_path
        self.n_x = n_x
        self.n_y = n_y

        self.out_file_name = out_file_name


    def run(self):

        max_num_slices = 0
        min_num_slices = np.inf

        max_through_plane_res = 0
        min_through_plane_res = np.inf


        for folder in os.listdir(self.base_path):

            folder_path = os.path.join(self.base_path, folder)

            if os.path.isdir(folder_path):

                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                    print(file)

                    file_img = file

                    img_dat = utils.load_nii(file_img)

                    img = img_dat[0]

                    num_slices = img.shape[2]
                    through_plane_res = img_dat[2].structarr['pixdim'][3]

                    if through_plane_res == 10:
                        num_slices *= 2

                    if num_slices > max_num_slices:
                        max_num_slices = num_slices
                    if num_slices < min_num_slices:
                        min_num_slices = num_slices
                    if through_plane_res > max_through_plane_res:
                        max_through_plane_res = through_plane_res
                    if through_plane_res < min_through_plane_res:
                        min_through_plane_res = through_plane_res


        print('Max num slices: %d' % max_num_slices)
        print('Min num slices: %d' % min_num_slices)
        print('Max through-plane res: %f' % max_through_plane_res)
        print('Min through-plane res: %f' % min_through_plane_res)



if __name__ == '__main__':

    n_x = 288
    n_y = 288

    base_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    # data_maker = DataMakerACDC('data_288x288.hdf5', base_path, n_x, n_y)
    # data_maker = DataMakerACDC('newdata_288x288.hdf5', base_path, n_x, n_y)
    data_maker = DataMakerACDC('bla', base_path, n_x, n_y)

    data_maker.run()

