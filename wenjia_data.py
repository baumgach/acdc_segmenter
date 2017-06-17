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

        mask_list = {'test': [], 'train': [] }
        img_list = {'test': [], 'train': [] }
        diag_list = {'test': [], 'train': []}
        height_list = {'test': [], 'train': []}
        weight_list = {'test': [], 'train': []}
        patient_id_list =  {'test': [], 'train': []}
        cardiac_phase_list = {'test': [], 'train': []}

        for folder in os.listdir(self.base_path):

            folder_path = os.path.join(self.base_path, folder)

            if os.path.isdir(folder_path):

                train_test = 'train'

                print('------------------------')
                print(folder)
                print(train_test)

                patient_id = int(folder.lstrip('Atlas')) + 100

                file_names = ['lvsa_ED.nii.gz', 'lvsa_ES.nii.gz']
                segmentation_names = ['segmentation_ED.nii.gz', 'segmentation_ES.nii.gz']

                for file_img, file_mask in zip(file_names, segmentation_names):

                    print(file_img)

                    file_base = file_img.split('.nii.gz')[0]
                    cardiac_phase = file_base.split('_')[-1]

                    img_dat = utils.load_nii(os.path.join(folder_path, file_img))
                    mask_dat = utils.load_nii(os.path.join(folder_path, file_mask))

                    img = img_dat[0]
                    mask = mask_dat[0]

                    img = image_utils.normalise_image(img)

                    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])

                    print('pixel size:')
                    print(pixel_size)

                    for zz in range(img.shape[2]):

                        slice_img = np.squeeze(img[:,:,zz])
                        slice_rescaled = image_utils.rescale_image(slice_img, pixel_size)

                        slice_mask = np.squeeze(mask[:,:,zz])
                        mask_rescaled = image_utils.rescale_image(slice_mask, pixel_size, interp=cv2.INTER_NEAREST)
                        # mask_rescaled = image_utils.rescale_labels_lisa_style(slice_mask, pixel_size, num_labels=5)

                        # cv2.imshow('img', image_utils.convert_to_uint8(slice_rescaled))
                        # cv2.imshow('mask', image_utils.convert_to_uint8(mask_rescaled))
                        # cv2.waitKey(50)
                        #
                        # plt.figure()
                        # plt.imshow(slice_mask)
                        #
                        # plt.figure()
                        # plt.imshow(mask_rescaled_2)
                        #
                        # plt.figure()
                        # plt.imshow(mask_rescaled)
                        #
                        # plt.show()

                        x, y = slice_rescaled.shape

                        # if x > 500:
                        #     print('W: Buggy case downscaling by factor 2')
                        #     slice_rescaled = image_utils.rescale_image(slice_rescaled, (0.5, 0.5))
                        #     mask_rescaled = image_utils.rescale_image(mask_rescaled, (0.5, 0.5), interp=cv2.INTER_NEAREST)
                        #     # mask_rescaled = image_utils.rescale_labels_lisa_style(mask_rescaled, (0.5, 0.5), num_labels=4)
                        #     x, y = slice_rescaled.shape

                        x_s = (x - self.n_x) // 2
                        y_s = (y - self.n_y) // 2
                        x_c = (self.n_x - x) // 2
                        y_c = (self.n_y - y) // 2

                        if x > self.n_x and y > self.n_y:
                            slice_cropped = slice_rescaled[x_s:x_s+self.n_x, y_s:y_s+self.n_y]
                            mask_cropped = mask_rescaled[x_s:x_s+self.n_x, y_s:y_s+self.n_y]
                        else:
                            slice_cropped = np.zeros((self.n_x,self.n_y))
                            mask_cropped = np.zeros((self.n_x,self.n_y))
                            if x <= self.n_x and y > self.n_y:
                                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + self.n_y]
                                mask_cropped[x_c:x_c + x, :] = mask_rescaled[:, y_s:y_s + self.n_y]
                            elif x > self.n_x and y <= self.n_y:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + self.n_x, :]
                                mask_cropped[:, y_c:y_c + y] = mask_rescaled[x_s:x_s + self.n_x, :]

                            else:
                                slice_cropped[x_c:x_c+x, y_c:y_c + x] = slice_rescaled[:, :]
                                mask_cropped[x_c:x_c+x, y_c:y_c + x] = mask_rescaled[:, :]


                        # cv2.imshow('img', image_utils.convert_to_uint8(slice_cropped))
                        # cv2.imshow('mask', image_utils.convert_to_uint8(mask_cropped))
                        # cv2.waitKey(100)

                        # Wenjia labels
                        # 0 - Background
                        # 1 - Left ventricle
                        # 2 - Myocardium
                        # 4 - Right ventricle

                        # ACDC Labels
                        # 0 - Background
                        # 1 - Right ventricle
                        # 2 - Myocardium
                        # 3 - Left ventricle

                        # plt.figure()
                        # plt.imshow(mask_cropped)

                        mask_cropped_corr_lbls = np.where(mask_cropped == 1, [3], mask_cropped)
                        mask_cropped = np.where(mask_cropped_corr_lbls == 4, [1], mask_cropped_corr_lbls)

                        # plt.figure()
                        # plt.imshow(mask_cropped)
                        #
                        # plt.show()

                        if cardiac_phase == 'ES':
                            cardiac_phase_list[train_test].append(1)  # 1 == systole
                        elif cardiac_phase == 'ED':
                            cardiac_phase_list[train_test].append(2)  # 2 == diastole
                        else:
                            cardiac_phase_list[train_test].append(0)  # 0 means other phase

                        img_list[train_test].append(slice_cropped)
                        mask_list[train_test].append(mask_cropped)

                        diag_list[train_test].append(42)
                        weight_list[train_test].append(-1)
                        height_list[train_test].append(-1)

                        patient_id_list[train_test].append(patient_id)

        hdf5_file = h5py.File(self.out_file_name, "w")

        for tt in ['test', 'train']:
            hdf5_file.create_dataset('images_%s' % tt, data=np.asarray(img_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('masks_%s' % tt, data=np.asarray(mask_list[tt], dtype=np.uint8))
            hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
            hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
            hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
            hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))

        hdf5_file.close()



if __name__ == '__main__':

    n_x = 288
    n_y = 288

    base_path = '/scratch_net/bmicdl03/data/CIMAS_Wenjia_Cardiac/'
    # data_maker = DataMakerACDC('data_288x288.hdf5', base_path, n_x, n_y)
    data_maker = DataMakerACDC('wenjiadata_288x288.hdf5', base_path, n_x, n_y)

    data_maker.run()

