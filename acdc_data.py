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


    def _diagnosis_to_int(self, diagnosis):
        if diagnosis == 'NOR':  # normal
            return 0
        if diagnosis == 'MINF': # previous myocardial infarction (EF < 40%)
            return 1
        if diagnosis == 'DCM':  # dialated cardiomyopathy
            return 2
        if diagnosis == 'HCM':  # hypertrophic cardiomyopathy
            return 3
        if diagnosis == 'RV':   # abnormal right ventricle (high volume or low EF)
            return 4
        else:
            raise ValueError('Unknown diagnosis encountered.')

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

                train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

                print('------------------------')
                print(folder)
                print(train_test)

                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = int(folder.lstrip('patient'))


                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                    print(file)

                    file_img = file
                    file_base = file.split('.nii.gz')[0]
                    file_mask = file_base + '_gt.nii.gz'

                    frame = int(file_base.split('frame')[-1])

                    img_dat = utils.load_nii(file_img)
                    mask_dat = utils.load_nii(file_mask)

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
                        # mask_rescaled = image_utils.rescale_labels_lisa_style(slice_mask, pixel_size, num_labels=4)

                        cv2.imshow('img', image_utils.convert_to_uint8(slice_rescaled))
                        cv2.imshow('mask', image_utils.convert_to_uint8(mask_rescaled))
                        cv2.waitKey(400)

                        x, y = slice_rescaled.shape

                        if x > 500:
                            print('W: Buggy case downscaling by factor 2')
                            slice_rescaled = image_utils.rescale_image(slice_rescaled, (0.5, 0.5))
                            mask_rescaled = image_utils.rescale_image(mask_rescaled, (0.5, 0.5), interp=cv2.INTER_NEAREST)
                            # mask_rescaled = image_utils.rescale_labels_lisa_style(mask_rescaled, (0.5, 0.5), num_labels=4)
                            x, y = slice_rescaled.shape

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
                                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]
                                mask_cropped[x_c:x_c+x, y_c:y_c + y] = mask_rescaled[:, :]

                        systole_frame = int(infos['ES'])
                        diastole_frame = int(infos['ED'])

                        if frame == systole_frame:
                            cardiac_phase_list[train_test].append(1)  # 1 == systole
                        elif frame == diastole_frame:
                            cardiac_phase_list[train_test].append(2)  # 2 == diastole
                        else:
                            cardiac_phase_list[train_test].append(0)  # 0 means other phase

                        img_list[train_test].append(slice_cropped)
                        mask_list[train_test].append(mask_cropped)

                        diag_list[train_test].append(self._diagnosis_to_int(infos['Group']))
                        weight_list[train_test].append(infos['Weight'])
                        height_list[train_test].append(infos['Height'])

                        patient_id_list[train_test].append(patient_id)

        # hdf5_file = h5py.File(self.out_file_name, "w")

        # for tt in ['test', 'train']:
        #     hdf5_file.create_dataset('images_%s' % tt, data=np.asarray(img_list[tt], dtype=np.float32))
        #     hdf5_file.create_dataset('masks_%s' % tt, data=np.asarray(mask_list[tt], dtype=np.uint8))
        #     hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        #     hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        #     hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
        #     hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        #     hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))
        #
        # hdf5_file.close()



if __name__ == '__main__':

    n_x = 288
    n_y = 288

    base_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    # data_maker = DataMakerACDC('data_288x288.hdf5', base_path, n_x, n_y)
    # data_maker = DataMakerACDC('newdata_288x288.hdf5', base_path, n_x, n_y)
    data_maker = DataMakerACDC('debug_288x288.hdf5', base_path, n_x, n_y)

    data_maker.run()

