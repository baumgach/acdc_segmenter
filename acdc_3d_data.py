import utils
import image_utils

import os
import glob

# import matplotlib.pyplot as plt

import numpy as np
import cv2
import logging

import model as model
import tensorflow as tf

import model_zoo
import time
import gc

import h5py
from skimage import transform

from models_refine3d import unet_bn as inference_fct

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


FILE_BUFFER = 5

def diagnosis_to_int(diagnosis):
    if diagnosis == 'NOR':  # normal
        return 0
    if diagnosis == 'MINF':  # previous myocardial infarction (EF < 40%)
        return 1
    if diagnosis == 'DCM':  # dialated cardiomyopathy
        return 2
    if diagnosis == 'HCM':  # hypertrophic cardiomyopathy
        return 3
    if diagnosis == 'RV':  # abnormal right ventricle (high volume or low EF)
        return 4
    else:
        raise ValueError('Unknown diagnosis encountered.')

def placeholder_inputs(batch_size, nx, ny):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='labels')
    return images_placeholder, labels_placeholder

def generate_featuremaps(input_folder, output_file):

    # nx = 212
    # ny = 212
    # nz_max = 28
    #
    # target_resolution = (1.36719, 1.36719)

    # nx = 172
    # ny = 172
    # nz_max = 28
    #
    # target_resolution = (1.68, 1.68, 5.0)

    # nx = 148
    # ny = 148
    # nz_max = 28
    #
    # target_resolution = (2.0, 2.0, 5.0)
    #
    # nx = 100
    # ny = 100
    # nz_max = 28
    #
    # target_resolution = (3.0, 3.0, 5.0)

    nx = 116
    ny = 116
    nz_max = 28

    target_resolution = (2.5, 2.5, 5.0)

    hdf5_file = h5py.File(output_file, "w")


    diag_list = {'test': [], 'train': []}
    height_list = {'test': [], 'train': []}
    weight_list = {'test': [], 'train': []}
    patient_id_list =  {'test': [], 'train': []}
    cardiac_phase_list = {'test': [], 'train': []}

    file_list = {'test': [], 'train': []}

    utils.log_and_print('Counting files...')
    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')

            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                file_list[train_test].append(file)

                diag_list[train_test].append(diagnosis_to_int(infos['Group']))
                weight_list[train_test].append(infos['Weight'])
                height_list[train_test].append(infos['Height'])

                patient_id_list[train_test].append(patient_id)

                systole_frame = int(infos['ES'])
                diastole_frame = int(infos['ED'])

                file_base = file.split('.')[0]
                frame = int(file_base.split('frame')[-1])
                if frame == systole_frame:
                    cardiac_phase_list[train_test].append(1)  # 1 == systole
                elif frame == diastole_frame:
                    cardiac_phase_list[train_test].append(2)  # 2 == diastole
                else:
                    cardiac_phase_list[train_test].append(0)  # 0 means other phase


    n_train = len(file_list['train'])
    n_test = len(file_list['test'])

    # Write the small datasets
    for tt in ['test', 'train']:
        hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))

    # Create placeholders for large data
    data = {}
    for tt, num_points in zip(['test', 'train'], [n_test, n_train]):
        data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, (num_points, nx, ny, nz_max), dtype=np.float32)
        data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, (num_points, nx, ny, nz_max), dtype=np.uint8)


    mask_list = {'test': [], 'train': [] }
    img_list = {'test': [], 'train': [] }

    for train_test in ['test', 'train']:

        file_buffer = 0
        counter_from = 0

        for file in file_list[train_test]:

            utils.log_and_print(' ----- Doing image: -------------------------')
            utils.log_and_print(train_test)
            utils.log_and_print(file)
            utils.log_and_print(file_buffer)
            utils.log_and_print(' --------------------------------------------')

            logging.info('Doing: %s' % file)

            file_img = file
            file_base = file.split('.nii.gz')[0]
            file_mask = file_base + '_gt.nii.gz'

            img_dat = utils.load_nii(file_img)
            mask_dat = utils.load_nii(file_mask)

            img = img_dat[0].copy()
            mask = mask_dat[0]

            img = image_utils.normalise_image(img)

            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2], img_dat[2].structarr['pixdim'][3])
            print(pixel_size)

            # if pixel_size[2] == 10:
            #     scale_z = 2.0
            # else:
            #     scale_z = 1.0

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1], pixel_size[2]/ target_resolution[2]]

            img_scaled = transform.rescale(img, scale_vector, order=1, preserve_range=True, multichannel=False)
            mask_scaled = transform.rescale(mask, scale_vector, order=0, preserve_range=True, multichannel=False)

            slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
            mask_vol = np.zeros((nx, ny, nz_max), dtype=np.uint8)

            nz_curr = img_scaled.shape[2]
            stack_from = (nz_max - nz_curr) // 2

            for zz in range(nz_curr):

                slice_rescaled = img_scaled[:,:,zz]
                mask_rescaled = mask_scaled[:,:,zz]

                x, y = slice_rescaled.shape

                x_s = (x - nx) // 2
                y_s = (y - ny) // 2
                x_c = (nx - x) // 2
                y_c = (ny - y) // 2

                if x > nx and y > ny:
                    slice_cropped = slice_rescaled[x_s:x_s + nx, y_s:y_s + ny]
                    mask_cropped = mask_rescaled[x_s:x_s + nx, y_s:y_s + ny]
                else:
                    slice_cropped = np.zeros((nx, ny))
                    mask_cropped = np.zeros((nx, ny))
                    if x <= nx and y > ny:
                        slice_cropped[x_c:x_c + x, :] = slice_rescaled[:, y_s:y_s + ny]
                        mask_cropped[x_c:x_c + x, :] = mask_rescaled[:, y_s:y_s + ny]
                    elif x > nx and y <= ny:
                        slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                        mask_cropped[:, y_c:y_c + y] = mask_rescaled[x_s:x_s + nx, :]

                    else:
                        slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice_rescaled[:, :]
                        mask_cropped[x_c:x_c + x, y_c:y_c + y] = mask_rescaled[:, :]


                slice_vol[:,:,stack_from] = slice_cropped
                mask_vol[:,:,stack_from] = mask_cropped

                stack_from += 1


            img_list[train_test].append(slice_vol)
            mask_list[train_test].append(mask_vol)


            file_buffer += 1

            if file_buffer >= FILE_BUFFER:

                counter_to = counter_from + file_buffer
                utils.log_and_print('Writing data from %d to %d' % ( counter_from, counter_to))

                img_arr = np.asarray(img_list[train_test], dtype=np.float32)
                mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

                data['images_%s' % train_test][counter_from:counter_to, :, :, :] = img_arr
                data['masks_%s' % train_test][counter_from:counter_to, :, :, :] = mask_arr

                counter_from = counter_to

                file_buffer = 0
                img_list[train_test] = []
                mask_list[train_test] = []
                gc.collect()

        # after file loop: Write the remaining data

        utils.log_and_print('Writing remaining data')
        counter_to = counter_from + file_buffer

        img_arr = np.asarray(img_list[train_test], dtype=np.float32)
        mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

        data['images_%s' % train_test][counter_from:counter_to, :, :, :] = img_arr
        data['masks_%s' % train_test][counter_from:counter_to, :, :, :] = mask_arr

        # release mem
        img_list[train_test] = []
        mask_list[train_test] = []
        gc.collect()


    # After test train loop:
    hdf5_file.close()


if __name__ == '__main__':

    base_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/acdc_logdir'

    output_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/data3D_116x116x28.hdf5'

    input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'

    generate_featuremaps(input_path, output_path)

