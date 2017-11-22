# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform

import utils
import image_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dialated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(input_folder, output_file, mode, size, target_resolution, split_test_train=True):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    hdf5_file = h5py.File(output_file, "w")

    diag_list = {'test': [], 'train': []}
    height_list = {'test': [], 'train': []}
    weight_list = {'test': [], 'train': []}
    patient_id_list = {'test': [], 'train': []}
    cardiac_phase_list = {'test': [], 'train': []}

    file_list = {'test': [], 'train': []}
    num_slices = {'test': 0, 'train': 0}

    logging.info('Counting files and parsing meta data...')

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            if split_test_train:
                train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'
            else:
                train_test = 'train'

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')

            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                file_list[train_test].append(file)

                # diag_list[train_test].append(diagnosis_to_int(infos['Group']))
                diag_list[train_test].append(diagnosis_dict[infos['Group']])
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

                nifty_img = nib.load(file)
                num_slices[train_test] += nifty_img.shape[2]

    # Write the small datasets
    for tt in ['test', 'train']:
        hdf5_file.create_dataset('diagnosis_%s' % tt, data=np.asarray(diag_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('weight_%s' % tt, data=np.asarray(weight_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('height_%s' % tt, data=np.asarray(height_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patient_id_%s' % tt, data=np.asarray(patient_id_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('cardiac_phase_%s' % tt, data=np.asarray(cardiac_phase_list[tt], dtype=np.uint8))

    if mode == '3D':
        nx, ny, nz_max = size
        n_train = len(file_list['train'])
        n_test = len(file_list['test'])

    elif mode == '2D':
        nx, ny = size
        n_test = num_slices['test']
        n_train = num_slices['train']

    else:
        raise AssertionError('Wrong mode setting. This should never happen.')

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train'], [n_test, n_train]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    mask_list = {'test': [], 'train': [] }
    img_list = {'test': [], 'train': [] }

    logging.info('Parsing image files')

    train_test_range = ['test', 'train'] if split_test_train else ['train']
    for train_test in train_test_range:

        write_buffer = 0
        counter_from = 0

        for file in file_list[train_test]:

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % file)

            file_base = file.split('.nii.gz')[0]
            file_mask = file_base + '_gt.nii.gz'

            img_dat = utils.load_nii(file)
            mask_dat = utils.load_nii(file_mask)

            img = img_dat[0].copy()
            mask = mask_dat[0].copy()

            img = image_utils.normalise_image(img)

            pixel_size = (img_dat[2].structarr['pixdim'][1],
                          img_dat[2].structarr['pixdim'][2],
                          img_dat[2].structarr['pixdim'][3])

            logging.info('Pixel size:')
            logging.info(pixel_size)

            ### PROCESSING LOOP FOR 3D DATA ################################
            if mode == '3D':

                scale_vector = [pixel_size[0] / target_resolution[0],
                                pixel_size[1] / target_resolution[1],
                                pixel_size[2]/ target_resolution[2]]

                img_scaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               mode='constant')
                mask_scaled = transform.rescale(mask,
                                                scale_vector,
                                                order=0,
                                                preserve_range=True,
                                                multichannel=False,
                                                mode='constant')

                slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                mask_vol = np.zeros((nx, ny, nz_max), dtype=np.uint8)

                nz_curr = img_scaled.shape[2]
                stack_from = (nz_max - nz_curr) // 2

                if stack_from < 0:
                    raise AssertionError('nz_max is too small for the chosen through plane resolution. Consider changing'
                                         'the size or the target resolution in the through-plane.')

                for zz in range(nz_curr):

                    slice_rescaled = img_scaled[:,:,zz]
                    mask_rescaled = mask_scaled[:,:,zz]

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    slice_vol[:,:,stack_from] = slice_cropped
                    mask_vol[:,:,stack_from] = mask_cropped

                    stack_from += 1

                img_list[train_test].append(slice_vol)
                mask_list[train_test].append(mask_vol)

                write_buffer += 1

                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            elif mode == '2D':

                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

                for zz in range(img.shape[2]):

                    slice_img = np.squeeze(img[:, :, zz])
                    slice_rescaled = transform.rescale(slice_img,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')

                    slice_mask = np.squeeze(mask[:, :, zz])
                    mask_rescaled = transform.rescale(slice_mask,
                                                      scale_vector,
                                                      order=0,
                                                      preserve_range=True,
                                                      multichannel=False,
                                                      mode='constant')

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    img_list[train_test].append(slice_cropped)
                    mask_list[train_test].append(mask_cropped)

                    write_buffer += 1

                    # Writing needs to happen inside the loop over the slices
                    if write_buffer >= MAX_WRITE_BUFFER:

                        counter_to = counter_from + write_buffer
                        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                        _release_tmp_memory(img_list, mask_list, train_test)

                        # reset stuff for next iteration
                        counter_from = counter_to
                        write_buffer = 0

        # after file loop: Write the remaining data

        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)


    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr


def _release_tmp_memory(img_list, mask_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=False,
                                split_test_train=True):

    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data
    
    :param input_folder: Folder where the raw ACDC challenge data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'. 2D saves the data slice-by-slice, 3D saves entire volumes
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    if not split_test_train:
        data_file_name = 'data_%s_size_%s_res_%s_onlytrain.hdf5' % (mode, size_str, res_str)
    else:
        data_file_name = 'data_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution, split_test_train=split_test_train)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':

    input_folder = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617'
    preprocessing_folder = 'preproc_data'

    # d=load_and_maybe_process_data(input_folder, preprocessing_folder, '3D', (116,116,28), (2.5,2.5,5))
    d=load_and_maybe_process_data(input_folder, preprocessing_folder, '2D', (212,212), (1.36719, 1.36719))

