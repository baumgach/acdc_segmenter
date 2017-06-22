import utils
import image_utils

import os
import glob

#import matplotlib.pyplot as plt

import numpy as np
import cv2
import logging

import model
import models_refine3d
import models3d
import tensorflow as tf

import model_zoo
import time
import gc

import h5py
from skimage import transform
from importlib.machinery import SourceFileLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def placeholder_inputs(batch_size, nx, ny, nz):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, nz, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, nz, 1), name='labels')
    return images_placeholder, labels_placeholder

def score_cases(input_folder, output_folder, model_path, inference_handle, downsampling_factor, pre_classifier=None, pre_classifier_path=None):

    nx = 288
    ny = 288
    nz_max = 24  #24
    ds_fact = downsampling_factor
    use_pred = False

    if use_pred:
        num_feature_channels = 4
    else:
        num_feature_channels = 64

    if pre_classifier:
        num_channels = num_feature_channels  # 64
    else:
        num_channels = 1

    g3D = tf.Graph()
    g2D = tf.Graph()

    with g3D.as_default():

        input3D_pl = tf.placeholder(tf.float32, shape=(1, nx//ds_fact, ny//ds_fact, nz_max, num_channels), name='images')
        mask_3d_pl, softmax_3d_pl = model.predict(input3D_pl, inference_handle)

        saver3d = tf.train.Saver()
        init3d = tf.global_variables_initializer()

    with g2D.as_default():

        if pre_classifier:

            input2D_pl = tf.placeholder(tf.float32, shape=(1, nx, ny, 1), name='images')
            pred_2d_pl, fm_2d_pl = pre_classifier(input2D_pl, training=tf.constant(False))

            saver2d = tf.train.Saver()
            init2d = tf.global_variables_initializer()

    if pre_classifier:
        sess2d = tf.Session(graph=g2D)
        sess2d.run(init2d)

    sess3d = tf.Session(graph=g3D)
    sess3d.run(init3d)

    # automatically find latest best file
    checkpoint_path = utils.get_best_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
    logging.info('Checkpoint path: %s' % checkpoint_path)
    saver3d.restore(sess3d, checkpoint_path)

    if pre_classifier:
        logging.info('Also loading preclassifier')
        checkpoint_path = utils.get_best_model_checkpoint_path(preclassifier_path, 'model_best_dice.ckpt')
        saver2d.restore(sess2d, checkpoint_path)

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

            if train_test == 'test':

                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = folder.lstrip('patient')
                ED_frame = int(infos['ED'])
                ES_frame = int(infos['ES'])

                start_time_total = time.time()

                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                    logging.info(' ----- Doing image: -------------------------')
                    logging.info(file)
                    logging.info(' --------------------------------------------')


                    logging.info('Doing: %s' % file)

                    file_img = file
                    file_base = file.split('.nii.gz')[0]
                    file_mask = file_base + '_gt.nii.gz'

                    frame = int(file_base.split('frame')[-1])

                    img_dat = utils.load_nii(file_img)
                    mask_dat = utils.load_nii(file_mask)

                    img = img_dat[0].copy()
                    mask = mask_dat[0]

                    img = image_utils.normalise_image(img)

                    pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2], img_dat[2].structarr['pixdim'][3])

                    if pixel_size[2] == 10:
                        scale_z = 2.0
                    else:
                        scale_z = 1.0

                    scale_vector = [pixel_size[0], pixel_size[1], scale_z]

                    img_scaled = transform.rescale(img, scale_vector, order=1, preserve_range=True, multichannel=False)

                    slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                    feature_maps_vol = np.zeros((nx, ny, nz_max, num_channels), dtype=np.float32)

                    nz_curr = img_scaled.shape[2]
                    stack_from = (nz_max - nz_curr) // 2
                    stack_counter = stack_from

                    x, y, z = img_scaled.shape

                    x_s = (x - nx) // 2
                    y_s = (y - ny) // 2
                    x_c = (nx - x) // 2
                    y_c = (ny - y) // 2

                    for zz in range(nz_curr):

                        slice_rescaled = img_scaled[:,:,zz]

                        if x > nx and y > ny:
                            slice_cropped = slice_rescaled[x_s:x_s + nx, y_s:y_s + ny]
                        else:
                            slice_cropped = np.zeros((nx, ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c + x, :] = slice_rescaled[:, y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]

                            else:
                                slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice_rescaled[:, :]


                        if pre_classifier:
                            network_input = np.float32(np.reshape(slice_cropped, (1, nx, ny, 1)))

                            # pred, feature_map = np.zeros((1,288,288,4)), np.zeros((1,288,288,64))
                            start_time = time.time()
                            pred, feature_map = sess2d.run([pred_2d_pl, fm_2d_pl], feed_dict={input2D_pl: network_input})
                            if use_pred:
                                feature_maps_vol[:, :, stack_counter, :] = pred
                            else:
                                feature_maps_vol[:, :, stack_counter, :] = feature_map

                            logging.info('2D classified slice %d: %f secs' % (zz, time.time()-start_time))

                            # plt.figure()
                            # plt.imshow(np.squeeze(np.argmax(pred, axis=-1)))
                            # plt.show()

                        slice_vol[:,:,stack_counter] = slice_cropped
                        stack_counter += 1

                    stack_to = stack_counter

                    # prediction
                    if not pre_classifier:
                        network_input = np.float32(np.reshape(slice_vol, (1, nx, ny, nz_max, 1)))
                    else:
                        network_input = np.float32(np.reshape(feature_maps_vol, (1, nx, ny, nz_max, num_channels)))

                    if not ds_fact == 1:
                        network_input = transform.rescale(network_input, (1, 1.0/ds_fact, 1.0/ds_fact, 1, 1), order=1, preserve_range=True, multichannel=False)

                    # plt.figure()
                    # plt.imshow(np.squeeze(feature_maps_vol[:,:,11,:]))
                    # plt.show()

                    #with g3D.as_default():
                    start_time = time.time()
                    mask_out, logits_out = sess3d.run([mask_3d_pl, softmax_3d_pl], feed_dict={input3D_pl: network_input})
                    logging.info('Classified 3D: %f secs' % (time.time()-start_time))
                    # mask_out, logits_out = np.zeros((1,288,288,22)), np.zeros((1,288,288,22,4))

                    # plt.figure()
                    # plt.imshow(np.squeeze(mask_out[0,:,:,11]))
                    # plt.show()

                    if not ds_fact == 1:
                        logits_out = transform.rescale(logits_out, (1, ds_fact, ds_fact, 1, 1), order=1, preserve_range=True, multichannel=False)
                        mask_out = np.uint8(np.argmax(logits_out, axis=-1))

                    # prediction_nzs = logits_out[0,:,:,stack_from:stack_to, :]   # non-zero-slices
                    prediction_nzs = mask_out[0,:,:,stack_from:stack_to]   # non-zero-slices

                    if not prediction_nzs.shape[2] == nz_curr:
                        raise ValueError('sizes mismatch')

                    # ASSEMBLE BACK THE SLICES

                    # prediction_scaled = np.zeros(list(img_scaled.shape) + [4])  # last dim is for logits classes
                    prediction_scaled = np.zeros(img_scaled.shape)  # last dim is for logits classes

                    # Not really sure why this is necessary...
                    x_s -= 1
                    y_s -= 1
                    x_c -= 1
                    y_c -= 1

                    # insert cropped region into original image again
                    if x > nx and y > ny:
                        prediction_scaled[x_s:x_s + nx, y_s:y_s + ny, :] = prediction_nzs
                    else:
                        if x <= nx and y > ny:
                            prediction_scaled[:, y_s:y_s + ny, :] = prediction_nzs[x_c:x_c + x, :, :]
                        elif x > nx and y <= ny:
                            prediction_scaled[x_s:x_s + nx, :, :] = prediction_nzs[:, y_c:y_c + y, :]
                        else:
                            prediction_scaled[:, :, :] = prediction_nzs[x_c:x_c + x, y_c:y_c + y, :]


                    if scale_z == 2:
                        # This prevents from the Apices beeing chopped off when interpolating across thick slices
                        prediction_scaled = image_utils.scale_z_with_max(prediction_scaled, scale=int(scale_z))

                    prediction = transform.resize(prediction_scaled, (mask.shape[0], mask.shape[1], mask.shape[2]), order=0, preserve_range=True)
                    # prediction = np.argmax(prediction, axis=3)

                    prediction = np.asarray(prediction, dtype=np.uint8)
                   

                    if frame == ED_frame:
                        frame_suffix = '_ED'
                    elif frame == ES_frame:
                        frame_suffix = '_ES'
                    else:
                        raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                         (frame, ED_frame, ES_frame))

                    out_file_name = os.path.join(output_folder, 'prediction',
                                                 'patient' + patient_id + frame_suffix + '.nii.gz')

                    out_affine = mask_dat[1]
                    out_header = mask_dat[2]

                    logging.info('saving to: %s' % out_file_name)
                    utils.save_nii(out_file_name, prediction, out_affine, out_header)

                    gt_file_name = os.path.join(output_folder, 'ground_truth',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % gt_file_name)
                    utils.save_nii(gt_file_name, mask, out_affine, out_header)

                    # Compute difference mask between GT and pred
                    difference_mask = np.where(np.abs(prediction - mask) > 0, [1], [0])
                    difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                    diff_file_name = os.path.join(output_folder, 'difference',
                                                  'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % diff_file_name)
                    utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

                    # Save image data for convencience
                    image_file_name = os.path.join(output_folder, 'image',
                                                   'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % image_file_name)
                    utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

                    # Calculate wrong voxels
                    wrong_pixels = np.sum(difference_mask)
                    logging.info('Wrong pixels: %d' % wrong_pixels)

                elapsed_time_total = time.time() - start_time_total
                logging.info('Evaluation of this file took %f secs overall' % elapsed_time_total)


if __name__ == '__main__':

    base_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/acdc_logdir3d/'

    preclassifier_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/acdc_logdir/unet_bn_rerun/'
    preclassifier_model = None  #models_refine3d.unet_bn
    # preclassifier_model = models_refine3d.unet_bn


    # EXP_NAME = 'refine_residual_on_FMs_2'
    # EXP_NAME = 'unet_2D_AND_3D_newarch'
    EXP_NAME = 'unet_3D'
    # EXP_NAME = 'unet_2D_as_3D_2'


    model_path = os.path.join(base_path, EXP_NAME)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_configs = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    inference_handle = exp_configs.model_handle
    downsampling_factor = exp_configs.down_sampling_factor

    logging.info('----------------------- Using the following settings: -----------------------')
    logging.info('model_path: %s' % model_path)
    logging.info('downsampling_factor: %s' % downsampling_factor)
    logging.info('----------------------------------------------------- -----------------------')

    input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    output_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/prediction_data_3D/'

    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_image = os.path.join(output_path, 'image')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)
    utils.makefolder(path_diff)
    utils.makefolder(path_image)

    score_cases(input_path, output_path, model_path, inference_handle, downsampling_factor, pre_classifier=preclassifier_model, pre_classifier_path=preclassifier_path)

    import metrics_acdc_nocsv
    [dice1, dice2, dice3, vold1, vold2, vold3] = metrics_acdc_nocsv.compute_metrics_on_directories(path_gt, path_pred)

    print('Dice 1: %f' % dice1)
    print('Dice 2: %f' % dice2)
    print('Dice 3: %f' % dice3)
    print('Mean dice: %f' % np.mean([dice1, dice2, dice3]))

    logging.info('Model: %s' % model_path)

    logging.info('Dice 1: %f' % dice1)
    logging.info('Dice 2: %f' % dice2)
    logging.info('Dice 3: %f' % dice3)
    logging.info('Mean dice: %f' % np.mean([dice1, dice2, dice3]))
