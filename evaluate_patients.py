# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging

import argparse
import metrics_acdc_simple
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform

import config.system as sys_config
import model as model
import utils
import image_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()


def score_data(input_folder, output_folder, model_path, exp_config, do_postprocessing=False):

    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, exp_config.model_handle)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])

        total_time = 0
        total_volumes = 0

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

                    for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                        logging.info(' ----- Doing image: -------------------------')
                        logging.info('Doing: %s' % file)
                        logging.info(' --------------------------------------------')

                        file_base = file.split('.nii.gz')[0]
                        file_mask = file_base + '_gt.nii.gz'

                        frame = int(file_base.split('frame')[-1])

                        img_dat = utils.load_nii(file)
                        mask_dat = utils.load_nii(file_mask)

                        img = img_dat[0].copy()
                        mask = mask_dat[0]

                        img = image_utils.normalise_image(img)



                        start_time = time.time()

                        if exp_config.data_mode == '2D':

                            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                            scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                            pixel_size[1] / exp_config.target_resolution[1])

                            predictions = []

                            for zz in range(img.shape[2]):

                                slice_img = np.squeeze(img[:,:,zz])
                                slice_rescaled = transform.rescale(slice_img,
                                                                   scale_vector,
                                                                   order=1,
                                                                   preserve_range=True,
                                                                   multichannel=False,
                                                                   mode='constant')

                                x, y = slice_rescaled.shape

                                x_s = (x - nx) // 2
                                y_s = (y - ny) // 2
                                x_c = (nx - x) // 2
                                y_c = (ny - y) // 2

                                # Crop section of image for prediction
                                if x > nx and y > ny:
                                    slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                                else:
                                    slice_cropped = np.zeros((nx,ny))
                                    if x <= nx and y > ny:
                                        slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                                    elif x > nx and y <= ny:
                                        slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                                    else:
                                        slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]


                                # GET PREDICTION
                                network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                                prediction_cropped = np.squeeze(logits_out[0,...])

                                # ASSEMBLE BACK THE SLICES
                                slice_predictions = np.zeros((x,y,num_channels))
                                # insert cropped region into original image again
                                if x > nx and y > ny:
                                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                                else:
                                    if x <= nx and y > ny:
                                        slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                                    elif x > nx and y <= ny:
                                        slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                                    else:
                                        slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                                # RESCALING ON THE LOGITS
                                prediction = transform.rescale(slice_predictions,
                                                               (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                               order=1,
                                                               preserve_range=True,
                                                               multichannel=False,
                                                               mode='constant')
                                prediction = np.uint8(np.argmax(prediction, axis=-1))
                                predictions.append(prediction)


                            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

                        elif exp_config.data_mode == '3D':


                            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2],
                                          img_dat[2].structarr['pixdim'][3])

                            scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                            pixel_size[1] / exp_config.target_resolution[1],
                                            pixel_size[2] / exp_config.target_resolution[2])

                            vol_scaled = transform.rescale(img,
                                                           scale_vector,
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                            nz_max = exp_config.image_size[2]
                            slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)

                            nz_curr = vol_scaled.shape[2]
                            stack_from = (nz_max - nz_curr) // 2
                            stack_counter = stack_from

                            x, y, z = vol_scaled.shape

                            x_s = (x - nx) // 2
                            y_s = (y - ny) // 2
                            x_c = (nx - x) // 2
                            y_c = (ny - y) // 2

                            for zz in range(nz_curr):

                                slice_rescaled = vol_scaled[:, :, zz]

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

                                slice_vol[:, :, stack_counter] = slice_cropped
                                stack_counter += 1

                            stack_to = stack_counter

                            network_input = np.float32(np.reshape(slice_vol, (1, nx, ny, nz_max, 1)))

                            start_time = time.time()
                            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                            logging.info('Classified 3D: %f secs' % (time.time() - start_time))

                            prediction_nzs = mask_out[0, :, :, stack_from:stack_to]  # non-zero-slices

                            if not prediction_nzs.shape[2] == nz_curr:
                                raise ValueError('sizes mismatch')

                            # ASSEMBLE BACK THE SLICES
                            prediction_scaled = np.zeros(vol_scaled.shape)  # last dim is for logits classes

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

                            logging.info('Prediction_scaled mean %f' % (np.mean(prediction_scaled)))

                            prediction = transform.resize(prediction_scaled,
                                                          (mask.shape[0], mask.shape[1], mask.shape[2], num_channels),
                                                          order=1,
                                                          preserve_range=True,
                                                          mode='constant')
                            prediction = np.argmax(prediction, axis=-1)
                            prediction_arr = np.asarray(prediction, dtype=np.uint8)


                        # This is the same for 2D and 3D again
                        if do_postprocessing:
                            prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

                        elapsed_time = time.time() - start_time
                        total_time += elapsed_time
                        total_volumes += 1

                        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                        if frame == ED_frame:
                            frame_suffix = '_ED'
                        elif frame == ES_frame:
                            frame_suffix = '_ES'
                        else:
                            raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                             (frame, ED_frame, ES_frame))

                        # Save prediced mask
                        out_file_name = os.path.join(output_folder, 'prediction', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]
                        logging.info('saving to: %s' % out_file_name)
                        utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                        # Save GT image
                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % gt_file_name)
                        utils.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                      'difference',
                                                      'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

                        # Save image data to the same folder for convenience
                        image_file_name = os.path.join(output_folder, 'image',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % image_file_name)
                        utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

        logging.info('Average time per volume: %f' % (total_time/total_volumes))

    return init_iteration


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    input_path = sys_config.data_root


    output_path = os.path.join(model_path, 'predictions')

    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_image = os.path.join(output_path, 'image')
    path_eval = os.path.join(output_path, 'eval')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)
    utils.makefolder(path_diff)
    utils.makefolder(path_image)

    init_iteration = score_data(input_path, output_path, model_path, exp_config=exp_config)

    metrics_acdc_simple.main(path_gt, path_pred, path_eval)




