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
from importlib.machinery import SourceFileLoader

# import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

from skimage import transform


def placeholder_inputs(batch_size, nx, ny):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='labels')
    return images_placeholder, labels_placeholder

def score_data(input_folder, output_folder, model_path, inference_handle, image_size, target_resolution):

    nx, ny = image_size
    batch_size = 1

    images_pl, labels_placeholder = placeholder_inputs(batch_size, nx, ny)
    mask_pl, softmax_pl = model.predict(images_pl, inference_handle)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        # Get the latest checkpoint file
        #saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # Use specific best file
        # saver.restore(sess, os.path.join(model_path, 'model_best_dice.ckpt-5799'))

        # automatically find latest best file
        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

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

                        file_img = file
                        file_base = file.split('.nii.gz')[0]
                        file_mask = file_base + '_gt.nii.gz'

                        frame = int(file_base.split('frame')[-1])

                        img_dat = utils.load_nii(file_img)
                        mask_dat = utils.load_nii(file_mask)

                        img = img_dat[0].copy()
                        mask = mask_dat[0]

                        img = image_utils.normalise_image(img)

                        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])

                        scaling_factor = (pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1])

                        predictions = []

                        start_time = time.time()
                        for zz in range(img.shape[2]):

                            slice_img = np.squeeze(img[:,:,zz])
                            slice_rescaled = image_utils.rescale_image(slice_img, scaling_factor)

                            x, y = slice_rescaled.shape

                            if x > 500:
                                print('W: Buggy case downscaling by factor 2')
                                slice_rescaled = image_utils.rescale_image(slice_rescaled, (0.5, 0.5))
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

                            # plt.figure()
                            # plt.imshow(slice_cropped, cmap='gray')

                            # GET PREDICTION
                            network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})

                            # plt.figure()
                            # plt.imshow(np.squeeze(mask_out[0,...]))
                            # plt.show()

                            # prediction_cropped = np.squeeze(mask_out[0,...])
                            prediction_cropped = np.squeeze(logits_out[0,...])
                            #prediction_cropped = post_process_prediction(prediction_cropped)



                            # ASSEMBLE BACK THE SLICES
                            # slice_predictions = np.zeros((x,y))
                            slice_predictions = np.zeros((x,y,4))

                            # Not really sure why this is necessary...
                            # x_s -= 1
                            # y_s -= 1
                            # x_c -= 1
                            # y_c -= 1

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

                            # prediction = image_utils.rescale_image(slice_predictions, (1.0/scaling_factor[0], 1.0/scaling_factor[1]), interp=cv2.INTER_NEAREST)
                            prediction = transform.rescale(slice_predictions, (1.0/scaling_factor[0], 1.0/scaling_factor[1], 1), order=1,
                                              preserve_range=True, multichannel=False)

                            prediction = np.uint8(np.argmax(prediction, axis=-1))

                            if not prediction.shape == mask.shape[:2]:
                                logging.warning('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                                logging.warning('Prediction shape is not mask.shape')
                                logging.warning('Prediction shape')
                                logging.warning(prediction.shape)
                                logging.warning('Mask shape')
                                logging.warning(mask.shape)

                            #network_input = transform.rescale(network_input, (1, 1.0/ds_fact, 1.0/ds_fact, 1, 1), order=1, preserve_range=True, multichannel=False)


                            # prediction = image_utils.resize_image(slice_predictions, (mask.shape[0], mask.shape[1]), interp=cv2.INTER_NEAREST)
                            #prediction = image_utils.resize_labels_lisa_style(slice_predictions, (mask.shape[0], mask.shape[1]), num_labels=4)

                            predictions.append(prediction)


                        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

                        prediction_arr = utils.post_process_prediction_3D(prediction_arr)

                        elapsed_time = time.time() - start_time
                        total_time += elapsed_time
                        total_volumes += 1

                        logging.info('Evaluation of volume took %f secs.' % elapsed_time)
                        print('Evaluation of volume took %f secs.' % elapsed_time)

                        if frame == ED_frame:
                            frame_suffix = '_ED'
                        elif frame == ES_frame:
                            frame_suffix = '_ES'
                        else:
                            raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                             (frame, ED_frame, ES_frame))

                        out_file_name = os.path.join(output_folder, 'prediction', 'patient' + patient_id + frame_suffix + '.nii.gz')

                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]

                        print('saving to: %s' % out_file_name)
                        utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        print('saving to: %s' % gt_file_name)
                        utils.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Compute difference mask between GT and pred
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder, 'difference',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                        print('saving to: %s' % diff_file_name)
                        utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

                        # Save image data for convencience
                        image_file_name = os.path.join(output_folder, 'image',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                        print('saving to: %s' % image_file_name)
                        utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

                        # Calculate wrong voxels
                        wrong_pixels = np.sum(difference_mask)
                        print('Wrong pixels: %d' % wrong_pixels)

        print('Average time per volume: %f' % (total_time/total_volumes))


# def get_prediction_for_image(img, sess, images_placeholder, mask, softmax, model_path, inference_handle):
#
#     nx = img.shape[0]
#     ny = img.shape[1]
#
#
#
#         # logits_crf = np.squeeze(logits_out)
#         # logits_crf = np.transpose(logits_crf, (2, 0, 1))
#         #
#         # d = dcrf.DenseCRF2D(nx, ny, 4)  # width, height, nlabels
#         # U = unary_from_softmax(logits_crf)
#         # U = U.reshape((4, -1))
#         # d.setUnaryEnergy(U)
#         # d.addPairwiseGaussian(sxy=3, compat=3)
#         #
#         # Q = d.inference(50)
#         #
#         # # Find out the most probable class for each pixel.
#         # MAP = np.argmax(Q, axis=0)
#         # MAP = np.reshape(MAP, [nx, ny])
#
#     return mask_out

def post_process_prediction(img):

    # Hook for some possible image postprocessing

    # nx = img.shape[0]
    # ny = img.shape[1]
    #
    # logits_crf = np.squeeze(logits_out)
    # logits_crf = np.transpose(logits_crf, (2, 0, 1))
    #
    # d = dcrf.DenseCRF2D(nx, ny, 4)  # width, height, nlabels
    # U = unary_from_softmax(logits_crf)
    # U = U.reshape((4, -1))
    # d.setUnaryEnergy(U)
    # d.addPairwiseGaussian(sxy=3, compat=3)
    #
    # Q = d.inference(50)
    #
    # # Find out the most probable class for each pixel.
    # MAP = np.argmax(Q, axis=0)
    # MAP = np.reshape(MAP, [nx, ny])

    return img


if __name__ == '__main__':

    base_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/acdc_logdir/'

    EXP_NAME = 'unet_bn_rerun'  # 0.89623 @ 14499
    # EXP_NAME = 'unet_bn_rerun_smaller_batchsize' # 0.893037
    # EXP_NAME = 'unet_bn_bottleneck16' # 0.890652
    # EXP_NAME = 'unet_bn_fixed_xent_and_dice'  #0.876541
    
    # EXP_NAME = 'unet_bn_fixed_undw_xent' # 0.885276  -- finished  @ 17299
    # EXP_NAME = 'unet_bn_fixed' # 0.891060 @ 18199,
    # EXP_NAME = 'unet_bn_fixed_dice' #  0.867664  w/o pp 0.865265, 0.877424 @ 17799
    # EXP_NAME = 'unet_bn_224_224'

    model_path = os.path.join(base_path, EXP_NAME)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_configs = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    inference_handle = exp_configs.model_handle
    image_size = exp_configs.image_size

    input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    output_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/prediction_data/'
    eval_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/prediction_data/tmp_eval'

    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_image = os.path.join(output_path, 'image')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)
    utils.makefolder(path_diff)
    utils.makefolder(path_image)

    if image_size[0] == 288:
        target_resolution = (1.0, 1.0)
    elif image_size[0] == 224:
        target_resolution = (1.36719, 1.36719)
    else:
        raise ValueError('Unknown target resolution')

    score_data(input_path, output_path, model_path, inference_handle, image_size=image_size, target_resolution=target_resolution)

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


