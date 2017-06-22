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

# import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


def placeholder_inputs(batch_size, nx, ny):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='labels')
    return images_placeholder, labels_placeholder

def score_data(input_folder, output_folder, model_path, inference_handle):

    nx = 288
    ny = 288

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

                        print(' ----- Doing image: -------------------------')
                        print(file)
                        print(' --------------------------------------------')

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

                        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])

                        predictions = []

                        start_time = time.time()
                        for zz in range(img.shape[2]):

                            slice_img = np.squeeze(img[:,:,zz])
                            slice_rescaled = image_utils.rescale_image(slice_img, pixel_size)

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

                            prediction_cropped = np.squeeze(mask_out[0,...])
                            prediction_cropped = post_process_prediction(prediction_cropped)



                            # ASSEMBLE BACK THE SLICES
                            slice_predictions = np.zeros((x,y))

                            # Not really sure why this is necessary...
                            x_s -= 1
                            y_s -= 1
                            x_c -= 1
                            y_c -= 1

                            # insert cropped region into original image again
                            if x > nx and y > ny:
                                slice_predictions[x_s:x_s+nx, y_s:y_s+ny] = prediction_cropped
                            else:
                                if x <= nx and y > ny:
                                    slice_predictions[:, y_s:y_s+ny] = prediction_cropped[x_c:x_c+ x, :]
                                elif x > nx and y <= ny:
                                    slice_predictions[x_s:x_s + nx, :] = prediction_cropped[:, y_c:y_c + y, :]
                                else:
                                    slice_predictions[:, :] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y, :]

                            prediction = image_utils.resize_image(slice_predictions, (mask.shape[0], mask.shape[1]), interp=cv2.INTER_NEAREST)
                            #prediction = image_utils.resize_labels_lisa_style(slice_predictions, (mask.shape[0], mask.shape[1]), num_labels=4)

                            predictions.append(prediction)


                        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

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
    # model_path = './good_models/lisa_net_deeper_adam_reg0.00005_lr0.001_long'  # 0.777998
    # model_path = './good_models/lisa_net_deeper_adam_autosched2' # 0.800817
    # model_path = './good_models/lisa_net_deeper_adam_reg0.00005_lr0.001_augm3' # 0.765803
    # model_path = './good_models/dilation_after_maxpool__reg0.00005' # 0.764761
    # model_path = './good_models/lisa_net_deeper_wd_new_0.00000' # 0.822012
    # model_path = './acdc_logdir/lisa_net_deeper_sgd_sched_reg0.00005_lr0.1_aug_bn2'  # 0.740749 (early stop)
    # model_path = './good_models/lisa_net_deeper_autosched_mom_reg0.00005_lr0.1_bn'  # 0.813045
    # model_path = './acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.001_aug' #  0.800794 -- base model for refinements below
    # model_path = './acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.0001_aug_refdice' # 0.799363  --> refinement made it slighly worse...
    # model_path = './acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.001_aug_refunweighted'  # 0.809884 --> this refinement made it almost 1% better
    # model_path = './acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.0001_aug_refunweighted'  # 0.808602 --> this refinement also improved it a bit
    # model_path = './acdc_logdir/lisa_net_deeper_mom0.9_sched_reg0.00005_lr0.1_aug_newbn' # 0.812817
    # model_path = './acdc_logdir/lisa_net_deeper_mom0.9_sched_reg0.00000_lr0.1_aug_newbn' #0.812500
    # model_path = './acdc_logdir/lisa_net_deeper_adam_nosched_reg0.00000_lr0.01_aug_newbn' # 0.825071, 0.833786
    # model_path = os.path.join(base_path, 'unet_gbn_adam_reg0.00000_lr0.01_aug')
    #model_path = os.path.join(base_path, 'VGG16_FCN_8_gbn_adam_reg0.00000_lr0.01_aug')
    #model_path = os.path.join(base_path, 'unet_bn_long')
    # model_path = os.path.join(base_path, 'unet_bn_lisadata')
    # model_path = os.path.join(base_path, 'unet_bn_fliplr')
    # model_path = os.path.join(base_path, 'unet_bn_rotate')
    # model_path = os.path.join(base_path, 'unet_bn_rerun')
    # model_path = os.path.join(base_path, 'unet_dilated_bn')
    # model_path = os.path.join(base_path, 'unet_bn_RV_more_weight')
    model_path = os.path.join(base_path, 'unet_bn_merged_wenjia_new')

    # inference_handle = model_zoo.lisa_net_deeper
    # inference_handle = model_zoo.lisa_net_deeper_bn
    # inference_handle = model_zoo.dilation_after_max_pool
    inference_handle = model_zoo.unet_bn
    # inference_handle = model_zoo.unet_dilated_bn
    # inference_handle = model_zoo.VGG16_FCN_8_bn

    input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    output_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/prediction_data/'

    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_image = os.path.join(output_path, 'image')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)
    utils.makefolder(path_diff)
    utils.makefolder(path_image)

    score_data(input_path, output_path, model_path, inference_handle)

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


