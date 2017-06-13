import utils
import image_utils

import os
import glob

import matplotlib.pyplot as plt

import numpy as np
import cv2

import model as model
import tensorflow as tf

import model_zoo

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


def placeholder_inputs(batch_size, nx, ny):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, nx, ny, 1), name='labels')
    return images_placeholder, labels_placeholder

def score_data(input_folder, output_folder, model_path, inference_handle):

    nx = 288
    ny = 288

    images_pl, labels_placeholder = placeholder_inputs(1, nx, ny)
    mask_pl, softmax_pl = model.predict(images_pl, inference_handle)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))


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

                        predictions = []

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

                            # GET PREDICTION
                            network_input = np.float32(np.reshape(slice_cropped, (1, nx, ny, 1)))
                            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                            prediction_cropped = np.squeeze(mask_out)
                            prediction_cropped = post_process_prediction(prediction_cropped)

                            # ASSEMBLE BACK THE SLICES
                            slice_predictions = np.zeros((x,y))

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

                            # prediction = utils.rescale_image(slice_predictions, (1.0 / pixel_size[0], 1.0 / pixel_size[1]))
                            prediction = image_utils.resize_image(slice_predictions, (mask.shape[0], mask.shape[1]), interp=cv2.INTER_NEAREST)

                            predictions.append(prediction)


                        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

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

    # model_path = './good_models/lisa_net_deeper_adam_reg0.00005_lr0.001_long'  # 0.777998
    model_path = './good_models/lisa_net_deeper_adam_autosched2'
    inference_handle = model_zoo.lisa_net_deeper

    input_path = '/scratch_net/bmicdl03/data/ACDC_challenge/'
    output_path = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/prediction_data/'

    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)

    score_data(input_path, output_path, model_path, inference_handle)

    import metrics_acdc_nocsv
    [dice1, dice2, dice3, vold1, vold2, vold3] = metrics_acdc_nocsv.compute_metrics_on_directories(path_gt, path_pred)

    print('Dice 1: %f' % dice1)
    print('Dice 2: %f' % dice2)
    print('Dice 3: %f' % dice3)
    print('Mean dice: %f' % np.mean([dice1, dice2, dice3]))


