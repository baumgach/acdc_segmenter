"""
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017

DESCRIPTION :
The script provide helpers functions to handle nifti image format:
    - load_nii()
    - save_nii()

to generate metrics for two images:
    - metrics()

And it is callable from the command line (see below).
Each function provided in this script has comments to understand
how they works.

HOW-TO:

This script was tested for python 3.4.

First, you need to install the required packages with
    pip install -r requirements.txt

After the installation, you have two ways of running this script:
    1) python metrics.py ground_truth/patient001_ED.nii.gz prediction/patient001_ED.nii.gz
    2) python metrics.py ground_truth/ prediction/

The first option will print in the console the dice and volume of each class for the given image.
The second option wiil output a csv file where each images will have the dice and volume of each class.


Link: http://acdc.creatis.insa-lyon.fr

"""

import os
from glob import glob
import time
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc, assd
import numpy as np



HEADER = ["Name", "Dice LV", "Volume LV", "Err LV(ml)",
          "Dice RV", "Volume RV", "Err RV(ml)",
          "Dice MYO", "Volume MYO", "Err MYO(ml)"]

#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


#
# Utils function to load and save nifti files with the nibabel package
#
def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


#
# Functions to process files, directories and metrics
#
def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volpred-volgt]

    return res


def compute_metrics_on_directories(dir_gt, dir_pred):
    """
    Function to generate a csv file for each images of two directories.

    Parameters
    ----------

    path_gt: string
    Directory of the ground truth segmentation maps.

    path_pred: string
    Directory of the predicted segmentation maps.
    """

    res_mat, _, _ = compute_metrics_on_directories_raw(dir_gt, dir_pred)

    dice1 = np.mean(res_mat[:,0])
    dice2 = np.mean(res_mat[:,3])
    dice3 = np.mean(res_mat[:,6])

    vold1 = np.mean(res_mat[:,2])
    vold2 = np.mean(res_mat[:,5])
    vold3 = np.mean(res_mat[:,8])

    return [dice1, dice2, dice3, vold1, vold2, vold3]

def compute_metrics_on_directories_raw(dir_gt, dir_pred):
    """
    Calculates all possible metrics (the ones from the metrics script as well as
    hausdorff and average symmetric surface distances)

    :param dir_gt: Directory of the ground truth segmentation maps.
    :param dir_pred: Directory of the predicted segmentation maps.
    :return:
    """

    lst_gt = sorted(glob(os.path.join(dir_gt, '*')), key=natural_order)
    lst_pred = sorted(glob(os.path.join(dir_pred, '*')), key=natural_order)

    res = []
    cardiac_phase = []

    measure_names = ['Dice LV', 'Volume LV', 'Err LV(ml)',
                     'Dice RV', 'Volume RV', 'Err RV(ml)', 'Dice MYO', 'Volume MYO', 'Err MYO(ml)',
                     'Hausdorff LV', 'Hausdorff RV', 'Hausdorff Myo',
                     'ASSD LV', 'ASSD RV', 'ASSD Myo']

    res_mat = np.zeros((len(lst_gt), len(measure_names)))

    ind = 0
    for p_gt, p_pred in zip(lst_gt, lst_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))

        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        res.append(metrics(gt, pred, zooms))
        cardiac_phase.append(os.path.basename(p_gt).split('.nii.gz')[0].split('_')[-1])

        res_mat[ind, :9] = metrics(gt, pred, zooms)

        for ii, struc in enumerate([3,1,2]):

            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1

            res_mat[ind, 9+ii] = hd(gt_binary, pred_binary, voxelspacing=zooms, connectivity=1)
            res_mat[ind, 12+ii] = assd(pred_binary, gt_binary, voxelspacing=zooms, connectivity=1)

        ind += 1

    return res_mat, cardiac_phase, measure_names


def mat_to_df(metrics_out, phase, measure_names):

    num_subj = len(phase)

    measure_ind_dict = {k: v for v, k in enumerate(measure_names)}

    struc_name = ['LV'] * num_subj + ['RV'] * num_subj + ['Myo'] * num_subj

    dices_list = np.concatenate((metrics_out[:, measure_ind_dict['Dice LV']],
                                 metrics_out[:, measure_ind_dict['Dice RV']],
                                 metrics_out[:, measure_ind_dict['Dice MYO']]))
    hausdorff_list = np.concatenate((metrics_out[:, measure_ind_dict['Hausdorff LV']],
                                 metrics_out[:, measure_ind_dict['Hausdorff RV']],
                                 metrics_out[:, measure_ind_dict['Hausdorff Myo']]))
    assd_list = np.concatenate((metrics_out[:, measure_ind_dict['ASSD LV']],
                                 metrics_out[:, measure_ind_dict['ASSD RV']],
                                 metrics_out[:, measure_ind_dict['ASSD Myo']]))

    vol_list = np.concatenate((metrics_out[:, measure_ind_dict['Volume LV']],
                                 metrics_out[:, measure_ind_dict['Volume RV']],
                                 metrics_out[:, measure_ind_dict['Volume MYO']]))

    vol_err_list = np.concatenate((metrics_out[:, measure_ind_dict['Err LV(ml)']],
                                 metrics_out[:, measure_ind_dict['Err RV(ml)']],
                                 metrics_out[:, measure_ind_dict['Err MYO(ml)']]))

    phases_list = phase * 3

    df = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list, 'assd': assd_list,
                       'vol': vol_list, 'vol_err': vol_err_list,
                      'phase': phases_list, 'struc': struc_name})

    return df
#
# def clinical_measures(metrics_out, phase, measure_names, measures_query):
#     pass
#
# def clinical_measures(df):
#
#     #todo: calculate the following clinical measures per patient:
#     # LV EF corr / bias / loa
#     # LV Vol ED corr / bias / loa
#     # LV Vol ES corr / bias / loa
#
#     # RV EF ..
#     # RV Vol ED corr
#     # RF Vol ES corr ..
#
#     # Myo Mass ED corr
#     # Myo Vol ES corr
#
#     lv = df.loc[df['struc'] == 'LV']
#
#     ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
#     ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
#     EF_pred = (ED_vol - ES_vol) / ED_vol
#
#     ED_vol_gt = np.array(lv.loc[lv['phase'] == 'ED']['vol']) - np.array(lv.loc[lv['phase'] == 'ED']['vol_err'])
#     ES_vol_gt = np.array(lv.loc[(lv['phase'] == 'ES')]['vol']) - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_err'])
#
#     EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt
#
#     LV_EF_corr = np.corrcoef(EF_pred, EF_gt)
#
#     lv = df.loc[df['struc'] == 'RV']
#
#     ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
#     ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
#     EF_pred = (ED_vol - ES_vol) / ED_vol
#
#     ED_vol_gt = np.array(lv.loc[lv['phase'] == 'ED']['vol']) - np.array(lv.loc[lv['phase'] == 'ED']['vol_err'])
#     ES_vol_gt = np.array(lv.loc[(lv['phase'] == 'ES')]['vol']) - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_err'])
#
#     EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt
#
#     RV_EF_corr = np.corrcoef(EF_pred, EF_gt)



def boxplot_metrics(dir_gt, dir_pred, eval_dir):

    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_out, phase, measure_names = compute_metrics_on_directories_raw(dir_gt, dir_pred)
    df = mat_to_df(metrics_out, phase, measure_names)

    dice_file = os.path.join(eval_dir, 'boxplot_dice.eps')
    hd_file = os.path.join(eval_dir, 'boxplot_hd.eps')
    assd_file = os.path.join(eval_dir, 'boxplot_assd.eps')

    plt.figure()
    b = sns.boxplot(x='struc', y='dice', hue='phase', data=df, palette="PRGn")
    b.set_xlabel('')
    b.set_ylabel('')
    b.legend(fontsize=30)
    b.tick_params(labelsize=30)
    plt.savefig(dice_file)

    plt.figure()
    b = sns.boxplot(x='struc', y='hd', hue='phase', data=df, palette="PRGn")
    b.set_xlabel('')
    b.set_ylabel('')
    b.legend(fontsize=30)
    b.tick_params(labelsize=30)
    plt.savefig(hd_file)

    plt.figure()
    b = sns.boxplot(x='struc', y='assd', hue='phase', data=df, palette="PRGn")
    b.set_xlabel('')
    b.set_ylabel('')
    b.legend(fontsize=30)
    b.tick_params(labelsize=30)
    plt.savefig(assd_file)


    print('--------------------------------------------')
    print('the following measures should be the same as online')

    for struc_name in ['LV', 'RV', 'Myo']:
        for cardiac_phase in ['ED', 'ES']:
            dat = df.loc[(df['phase'] == cardiac_phase) & (df['struc'] == struc_name)]
            print('{} {}, mean Dice: {}'.format(cardiac_phase, struc_name, np.mean(dat['dice'])))
            print('{} {}, mean Hausdorff: {}'.format(cardiac_phase, struc_name, np.mean(dat['hd'])))
            print('{} {}, mean ASSD: {}'.format(cardiac_phase, struc_name, np.mean(dat['assd'])))

    print('--------------------------------------------')

    return 0


def main(path_gt, path_pred, eval_dir):
    """
    Main function to select which method to apply on the input parameters.
    """

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if os.path.isdir(path_gt) and os.path.isdir(path_pred):

        boxplot_metrics(path_gt, path_pred, eval_dir)

        [dice1, dice2, dice3, vold1, vold2, vold3] = compute_metrics_on_directories(path_gt, path_pred)

        print('Dice 1: %f' % dice1)
        print('Dice 2: %f' % dice2)
        print('Dice 3: %f' % dice3)
        print('Mean dice: %f' % np.mean([dice1, dice2, dice3]))

    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute ACDC challenge metrics.")
    parser.add_argument("GT_IMG", type=str, help="Ground Truth image")
    parser.add_argument("PRED_IMG", type=str, help="Predicted image")
    parser.add_argument("EVAL_DIR", type=str, help="path to output directory", default='.')
    args = parser.parse_args()
    main(args.GT_IMG, args.PRED_IMG, args.EVAL_DIR)
