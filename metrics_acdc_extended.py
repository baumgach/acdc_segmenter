"""
Code for evaluation of acdc metrics. Has some additional functions w.r.t
metrics_acdc_simple.py, such as plotting box plots and contours. 

Authors:
Christian F. Baumgartner (c.f.baumgartner@gmail.com)
Lisa. M. Koch (lisa.margret.koch@gmail.com)

Extended from code made available by
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017
Link: http://acdc.creatis.insa-lyon.fr

"""

import os
from glob import glob
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc, assd
import numpy as np

import SimpleITK as sitk
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

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

    res_mat, _, _, _ = compute_metrics_on_directories_raw(dir_gt, dir_pred)

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
    file_names = []

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

        file_names.append(os.path.basename(p_pred))

        res_mat[ind, :9] = metrics(gt, pred, zooms)

        for ii, struc in enumerate([3,1,2]):

            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1

            res_mat[ind, 9+ii] = hd(gt_binary, pred_binary, voxelspacing=zooms, connectivity=1)
            res_mat[ind, 12+ii] = assd(pred_binary, gt_binary, voxelspacing=zooms, connectivity=1)

        ind += 1

    return res_mat, cardiac_phase, measure_names, file_names


def mat_to_df(metrics_out, phase, measure_names, file_names):

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
    file_names_list = file_names * 3

    df = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list, 'assd': assd_list,
                       'vol': vol_list, 'vol_err': vol_err_list,
                      'phase': phases_list, 'struc': struc_name, 'filename': file_names_list})

    return df
#
# def clinical_measures(metrics_out, phase, measure_names, measures_query):
#     pass
#
def clinical_measures(df):

    import scipy.stats as stats

    # LV EF corr / bias / loa
    # LV Vol ED corr / bias / loa
    # LV Vol ES corr / bias / loa

    # RV EF ..
    # RV Vol ED corr
    # RF Vol ES corr ..

    # Myo Mass ED corr
    # Myo Vol ES corr

    logging.info('-----------------------------------------------------------')
    logging.info('the following measures should be the same as online')

    for struc_name in ['LV', 'RV']:

        lv = df.loc[df['struc'] == struc_name]

        ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
        ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
        EF_pred = (ED_vol - ES_vol) / ED_vol

        ED_vol_gt = np.array(lv.loc[lv['phase'] == 'ED']['vol']) - np.array(lv.loc[lv['phase'] == 'ED']['vol_err'])
        ES_vol_gt = np.array(lv.loc[(lv['phase'] == 'ES')]['vol']) - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_err'])

        EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt

        LV_EF_corr  = stats.pearsonr(EF_pred, EF_gt)
        logging.info('{}, EF corr: {}'.format(struc_name, LV_EF_corr[0]))


def get_measures(dir_gt, dir_pred, eval_dir):

    metrics_out, phase, measure_names, file_names = compute_metrics_on_directories_raw(dir_gt, dir_pred)
    df = mat_to_df(metrics_out, phase, measure_names, file_names)

    # clinical_measures(df)
    print_table1(df, eval_dir)
    print_table2(df, eval_dir)
    boxplot_metrics(df, eval_dir)


def sitk_show(img, title=None, margin=0.05, dpi=40, out_file=None):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

    if out_file is not None:
        plt.savefig(out_file)


def plot_examples(dir_gt, dir_pred, dir_img, eval_dir):


    # median dice: 0.900 (patient060_ED.nii.gz)
    # worst dice: 0.846 (patient085_ED.nii.gz)
    # best dice: 0.947 (patient040_ED.nii.gz)
    subjects = ['patient060_ED.nii.gz', 'patient085_ED.nii.gz', 'patient040_ED.nii.gz']

    for subj_file in subjects:
        plot_basal_mid_apical(subj_file, eval_dir, dir_gt, dir_pred, dir_img)

def plot_basal_mid_apical(subj_file, eval_dir, dir_gt, dir_pred, dir_img):


    subj_id = subj_file.split('.nii')[0]

    gt = sitk.ReadImage(os.path.join(dir_gt, subj_file))
    pred = sitk.ReadImage(os.path.join(dir_pred, subj_file))
    img = sitk.ReadImage(os.path.join(dir_img, subj_file))

    imgSmoothInt = sitk.Cast(sitk.RescaleIntensity(img), pred.GetPixelID())

    num_slices = img.GetDepth()

    for ind in [0, num_slices // 2, num_slices - 1]:

        sitk_show(sitk.LabelOverlay(imgSmoothInt[:,:,ind], gt[:,:,ind]),
                  out_file=os.path.join(eval_dir, 'gt_{}_{}.eps'.format(subj_id, ind)))
        sitk_show(sitk.LabelOverlay(imgSmoothInt[:,:,ind], pred[:,:,ind]),
                  out_file=os.path.join(eval_dir, 'pred_{}_{}.eps'.format(subj_id, ind)))

    pass


def print_table1(df, eval_dir):
    """
    prints mean (+- std) values for Dice and ASSD, all structures, averaged over both phases.

    :param df:
    :param eval_dir:
    :return:
    """

    out_file = os.path.join(eval_dir, 'table1.txt')

    header_string = ' & '
    line_string = 'METHOD '


    for s_idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
        for measure in ['dice', 'assd']:

            header_string += ' & {} ({}) '.format(measure, struc_name)

            dat = df.loc[df['struc'] == struc_name]

            if measure == 'dice':
                line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
            else:
                line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

        if s_idx < 2:
            header_string += ' & '
            line_string += ' & '

    header_string += ' \\\\ \n'
    line_string += ' \\\\ \n'

    with open(out_file, "w") as text_file:
        text_file.write(header_string)
        text_file.write(line_string)

    return 0

def print_table2(df, eval_dir):
    """
    prints mean (+- std) values for Dice, ASSD and HD, all structures, both phases separately

    :param df:
    :param eval_dir:
    :return:
    """

    out_file = os.path.join(eval_dir, 'table2.txt')


    with open(out_file, "w") as text_file:

        for idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['dice', 'assd', 'hd']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['struc'] == struc_name)]

                    if measure == 'dice':

                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

    return 0

def boxplot_metrics(df, eval_dir):

    import matplotlib.pyplot as plt
    import seaborn as sns

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

    return 0


def print_stats(df):

    logging.info('-----------------------------------------------------------')
    logging.info('The following measures should be equivalent to those ')
    logging.info('obtained from the online evaluation platform')

    for struc_name in ['LV', 'RV', 'Myo']:

        logging.info(struc_name)

        for cardiac_phase in ['ED', 'ES']:

            logging.info('    {}'.format(cardiac_phase))

            dat = df.loc[(df['phase'] == cardiac_phase) & (df['struc'] == struc_name)]

            for measure_name in ['dice', 'hd', 'assd']:

                logging.info('       {} -- mean (std): {:.3f} ({:.3f}) '.format(measure_name,
                                                                     np.mean(dat[measure_name]), np.std(dat[measure_name])))

                ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name])//2]
                logging.info('             median {}: {:.3f} ({})'.format(measure_name,
                                                            dat[measure_name].iloc[ind_med], dat['filename'].iloc[ind_med]))

                ind_worst = np.argsort(dat[measure_name]).iloc[0]
                logging.info('             worst {}: {:.3f} ({})'.format(measure_name,
                                                            dat[measure_name].iloc[ind_worst], dat['filename'].iloc[ind_worst]))

                ind_best = np.argsort(dat[measure_name]).iloc[-1]
                logging.info('             best {}: {:.3f} ({})'.format(measure_name,
                                                            dat[measure_name].iloc[ind_best], dat['filename'].iloc[ind_best]))


def main(path_gt, path_pred, eval_dir):
    """
    Main function to select which method to apply on the input parameters.
    """

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if os.path.isdir(path_gt) and os.path.isdir(path_pred):

        metrics_out, phase, measure_names, file_names = compute_metrics_on_directories_raw(path_gt, path_pred)
        df = mat_to_df(metrics_out, phase, measure_names, file_names)
        print_stats(df)
        print_table1(df, eval_dir)
        print_table2(df, eval_dir)

        [dice1, dice2, dice3, vold1, vold2, vold3] = compute_metrics_on_directories(path_gt, path_pred)

        logging.info('------------Average Dice Figures----------')
        logging.info('Dice 1: %f' % dice1)
        logging.info('Dice 2: %f' % dice2)
        logging.info('Dice 3: %f' % dice3)
        logging.info('Mean dice: %f' % np.mean([dice1, dice2, dice3]))
        logging.info('------------------------------------------')

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
