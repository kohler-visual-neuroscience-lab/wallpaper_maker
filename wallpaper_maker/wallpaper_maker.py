"""

The code will generate an arbitrary number of exemplars belonging to each of the 17 groups, as well as matched control exemplars using phase-scrambling and Portilla-Simoncelli scrambling.

To run the code use the following function with the available optional parameters: 

make_set(groups to create, number of images per group, visual angle, wallpaper size in pixels, .... 
sizing method, ratio of sizing method to wallpaper, filter frequency wallpapers,....
image save format, save raw, number of interpolated phase scrambled images, Portilla-Simoncelli scrambled, ....
use same magnitude across exemplars, print diagnostic images, save path, image masking)

Check function for expected data types for each argument.

"""
from numpy.core._multiarray_umath import ndarray

import pss_g
import filter
import os
from datetime import datetime

import numpy as np
import cairo as cr
import numpy.matlib  # could this not be replaced with just np.matlib?
import math
import time
from PIL import Image, ImageDraw
from skimage import draw as skd
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
import cv2 as cv

import logging
import argparse

from scipy.stats import mode
import scipy.ndimage

from IPython.display import display, Markdown

import scipy.io

from skimage import transform as tf

from scipy.interpolate import interpn

np.set_printoptions(threshold=sys.maxsize)


SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def make_set(groups: list = ['P1', 'P2', 'P4', 'P3', 'P6'], num_exemplars: int = 5, wp_size_dva: float = 30.0,
             wp_size_pix: int = 512, sizing: str = 'lattice', ratio: float = 0.05, filter_freqs: list = [], save_fmt: str = "png", save_raw: bool = False,
             phase_scramble: int = 1, ps_scramble: bool = False, same_magnitude: bool = False,
             is_diagnostic: bool = False, save_path: str = "", mask: bool = True):

    if ps_scramble and wp_size_pix % (4*2**5) != 0: # adjust size of wallpaper if size is no integer multiple of (4*2**5)
        new_wp_size_pix = (wp_size_pix // (4*2**5) + 1) * 4*2**5
        LOGGER.warning('wp_size_pix {} is not an integer multiple of {} and will be increase to {}'.format(wp_size_pix, 4*2**5, new_wp_size_pix))
        wp_size_pix = new_wp_size_pix
    
    # make groups all uppercase
    groups = [group.upper() for group in groups] 
    
    # make sizing input lowercase
    sizing = sizing.lower()
    
    # check if the ratio is valid
    is_valid_ratio(groups, sizing, ratio)

    # save parameters
    if not save_path:
        save_path = os.path.join(os.path.expanduser('~'), 'wallpapers')
    today = datetime.today()
    time_str = today.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # define group to index mapping

    key_set = ['P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG',
                   'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M']
    value_set = np.arange(101, 101 + len(key_set), 1)
    map_group = {}
    for i in range(value_set.shape[0]):
        map_group[key_set[i]] = value_set[i]
    Groups = groups
    raw_path = ''
    if (not filter_freqs):
        filter_freq_str = "No filtering applied"
    else:
        filter_freq_str = ','.join(str(x) for x in filter_freqs)
    try:
        if(save_raw):
            raw_path = os.path.join(save_path, "raw")
            os.makedirs(raw_path)
    except:
        print('Save Path: ', save_path, "\nGroups to Generate: ", groups,
              "\nNumber of Wallpapers per Group: ", num_exemplars, "\nFiltering Level: ", filter_freq_str)
    print('Save Path: ', save_path, '\nGroups to Generate: ', groups,
          '\nNumber of Wallpapers per Group: ', num_exemplars, '\nFiltering Level: ', filter_freq_str, '\n')

    # TODO: add configuration to argument parser
    fundamental_region_source_type = 'uniform_noise'
    
    # Generating WPs and scrambling
    cmap = 'gray'
    cm = plt.get_cmap(cmap)
    pdf_path = save_path + '/diagnostic_all.' + 'pdf'
    pdf = PdfPages(pdf_path)
    
    for group_idx, group in enumerate(Groups):
        # making regular images
        print('generating ', group)
        if sizing == 'lattice' or sizing == 'fr':
            area = np.round(wp_size_pix**2 * ratio)
            if sizing == 'fr':
                print("Area of fundamnetal region: " + str(area))
            else:
                print("Area of lattice region: " + str(area))
        else:
            raise SystemExit('Invalid sizing option. Must be either FR or lattice')
        
        if filter_freqs:
            this_groups_wallpapers = [[] for i in filter_freqs]
        else:
            this_groups_wallpapers = [[]]

        for k in range(num_exemplars):
            print('filter_freqs {}'.format(filter_freqs))
            
            raw = make_single(group, wp_size_pix, int(area), sizing, ratio, wp_size_dva, is_diagnostic,
                              filter_freqs, fundamental_region_source_type, cmap, save_path, k, pdf)
            raw = np.array(raw).squeeze()

            if raw.ndim==2:
                raw = np.expand_dims(raw,0)

            group_number = map_group[group]
            # save raw wallpapers
            if(save_raw):
                raw_path_tmp = raw_path + '/' + time_str + '_' + str(1000 * group_number + k + 1) + \
                    '_' + cmap + '.' + save_fmt
                display(Markdown(str(1000 * group_number + k + 1) + '_' + cmap + '_raw'))
                for s in range(len(raw)):
                    clipped_raw = clip_wallpaper(np.array(raw[s]), wp_size_pix)
                    display(Image.fromarray(
                        (clipped_raw[:, :] * 255).clip(0,255).astype(np.uint8)))
                    Image.fromarray(
                        (clipped_raw[:, :] * 255).clip(0,255).astype(np.uint8)).save(raw_path_tmp, save_fmt)

            for i, this_raw in enumerate(raw):
                this_groups_wallpapers[i].append(this_raw)

        this_groups_wallpapers = np.array(this_groups_wallpapers)

        if same_magnitude:
            # normalize psd across exemplars per group and filter
            this_groups_wallpapers_spec         = np.fft.rfft2(this_groups_wallpapers)
            this_groups_wallpapers_mag_mean     = np.abs(this_groups_wallpapers_spec).mean(-3)
            this_groups_wallpapers_spec_phase   = np.angle(this_groups_wallpapers_spec)
            this_groups_wallpapers_spec         = np.expand_dims(this_groups_wallpapers_mag_mean,-3) \
                                                  * np.exp(1j * this_groups_wallpapers_spec_phase)
            this_groups_wallpapers              = np.fft.irfft2(this_groups_wallpapers_spec)
        else:
            this_groups_wallpapers_mag_mean = max(1,len(filter_freqs))*[[]]

        # generate scrambled controls
        if (phase_scramble > 0 or ps_scramble):
            this_groups_controls = np.zeros((phase_scramble+int(ps_scramble),)+this_groups_wallpapers.shape)
            for i in range(this_groups_wallpapers.shape[0]): # over f0fr
                for j in range(this_groups_wallpapers.shape[1]): # over exemplars
                    # don't believe we need the option use_magnitude here, since the magnitude of the wallpaper is unchanged in the case of ctrl_images=='phase'
                    if phase_scramble >0:
                        this_control = replace_spectra(this_groups_wallpapers[i,j], phase_scramble, False,
                                                       this_groups_wallpapers_mag_mean[i],  cmap=cmap)
                        this_groups_controls[:phase_scramble,i,j] = np.transpose(this_control,(2,0,1)) # this needs to be addapted for ps_scramble!=1
                    if ps_scramble:
                        this_control = replace_spectra(clip_wallpaper(this_groups_wallpapers[i,j],wp_size_pix), 0, True,
                                                       this_groups_wallpapers_mag_mean[i],  cmap=cmap)
                        x_offset = round((this_groups_controls.shape[-2] - wp_size_pix) / 2)
                        y_offset = round((this_groups_controls.shape[-1] - wp_size_pix) / 2)
                        this_groups_controls[phase_scramble-1+ps_scramble,i,j,x_offset:x_offset+wp_size_pix,y_offset:y_offset+wp_size_pix] = np.transpose(this_control,(2,0,1)) # this needs to be addapted for ps_scramble!=1
        else:
            this_groups_controls = None

        # crop wallpapers and controls
        w_idxs = np.arange(wp_size_pix) + int( (this_groups_wallpapers.shape[-2]-wp_size_pix)//2)
        h_idxs = np.arange(wp_size_pix) + int( (this_groups_wallpapers.shape[-1]-wp_size_pix)//2)
        this_groups_wallpapers  = this_groups_wallpapers[...,w_idxs,:][...,h_idxs]
        if (phase_scramble > 0 or ps_scramble):
            this_groups_controls = this_groups_controls[..., w_idxs, :][..., h_idxs]
        
        for exemplar_idx in range(this_groups_wallpapers.shape[1]):
            for filter_idx in range(this_groups_wallpapers.shape[0]):
                radial_avg_pre = calc_radially_averaged_psd(this_groups_wallpapers[filter_idx,exemplar_idx])
                
                # smoothing and filtering for wallpapers
                this_groups_wallpapers[filter_idx,exemplar_idx] = filter_img(this_groups_wallpapers[filter_idx,exemplar_idx], wp_size_pix)
                
                if is_diagnostic:
                    # plotting of radial averages
                    group_wp = str(map_group[group]) + str(exemplar_idx).zfill(3)
                    fig, (ax1) = plt.subplots(1, figsize=(10, 10))
                    bins = np.linspace(0, 1, 100)
                    radial_avg_post = calc_radially_averaged_psd(this_groups_wallpapers[filter_idx,exemplar_idx])
                    ax1.hist(radial_avg_pre[radial_avg_pre != 0], bins, color=[1, 0, 1])
                    ax1.hist(radial_avg_post[radial_avg_post != 0], bins, color=[0, 1, 1])
                    ax1.set_title('Radial Averages for pre-smoothing and post-smoothing of wallpapers for ' + group_wp)
                    labels= ["pre-smoothing","post-smoothing"]
                    ax1.legend(labels)
                    bbox = ax1.get_tightbbox(fig.canvas.get_renderer())
                    ax1_path = save_path + '/' + group_wp + '_diagnostic_4' + '.' + "png"
                    fig.savefig(ax1_path,bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))
                    pdf.savefig(fig)
                    plt.show()
                
        for exemplar_idx in range(this_groups_wallpapers.shape[1]):
            for filter_idx in range(this_groups_wallpapers.shape[0]):
                for this_phase_step in range(phase_scramble):
                    this_groups_controls[this_phase_step,filter_idx,exemplar_idx] = filter_img(this_groups_controls[this_phase_step,filter_idx,exemplar_idx], wp_size_pix)
        # normalize range of pixel values
#        this_groups_wallpapers = this_groups_wallpapers - np.expand_dims(np.expand_dims(this_groups_wallpapers.min((-1,-2)),-1),-1)
#        this_groups_wallpapers = this_groups_wallpapers / np.expand_dims(np.expand_dims(this_groups_wallpapers.max((-1, -2)) - this_groups_wallpapers.min((-1, -2)), -1), -1)
#        this_groups_wallpapers = this_groups_wallpapers - np.expand_dims(np.expand_dims(this_groups_wallpapers.mean((-1,-2)),-1),-1)
#        this_groups_wallpapers = this_groups_wallpapers /(9* np.expand_dims(np.expand_dims(this_groups_wallpapers.std(axis=(-1, -2)),-1),-1))+0.5

        if this_groups_wallpapers.min()<0:
            LOGGER.warning('need to clip. wallpapers have sample values < 0: {}'.format(this_groups_wallpapers.min()))
        if this_groups_wallpapers.max()>1:
            LOGGER.warning('need to clip. wallpapers have sample values > 1: {}'.format(this_groups_wallpapers.max()))

        # mask images
        this_groups_wallpapers = mask_imgs(this_groups_wallpapers)
        # TODO: normalization not consistent with e.g. in minPhaseInterp
        if this_groups_controls is not None: # same for controls
            # normalize range of pixel values to 0...1
#            this_groups_controls = this_groups_controls - np.expand_dims(np.expand_dims(this_groups_controls.min((-1, -2)), -1), -1)
#            this_groups_controls = this_groups_controls / np.expand_dims(np.expand_dims(this_groups_controls.max((-1, -2)) - this_groups_controls.min((-1, -2)), -1), -1)
#            this_groups_controls = this_groups_controls - np.expand_dims(np.expand_dims(this_groups_controls.mean((-1, -2)), -1), -1)
#            this_groups_controls = this_groups_controls / (9*np.expand_dims(np.expand_dims(this_groups_controls.std((-1, -2)), -1), -1)) + 0.5

            if this_groups_controls.min() < 0:
                LOGGER.warning('need to clip. wallpapers have sample values < 0: {}'.format(this_groups_controls.min()))
            if this_groups_controls.max() > 1:
                LOGGER.warning('need to clip. wallpapers have sample values > 1: {}'.format(this_groups_controls.max()))

            # mask images
            this_groups_controls = mask_imgs(this_groups_controls)

        # save images
        for exemplar_idx in range(this_groups_wallpapers.shape[1]):
            for filter_idx in range(this_groups_wallpapers.shape[0]):
                group_wp = str(map_group[group]) + str(exemplar_idx).zfill(3)
                wp_filename =  '{save_path}/{group}_{cmap}_{filter_info}{ratio}.{save_fmt}'.format( save_path=save_path,
                                                                                                                      group=group_wp,
                                                                                                                      cmap=cmap,
                                                                                                                      filter_info='f0fr_{}cpd_'.format(filter_freqs[filter_idx]) if filter_freqs else '',
                                                                                                                      ratio=ratio,
                                                                                                                      save_fmt=save_fmt)
                Image.fromarray((this_groups_wallpapers[filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)).save(wp_filename, save_fmt)
                display(Markdown(str(1000 * group_number + exemplar_idx + 1) + '_' + cmap))
                display(Image.fromarray((this_groups_wallpapers[filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)))
                if this_groups_controls is not None:  # same for controls
                    for this_phase_step in range(phase_scramble):
                        if this_phase_step+1 == phase_scramble:
                            group_ctrl = str(map_group[group]) + str(exemplar_idx).zfill(3)
                        else:
                            group_ctrl = str(map_group[group] + 17) + str(exemplar_idx).zfill(3)
                        ctrl_filename =  '{save_path}/{group}_{cmap}_{filter_info}{ratio}_{this_phase_step}of{phase_scramble}.{save_fmt}'.format( save_path=save_path,
                                                                                                                      group=group_ctrl,
                                                                                                                      cmap=cmap,
                                                                                                                      filter_info='f0fr_{}cpd_'.format(filter_freqs[filter_idx]) if filter_freqs else '',
                                                                                                                      ratio=ratio,
                                                                                                                      this_phase_step = this_phase_step+1,
                                                                                                                      phase_scramble = phase_scramble,
                                                                                                                      save_fmt=save_fmt)
                        Image.fromarray((this_groups_controls[this_phase_step,filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)).save(ctrl_filename, save_fmt)
                        display(Markdown(str(1000 * group_number + exemplar_idx + 1) + '_' + str(this_phase_step + 1) + '_' + cmap + '_phase'))
                        display(Image.fromarray((this_groups_controls[this_phase_step,filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)))
                    if ps_scramble:
                        group_ctrl = str(map_group[group] + 34) + str(exemplar_idx).zfill(3)
                        ctrl_filename =  '{save_path}/CTRL_{group}_{cmap}_{filter_info}{ratio}_psscramble.{save_fmt}'.format( save_path=save_path,
                                                                                                                                                                   group=group_ctrl,
                                                                                                                                                                   cmap=cmap,
                                                                                                                                                                   filter_info='f0fr_{}cpd_'.format(filter_freqs[filter_idx]) if filter_freqs else '',
                                                                                                                                                                   ratio=ratio,
                                                                                                                                                                   save_fmt=save_fmt)
                        Image.fromarray((this_groups_controls[phase_scramble,filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)).save(ctrl_filename, save_fmt)
                        display(Markdown(str(1000 * group_number + exemplar_idx + 1) + '_' + cmap + '_ps'))
                        display(Image.fromarray((this_groups_controls[phase_scramble,filter_idx,exemplar_idx] * 255).clip(0,255).astype(np.uint8)))

    pdf.close()  


def new_p3(tile):
    # Generate p3 wallpaper

    # For mag_factor, use a multiple of 3 to avoid the rounding error
    # when stacking two_tirds, one_third tiles together

    mag_factor = 6
    tile1 = tf.rescale(tile, mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    height = np.size(tile1, 0)

    # fundamental region is equlateral rhombus with side length = s

    s1 = round((height / 3))
    s = 2 * s1

    # NOTE on 'ugly' way of calculating the widt = h
    # after magnification width(tile1) > tan(pi/6)*height(tile1) (should be equal)
    # and after 240 deg rotation width(tile240) < 2*width(tile1) (should be
    # bigger or equal, because we cat them together).
    # subtract one, to avoid screwing by imrotate(240)
    
    width = round(height / math.sqrt(3)) - 1
    # define rhombus-shaped mask
    xy = np.array(
        [[0, 0], [s1, width], [height, width], [2 * s1, 0], [0, 0]])

    mask = skd.polygon2mask((height, width), xy)
    tile0 = mask * tile1[:, :width]
    
    # rotate rectangle by 120, 240 degs

    # note on 120deg rotation: 120 deg rotation of rhombus-shaped
    # texture preserves the size, so 'crop' option can be used to
    # tile size.

    tile120 = tf.rotate(tile0, 120, resize=False, order=1)
    tile240 = tf.rotate(tile0, 240, resize=True, order=1)

    # manually trim the tiles:

    # tile120 should have the same size as rectangle: [heigh x width]

    # tile240 should have the size [s x 2*width]
    # find how much we need to cut from both sides
    diff = round(0.5 * (np.size(tile240, 1) - 2 * width))
    row_start = round(0.25 * s)
    row_end = round(0.25 * s) + s
    col_start = diff
    col_end = 2 * width + diff
    tile240 = tile240[row_start:row_end, col_start:col_end]

    # Start to pad tiles and glue them together
    # Resulting tile will have the size [3*height x 2* width]

    two_thirds1 = np.concatenate((tile0, tile120), axis=1)
    two_thirds2 = np.concatenate((tile120, tile0), axis=1)

    two_thirds = np.concatenate((two_thirds1, two_thirds2))

    # lower half of tile240 on the top, zero-padded to [height x 2 width]
    row_start = int(0.5 * s)
    col_end = 2 * width
    one_third11 = np.concatenate(
        (tile240[row_start:, :], np.zeros((s, col_end))))

    # upper half of tile240 on the bottom, zero-padded to [height x 2 width]
    row_end = int(0.5 * s)
    one_third12 = np.concatenate(
        (np.zeros((s, col_end)), tile240[:row_end, :]))

    # right half of tile240 in the middle, zero-padded to [height x 2 width]
    col_start = width
    one_third21 = np.concatenate(
        (np.zeros((s, width)), tile240[:, col_start:], np.zeros((s, width))))

    # left half of tile240in the middle, zero-padded to [height x 2 width]
    one_third22 = np.concatenate(
        (np.zeros((s, width)), tile240[:, :width], np.zeros((s, width))))

    # cat them together
    one_third1 = np.concatenate((one_third11, one_third12))
    one_third2 = np.concatenate((one_third21, one_third22), axis=1)

    # glue everything together, shrink and replicate
    one_third = np.maximum(one_third1, one_third2)

    # size(whole) = [3xheight 2xwidth]
    whole = np.maximum(two_thirds, one_third)
    whole = tf.rotate(whole, 90, resize=True, order=1)
    p3 = tf.rescale(whole, 1 / mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    return p3


def new_p3m1(tile):
    # Generate p3m1 wallpaper
    mag_factor = 10
    tile1 = tf.rescale(tile, mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    height = np.shape(tile1)[0]

    # fundamental region is equlateral triangle with side length = height
    width = round(0.5 * height * math.sqrt(3))

    y1 = round(height / 2)

    # vetrices of the triangle (closed polygon => four points)
    mask_xy = [[0, 0], [y1, width], [y1, 0], [0, 0]]

    # Create half of the mask
    # reflect and concatenate, to get the full mask:

    mask_half = skd.polygon2mask((y1, width), mask_xy)
    mask = np.concatenate((mask_half, np.flipud(mask_half)))

    # equilateral triangle inscribed into rectangle
    tile0 = tile1[:, :width] * mask

    # continue to modify the tile

    # reflect and rotate
    tile1_mirror = np.fliplr(tile0)
    tile240 = tf.rotate(tile1_mirror, 240, resize=True, order=1)
    # AY: I directly cut the tiles, because trim will
    # return slightly different size

    t_r1x = np.shape(tile240)[0]
    start_row = t_r1x - height
    tile240 = tile240[start_row:, :width]

    # AY: rotating mirrored tile(as opposed to tileR1) will cause less
    # border effects when we'll add it to two other tiles.
    tile120 = tf.rotate(tile1_mirror, 120, resize=True, order=1)
    tile120 = tile120[:height, :width]
    # Assembling the tiles

    # We have 3 tiles with the same triangle rotated 0, 120 and 240
    # pad them and put them together
    zero_tile = np.zeros((y1, width))
    tile2 = np.concatenate((zero_tile, tile0, zero_tile))
    tile240 = np.concatenate((zero_tile, zero_tile, tile240))
    tile120 = np.concatenate((tile120, zero_tile, zero_tile))

    # Using max() will give us smoother edges, as opppose to sum()
    half1 = np.maximum(tile2, tile240)
    half = np.maximum(half1, tile120)

    # By construction, size(whole) = [4*y1 2*x1]
    whole = np.concatenate((half, np.fliplr(half)), axis=1)
    # Shifting by 2 pix (as oppose to zero), we'll smoothly glue tile together.
    # Set delta_pix value to zero and see the difference
    delta_pix = 2
    start_row1 = 3 * y1 - delta_pix
    start_row2 = 3 * y1
    end_row1 = 4 * y1 - delta_pix
    end_row2 = y1 + delta_pix
    end_row3 = 4 * y1
    end_col1 = 2 * width

    top_bit = np.concatenate(
        (whole[start_row1:end_row1, width:end_col1], whole[start_row1:end_row1, :width]), axis=1)
    bot_bit = np.concatenate(
        (whole[delta_pix:end_row2, width:end_col1], whole[delta_pix:end_row2, :width]), axis=1)

    whole[:y1, :] = np.maximum(whole[delta_pix:end_row2, :], top_bit)
    whole[start_row2:end_row3, :] = np.maximum(
        whole[start_row1:end_row1, :], bot_bit)
    
    # cutting middle piece of tile
    mid_tile = whole[y1:start_row2, :width]
    
    # reflecting middle piece and glueing both pieces to the bottom
    # size(big_tile)  = [6*y1 2*x1]
    cat_mid_flip = np.concatenate((np.fliplr(mid_tile), mid_tile), axis=1)
    big_tile = np.concatenate((whole, cat_mid_flip))
    #big_tile = tf.rotate(big_tile, 90, resize=True, order=1)
    #display(Image.fromarray((cat_mid_flip * 255).clip(0,255).astype(np.uint8)))
    p3m1 = tf.rescale(big_tile, 1 / mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    return p3m1


def new_p31m(tile):
    # Generate p31m wallpaper
    mag_factor = 6
    tile0 = tf.rescale(tile, mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    height = np.shape(tile0)[0]
    width = round(0.5 * height / math.sqrt(3))
    y1 = round(height / 2)

    # fundamental region is an isosceles triangle with angles(30, 120, 30)

    # vetrices of the triangle (closed polygon => four points)
    mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]]

    # make half of the mask
    # reflect and concatenate, to get the full mask:
    mask_half = skd.polygon2mask((y1, width), mask_xy)

    mask = np.concatenate((mask_half, np.flipud(mask_half)))

    # size(tile0) = [height  width]
    tile0 = mask * tile0[:, :width]

    # rotate the tile
    tile120 = tf.rotate(tile0, 120, resize=True, order=1)
    tile240 = tf.rotate(tile0, 240, resize=True, order=1)

    # trim the tiles manually, using trigonometric laws
    # NOTE: floor and round give us values that differ by 1 pix.
    # to trim right, we'll have to derive the trim value from
    tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1)
    delta = np.shape(tile0)[1]

    # ideally we would've used
    # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);
    x120 = np.shape(tile120)[1] - delta
    y120 = np.shape(tile120)[0] - y1

    # size(tile120, tile240) = [height 3width]

    tile120 = tile120[y120:, x120:]
    tile240 = tile240[:y1, x120:]

    # we have 3 tiles that will comprise
    # equilateral triangle together
    # glue them together by padding and smoothing edges (max instead of sum)
    # tile1 already padded
    tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120))

    tile240 = np.concatenate((tile240, np.zeros((y1, width * 3))))

    # size(tri) = [height 3width]
    tri = np.maximum(np.maximum(tile0, tile120), tile240)
    mirror_tri = np.fliplr(tri)

    # use shift overlap, to smooth the edges
    delta_pix = 3
    row_start = y1 - delta_pix
    row_end1 = mirror_tri.shape[0] - delta_pix
    row_end2 = y1 + delta_pix
    shifted = np.concatenate(
        (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]))
    tile2 = np.maximum(shifted, tri)

    # size(tile3) = [height 6width]
    tile3 = np.concatenate((tile2, np.fliplr(tile2)), axis=1)
    p31m = tf.rescale(tile3, 1 / mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    return p31m


def new_p6(tile):
    # Generate p6 wallpaper
    mag_factor = 6
    tile1 = tf.rescale(tile, mag_factor, order=3, mode='symmetric', anti_aliasing=True)

    height = np.shape(tile1)[0]
    width = int(round(0.5 * height * np.tan(np.pi / 6)))
    y1 = round(height / 2)

    # fundamental region is an isosceles triangle with angles(30, 120, 30)

    # vetrices of the triangle (closed polygon => four points)
    mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]]

    # half of the mask
    # reflect and concatenate, to get the full mask:
    mask_half = skd.polygon2mask((y1, width), mask_xy)

    mask = np.concatenate((mask_half, np.flipud(mask_half)))

    # size(tile0) = [height x width]
    tile0 = mask * tile1[:, :width]

    # rotate tile1
    tile120 = tf.rotate(tile0, 120, resize=True, order=1)
    tile240 = tf.rotate(tile0, 240, resize=True, order=1)
    # trim the tiles manually, using trigonometric laws
    # NOTE: floor and round give us values that differ by 1 pix.
    # to trim right, we'll have to derive the trim value from
    tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1)
    delta = np.shape(tile0)[1]

    # ideally we would've used
    # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2
    x120 = np.shape(tile120)[1] - delta
    y120 = np.shape(tile120)[0] - y1

    # size(tile120, 240) = [y1 x 3x1]
    tile120 = tile120[y120:, x120:]
    tile240 = tile240[:y1, x120:]

    # we have 3 tiles that will comprise
    # equilateral triangle together

    # glue them together by padding and smoothing edges (max instead of sum)
    # tile0 already padded
    tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120))
    tile240 = np.concatenate((tile240, np.zeros((y1, width * 3))))

    # size(tri) = [2y1 x 3x1]
    tri = np.maximum(np.maximum(tile0, tile120), tile240)

    # mirror_tri = fliplr(tri); --wrong! should be (fliplr(flipud(tri)))
    mirror_tri = tf.rotate(tri, 180, resize=True, order=1)

    # shifw w.slight overlap,
    delta_pix = 3
    row_start = y1 - delta_pix
    row_end1 = mirror_tri.shape[0] - delta_pix
    row_end2 = y1 + delta_pix
    shifted = np.concatenate(
        (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]))

    tile2 = np.maximum(tri, shifted)
    t2 = int(np.floor(0.5 * np.shape(tile2)[0]))

    tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :]))

    # size(tile3) = [2y1 x 6x1]
    tile3 = np.concatenate((tile2, tile2_flipped), axis=1)
    p6 = tf.rescale(tile3, 1 / mag_factor, order=3, mode='symmetric', anti_aliasing=True)
    return p6


def new_p6m(tile):
    # Generate p6m wallpaper

    mag_factor = 30
    tile1 = tf.rescale(tile, mag_factor, order=1, mode='symmetric', anti_aliasing=False, preserve_range=True)

    height = np.shape(tile1)[0]
    
    width = round(height / math.sqrt(3))

    # fundamental region is right triangle with angles (30, 60)

    # vetrices of the triangle (closed polygon => four points)
    mask_xy = [[0, 0], [height, width], [height, 0], [0, 0]]

    # half of the mask
    # reflect and concatenate, to get the full mask:
    mask = skd.polygon2mask((height, width), mask_xy)

    # right triangle inscribed into rectangle
    tile0 = tile1[:, :width] * mask

    # size(tile0) = [height x width]
    tile0 = np.concatenate((tile0, np.flipud(tile0)))

    # rotate tile1
    tile120 = tf.rotate(tile0, 120, resize=True, order=1)
    tile240 = tf.rotate(tile0, 240, resize=True, order=1)

    # trim the tiles manually, using trigonometric laws
    # NOTE: floor and round give us values that differ by 1 pix.
    # to trim right, we'll have to derive the trim value from
    tile0 = np.concatenate(
        (tile0, np.zeros((height * 2, width * 2))), axis=1)
    delta = np.shape(tile0)[1]

    # ideally we would've used
    # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);
    x120 = np.shape(tile120)[1] - delta
    y120 = np.shape(tile120)[0] - height

    # size(tile120, 240) = [y1 x 3x1]
    tile120 = tile120[y120:, x120:]
    tile240 = tile240[:height, x120:]

    # we have 3 tiles that will comprise
    # equilateral triangle together

    # glue them together by padding and smoothing edges (max instead of sum)
    # tile0 already padded
    tile120 = np.concatenate((np.zeros((height, width * 3)), tile120))
    tile240 = np.concatenate((tile240, np.zeros((height, width * 3))))

    # size(tri) = [2y1 x 3x1]
    tri = np.maximum(np.maximum(tile0, tile120), tile240)
    mirror_tri = tf.rotate(tri, 180, resize=True, order=1)

    # shifw w.slight overlap,
    delta_pix = 3
    row_start = height - delta_pix
    row_end1 = mirror_tri.shape[0] - delta_pix
    row_end2 = height + delta_pix
    shifted = np.concatenate(
        (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]))

    tile2 = np.maximum(tri, shifted)
    t2 = int(np.floor(0.5 * np.shape(tile2)[0]))

    tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :]))
    # size(tile3) = [2y1 x 6x1]
    tile3 = np.concatenate((tile2, tile2_flipped), axis=1)
    p6m = tf.rescale(tile3, 1 / mag_factor, order=1, mode='symmetric', anti_aliasing=False, preserve_range=True)
    return p6m

def filter_tile(in_tile, filter_intensity):
    # generate random noise tile

    mu = 0.5;
    nx = np.size(in_tile, 0);
    ny = np.size(in_tile, 1);
    
    # make adaptive filtering
    
    sigma_x = 10*filter_intensity/nx;
    sigma_y = 10*filter_intensity/ny;
    
    x = np.linspace(0, 1, nx);
    y = np.linspace(0, 1, ny);
    
    gx = np.exp(-1 * (np.power((x - mu), 2)) / (2*(sigma_x**2))) / (sigma_x * math.sqrt(2 * math.pi));
    gy = np.exp(-1 * (np.power((y - mu), 2)) / (2*(sigma_y**2))) / (sigma_y * math.sqrt(2 * math.pi));

    gauss2 = np.matmul(gx.reshape(gx.shape[0], 1),gy.reshape(1, gy.shape[0]));
    gauss2 = gauss2 - gauss2.min();
    gauss2 = gauss2 / gauss2.max();
    gauss2 = gauss2 * 5;
    
    filtered = np.abs(np.fft.ifft2(np.fft.fft2(in_tile) * gauss2));
    
    # normalize tile

    outTile = filtered - filtered.min();
    outTile = outTile / outTile.max();
    return outTile;


def make_single(wp_type, N, n, sizing, ratio, angle, is_diagnostic, filter_freq,
                fundamental_region_source_type, cmap, save_path, k, pdf, opt_texture=None):
    #  make_single(type,N,n,opt_texture)
    # generates single wallaper group image
    # wp_type defines the wallpaper group
    # N is the size of the output image. For complex groups
    #   returned image will have size at least NxN.
    # n is the area of repeating pattern (lattice for lattice sizing and fundamental region for fr sizing) 
    #   for all groups in pixels.
    # is_diagnostic whether to generate diagnostic images (outlining fundamental region and lattice)
    # isSpatFreqFilt generate a spatial frequency filtered wallpaper
    # fwhm full width at half maximum of spatial frequency filter
    # whether spatialfrequency filter is lowpass or highpass

    # default
    # save paths for debugging

    if fundamental_region_source_type == 'uniform_noise':
        print('uniform noise')
        if n>(N**2):
            print('size of repeating pattern larger than size of wallpaper')
        #raw_texture = np.random.rand(max(n,N), max(n,N));
        raw_texture = np.random.rand(max(N,N), max(N,N));
    elif isinstance(fundamental_region_source_type, np.ndarray):
        print('texture was passed explicitly')
        opt_texture = fundamental_region_source_type
        min_dim = np.min(np.shape(opt_texture))
        # stretch user-defined texture, if it is too small for sampling
        if min_dim < n:
            ratio_texture = round(n / min_dim)
            opt_texture = np.array(Image.resize(
                reversed((opt_texture.shape * ratio_texture)), Image.NEAREST))
        raw_texture = opt_texture
    else:
        raise Exception('this source type ({})is not implemented'.format(
            type(fundamental_region_source_type)))
    # do filtering
    image = []
    if filter_freq:
        num_wallpapers = len(filter_freq)
    else:
        num_wallpapers = 1
    for i in range(num_wallpapers):
        if filter_freq:
            if n > (N**2):
                LOGGER.warning('Area of repeating region greater than area of wallpaper.')
            fundamental_region_filter = filter.Cosine_filter(filter_freq[i], N, angle )
        else:
            fundamental_region_filter = None
        if fundamental_region_filter:
            if isinstance(fundamental_region_filter, filter.Cosine_filter):
                texture = fundamental_region_filter.filter_image(raw_texture)
                # scale texture into range 0...1
                texture = (texture - texture.min()) / (texture.max() - texture.min())
            else:
                raise Exception('this filter type ({}) is not implemented'.format(
                    type(fundamental_region_filter)))
        else:
            texture = raw_texture
            texture = (texture-texture.min())/(texture.max() - texture.min())
        # else:
           # TODO: not exactly sure, what this lowpass filter is supposed to do. in any case:
           #       it should be adapted to this structure that separates the noise generation from the filtering
        
        # Not sure why we need to clip texture?
        
        #if N>n: # we need to crop from the WP-sized texture
        #  texture = texture[int(n//2):int(n//2)+n,int(n//2):int(n//2)+n]
        try:
            # generate the wallpapers
            if wp_type == 'P0':
                p0 = np.array(Image.resize(
                    reversed((texture.shape * round(N / n))), Image.NEAREST))
                image.append(p0)
            elif wp_type == 'P1':
                # Square fundemental region 
                # => side length =  sqrt(area of FR)
                side_length = int(np.round(np.sqrt(n)))
                width = side_length
                height = side_length
                p1 = texture[:height, :width]
                p1_image = cat_tiles(p1, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p1_image, wp_type, p1, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p1_image)
            elif wp_type == 'P2':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 2 
                # => height = sqrt(area of FR / 2) and width = sqrt(area of FR * 2)
                if (sizing == 'lattice'):
                    side_length = n / 2 # FR should be half the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length / 2)))
                width = int(np.round(np.sqrt(side_length * 2)))
                start_tile = texture[:height, :width]
                tileR180 = np.rot90(start_tile, 2)
                p2 = np.concatenate((start_tile, tileR180))
                p2_image = cat_tiles(p2, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p2_image, wp_type, p2, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p2_image)
            elif wp_type == 'PM':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 2 
                # => height = sqrt(area of FR / 2) and width = sqrt(area of FR * 2)
                if (sizing == 'lattice'):
                    side_length = n / 2 # FR should be half the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length / 2)))
                width = int(np.round(np.sqrt(side_length * 2)))
                start_tile = texture[:height, :width]
                mirror = np.flipud(start_tile)
                pm = np.concatenate((start_tile, mirror))
                pm_image = cat_tiles(pm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pm_image, wp_type, pm, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(pm_image)
            elif wp_type == 'PG':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 2 
                # => height = sqrt(area of FR / 2) and width = sqrt(area of FR * 2)
                if (sizing == 'lattice'):
                    side_length = n / 2 # FR should be half the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length / 2)))
                width = int(np.round(np.sqrt(side_length * 2)))
                start_tile = texture[:height, :width]
                tile = np.rot90(start_tile, 3)
                glide = np.flipud(tile)
                pg = np.concatenate((tile, glide), axis=1)
                pg_image = cat_tiles(pg.T, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pg_image, wp_type, pg, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(pg_image)
            elif wp_type == 'CM':
                # Triangular fundamental region 
                # => area of FR = area of lattice / 2 
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 2 # FR should be half the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length)))
                width = int(np.round(np.sqrt(side_length)))
                start_tile = texture[:height, :width]
                mirror = np.fliplr(start_tile)
                tile1 = np.concatenate((start_tile, mirror), axis=1)
                tile2 = np.concatenate((mirror, start_tile), axis=1)
                cm = np.concatenate((tile1, tile2))
                cm_image = cat_tiles(cm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(cm_image, wp_type, cm, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(cm_image)
            elif wp_type == 'PMM':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 4 
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 4 # FR should be 1 / 4 the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length)))
                width = int(np.round(np.sqrt(side_length)))
                start_tile = texture[:height, :width]
                mirror = np.fliplr(start_tile)
                concat_tmp1 = np.concatenate((start_tile, mirror), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.flipud(start_tile), np.flipud(mirror)), axis=1)
                pmm = np.concatenate((concat_tmp1, concat_tmp2))
                pmm_image = cat_tiles(pmm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pmm_image, wp_type, pmm, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(pmm_image)
            elif wp_type == 'PMG':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 4 
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 4 # FR should be 1 / 4 the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length)))
                width = int(np.round(np.sqrt(side_length)))
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                concat_tmp1 = np.concatenate(
                    (start_tile, start_tile_rot180), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.flipud(start_tile), np.fliplr(start_tile)), axis=1)
                pmg = np.concatenate((concat_tmp1, concat_tmp2))
                pmg_image = cat_tiles(pmg, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pmg_image, wp_type, pmg, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(pmg_image)
            elif wp_type == 'PGG':
                # Triangular fundamental region 
                # => area of FR = area of lattice / 4 
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 4 # FR should be half the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length)))
                width = int(np.round(np.sqrt(side_length)))
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                concat_tmp1 = np.concatenate(
                    (start_tile, np.flipud(start_tile)), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.fliplr(start_tile), start_tile_rot180), axis=1)
                pgg = np.concatenate((concat_tmp1, concat_tmp2))
                pgg_image = cat_tiles(pgg, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pgg_image, wp_type, pgg, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(pgg_image)
            elif wp_type == 'CMM':
                # Triangular fundamental region 
                # => area of FR = area of lattice / 4
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 2 # FR should be 1 / 4 the size if lattice sizing (fr is already halfed)
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length / 2))) #half the size of bottom half of lattice
                width = int(np.round(np.sqrt(side_length / 2))) #half the size of bottom half of lattice
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                tile1 = np.concatenate((start_tile, start_tile_rot180))
                tile2 = np.flipud(tile1)
                concat_tmp1 = np.concatenate((tile1, tile2), axis=1)
                concat_tmp2 = np.concatenate((tile2, tile1), axis=1)
                cmm = np.concatenate((concat_tmp1, concat_tmp2))
                cmm_image = cat_tiles(cmm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(cmm_image, wp_type, cmm, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(cmm_image)
            elif wp_type == 'P4':
                # Rectangular fundamental region 
                # => area of FR = area of lattice / 4 
                # => height = sqrt(area of FR) and width = sqrt(area of FR)
                if (sizing == 'lattice'):
                    side_length = n / 4 # FR should be 1 / 4 the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length)))
                width = int(np.round(np.sqrt(side_length)))
                start_tile = texture[:height, :width]
                start_tile_rot90 = np.rot90(start_tile, 1)
                start_tile_rot180 = np.rot90(start_tile, 2)
                start_tile_rot270 = np.rot90(start_tile, 3)
                concat_tmp1 = np.concatenate(
                    (start_tile, start_tile_rot270), axis=1)
                concat_tmp2 = np.concatenate(
                    (start_tile_rot90, start_tile_rot180), axis=1)
                p4 = np.concatenate((concat_tmp1, concat_tmp2))
                p4_image = cat_tiles(p4, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p4_image, wp_type, p4, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p4_image)
            elif wp_type == 'P4M':
                # Triangular fundamental region 
                # => area of FR = area of lattice / 8 
                # => height = sqrt(area of FR * sqrt(2)) and width = sqrt(area of FR * np.sqrt(2))
                if (sizing == 'lattice'):
                    side_length = n / 8 # FR should be 1 / 8 the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length) * np.sqrt(2)))
                width = int(np.round(np.sqrt(side_length) * np.sqrt(2)))
                start_tile = texture[:height, :width]
                xy = np.array([[0, 0], [width, height], [0, height], [0, 0]])
                mask = skd.polygon2mask((height, width), xy)
                tile1 = mask * start_tile
                tile2 = np.fliplr(tile1)
                tile2 = np.rot90(tile2, 1)
                tile = np.maximum(tile1, tile2)
                tile_rot90 = np.rot90(tile, 1)
                tile_rot180 = np.rot90(tile, 2)
                tile_rot270 = np.rot90(tile, 3)
                concat_tmp1 = np.concatenate((tile, tile_rot270), axis=1)
                concat_tmp2 = np.concatenate((tile_rot90, tile_rot180), axis=1)
                p4m = np.concatenate((concat_tmp1, concat_tmp2))
                p4m_image = cat_tiles(p4m, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p4m_image, wp_type, p4m, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p4m_image)
            elif wp_type == 'P4G':
                # Triangular fundamental region 
                # => area of FR = area of lattice / 8 
                # => height = sqrt(area of FR * sqrt(2)) and width = sqrt(area of FR * np.sqrt(2))
                if (sizing == 'lattice'):
                    side_length = n / 8 # FR should be 1 / 8 the size if lattice sizing
                    area = side_length
                else:
                    side_length = n
                height = int(np.round(np.sqrt(side_length) * np.sqrt(2)))
                width = int(np.round(np.sqrt(side_length) * np.sqrt(2)))
                start_tile = texture[:height, :width]
                xy = np.array([[0, 0], [width, 0], [width, height], [0, 0]])
                mask = skd.polygon2mask((height, width), xy)
                tile1 = mask * start_tile
                tile2 = np.fliplr(tile1)
                tile2 = np.rot90(tile2, 1)
                tile = np.maximum(tile1, tile2)
                tile_rot90 = np.rot90(tile, 1)
                tile_rot180 = np.rot90(tile, 2)
                tile_rot270 = np.rot90(tile, 3)
                concat_tmp1 = np.concatenate(
                    (tile_rot270, tile_rot180), axis=1)
                concat_tmp2 = np.concatenate((tile, tile_rot90), axis=1)
                p4g = np.concatenate((concat_tmp1, concat_tmp2))
                p4g_image = cat_tiles(p4g, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p4g_image, wp_type, p4g, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p4g_image)
            elif wp_type == 'P3':
                # Hexagonal fundamental region 
                # => area of FR = area of lattice / 3 
                # => height = round(np.sqrt(area_quarter_tile * (math.sqrt(3)))) and width = height / math.sqrt(3)) - 1
                # We have control over a quarter of the tile which contains the total area of 4.5 FRs
                # => area_quarter_tile = height * width (width = height / math.sqrt(3)) - 1)
                # => area_quarter_tile = height * round(height / math.sqrt(3)) - 1
                # => area_quarter_tile = height**2 / ((np.sqrt(3))- 1)
                # => height = round(np.sqrt(area_quarter_tile * (math.sqrt(3))))
                if (sizing == 'lattice'):
                    side_length = n / 3 # FR should be 1 / 3 the size if lattice sizing
                    area = side_length
                else:
                    area = n
                area_quarter_tile = area * 1.5
                height = round(np.sqrt(area_quarter_tile * (math.sqrt(3))))
                start_tile = texture[:height, :]
                p3 = new_p3(start_tile)
                p3_image = cat_tiles(p3, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p3_image, wp_type, p3, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                
                image.append(p3_image)
            elif wp_type == 'P3M1':
                # Hexagonal fundamental region 
                # => area of FR = area of lattice / 6 
                # => height = round(np.sqrt(area_sixth_tile / (0.5 * math.sqrt(3)))) and width = height * 0.5 * math.sqrt(3)
                # We have control over a sixth of the tile which contains the total area of 6 FRs
                # => area_sixth_tile = height * width (width = height * 0.5 * math.sqrt(3)))
                # => area_sixth_tile = height * round(height * 0.5 * math.sqrt(3))
                # => area_sixth_tile = height**2 * np.sqrt(3) * 0.5
                # => height = round(np.sqrt(area_sixth_tile / (0.5 * math.sqrt(3))))
                if (sizing == 'lattice'):
                    side_length = n / 6 # FR should be 1 / 6 the size if lattice sizing
                    area = side_length
                else:
                    area = n
                area_sixth_tile = area * 2 # we can control for a sixth of the size of a tile
                height = round(np.sqrt(area_sixth_tile / (0.5 * math.sqrt(3))))
                start_tile = texture[:height, :]
                p3m1 = new_p3m1(start_tile)
                p3m1_image = cat_tiles(p3m1, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p3m1_image, wp_type, p3m1, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p3m1_image)
            elif wp_type == 'P31M':
                # Hexagonal fundamental region 
                # => area of FR = area of lattice / 6 
                # => height = round(np.sqrt(area_sixth_tile * math.sqrt(3) / 0.5)) and width = height * 0.5 / math.sqrt(3)
                # We have control over a sixth of the tile which contains the total area of 6 FRs
                # => area_sixth_tile = height * width (width = height * 0.5 / math.sqrt(3)))
                # => area_sixth_tile = height * round(height * 0.5 / math.sqrt(3))
                # => area_sixth_tile = height**2 * 0.5 / np.sqrt(3)
                # => height = round(np.sqrt(area_sixth_tile * math.sqrt(3) / 0.5))
                if (sizing == 'lattice'):
                    side_length = n / 6 # FR should be 1 / 6 the size if lattice sizing
                    area = side_length
                else:
                    area = n
                area_sixth_tile = area * 2 # we can control for a sixth of the size of a tile
                height = round(np.sqrt(area_sixth_tile * math.sqrt(3) / 0.5))
                start_tile = texture[:height, :]
                p31m = new_p31m(start_tile)
                # ugly trick
                #p31m_1 = np.fliplr(np.transpose(p31m))
                p31m_image = cat_tiles(p31m, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p31m_image, wp_type, p31m, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p31m_image)
            elif wp_type == 'P6':
                # Hexagonal fundamental region 
                # => area of FR = area of lattice / 6 
                # => height = round(np.sqrt(area_sixth_tile / (np.tan(np.pi / 6) * 0.5))) and width = int(round(0.5 * height * np.tan(np.pi / 6)))
                # We have control over a sixth of the tile which contains the total area of 6 FRs
                # => area_sixth_tile = height * width (width = 0.5 * height * np.tan(np.pi / 6))
                # => area_sixth_tile = height * (0.5 * height * np.tan(np.pi / 6))
                # => area_sixth_tile = height**2 * 0.5 * np.tan(np.pi / 6)
                # => height = round(np.sqrt(area_sixth_tile / np.tan(np.pi / 6) * 0.5))
                if (sizing == 'lattice'):
                    side_length = n / 6 # FR should be 1 / 6 the size if lattice sizing
                    area = side_length
                else:
                    area = n
                area_sixth_tile = area * 2 # we can control for a sixth of the size of a tile
                height = round(np.sqrt(area_sixth_tile / (np.tan(np.pi / 6) * 0.5)))
                start_tile = texture[:height, :]
                p6 = new_p6(start_tile)
                p6_image = cat_tiles(p6, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p6_image, wp_type, p6, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p6_image)
            elif wp_type == 'P6M':
                # Hexagonal fundamental region 
                # => area of FR = area of lattice / 12 
                # => height = round(np.sqrt(area_twelfth_tile * math.sqrt(3))) and width = height / math.sqrt(3)
                # We have control over a twelfth of the tile which contains the total area of 6 FRs
                # => area_twelfth_tile = height * width (width = height / math.sqrt(3))
                # => area_twelfth_tile = height * (height / math.sqrt(3))
                # => area_twelfth_tile = height**2 / math.sqrt(3))
                # => height = round(np.sqrt(area_twelfth_tile * math.sqrt(3)))
                if (sizing == 'lattice'):
                    side_length = n / 12 # FR should be 1 / 12 the size if lattice sizing
                    area = side_length
                else:
                    area = n
                area_sixth_tile = area * 2 # we can control for a sixth of the size of a tile
                height = round(np.sqrt(area_sixth_tile * math.sqrt(3)))
                start_tile = texture[:height, :]
                p6m = new_p6m(start_tile)
                p6m_image = cat_tiles(p6m, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p6m_image, wp_type, p6m, sizing,
                               N, ratio, cmap, save_path, k, pdf)
                image.append(p6m_image)
            else:
                warnings.warn(
                    'Unexpected Wallpaper Group type. Returning random noise.', UserWarning)
                noise = np.matlib.repmat(texture, [np.ceil(N / np.sqrt(n)),  np.ceil(N / np.sqrt(n))])
                image.append(noise)
        except Exception as err:
            print('new_SymmetricNoise:Error making ' + wp_type)
            print(err.args)

    clipped_images = [clip_wallpaper(img,N) for img in image]
    return image #clipped_images


def cat_tiles(tile, N, wp_type):
    # #disp tile square
    sq = np.shape(tile)[0] * np.shape(tile)[1]
    
    # resize tile to ensure it will fit wallpaper size properly
    if (tile.shape[0] > N):
        tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
        N = tile.shape[0]
    if (tile.shape[1] > N):
        tile = tile[:, round((tile.shape[1] - N) / 2)
                             : round((N + (tile.shape[1] - N) / 2))]
        N = tile.shape[1]
    if (tile.shape[0] % 2 != 0):
        tile = tile[:tile.shape[0] - 1, :]
    if (tile.shape[1] % 2 != 0):
        tile = tile[:, :tile.shape[1] - 1]
    dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))

    row = dN[0]
    col = dN[1]

    # to avoid divide by zero errors
    if(dN[0] == 1):
        row = row + 1
    if(dN[1] == 1):
        col = col + 1

    # repeat tile to create initial wallpaper less the excess necessary to complete the wallpaper to a bigger size
    img_final = numpy.matlib.repmat(tile, row, col)
    return img_final

def clip_wallpaper(wallpaper, wp_size_pix):
    left_right_clip = round((wallpaper.shape[0] - wp_size_pix) / 2)
    up_down_clip = round((wallpaper.shape[1] - wp_size_pix) / 2)
    return wallpaper[left_right_clip : wallpaper.shape[0] - left_right_clip, up_down_clip : wallpaper.shape[1] - up_down_clip]

def diagcat_tiles(tile, N, diag_tile, wp_type):
    # Create diagnostic wallpaper
    # resize tile to ensure it will fit wallpaper size properly
    sq = np.shape(tile)[0] * np.shape(tile)[1]
    
    # resize tile to ensure it will fit wallpaper size properly
    if (tile.shape[0] > N):
        tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
        N = tile.shape[0]
    if (tile.shape[1] > N):
        tile = tile[:, round((tile.shape[1] - N) / 2)
                             : round((N + (tile.shape[1] - N) / 2))]
        N = tile.shape[1]
    
    if (tile.shape[0] % 2 != 0):
        tile = tile[:tile.shape[0] - 1, :]
    if (tile.shape[1] % 2 != 0):
        tile = tile[:, :tile.shape[1] - 1]
    dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))

    row = dN[0]
    col = dN[1]

    # to avoid divide by zero errors
    if(dN[0] == 1):
        row = row + 1
    if(dN[1] == 1):
        col = col + 1
    # repeat tile to create initial wallpaper less the excess necessary to complete the wallpaper to desired size
    img = np.tile(tile, (row, col))
    if (diag_tile.shape[0] % 2 != 0):
        diag_tile = diag_tile[:diag_tile.shape[0] - 1, :]
    if (diag_tile.shape[1] % 2 != 0):
        diag_tile = diag_tile[:, :diag_tile.shape[1] - 1]
    #img = np.rot90(img, 1)
    #diag_tile = np.rot90(diag_tile, 1)
    print(img.shape)
    print(diag_tile.shape)
    img[:diag_tile.shape[0], :diag_tile.shape[1], :diag_tile.shape[2]] = diag_tile[:, :, :]
    return img


def diagnostic(img, wp_type, tile, sizing, N, ratio, cmap, save_path, k, pdf):
    # function to take care of all diagnostic tasks related to the wallpaper generation
    # img is the full wallpaper
    # wp_type is the wallpaper type
    # tile is a single tile of the wallpaper
    # is_fr is if the wallpaper is sized as a ratio of the fundamental region
    # is_lattice is if the wallpaper is sized as a ratio of the lattice
    # N is the overall size of the wallpaper
    # ratio is the ratio of the FR/lattice sizing

    tile = np.array(tile * 255, dtype=np.uint8)
    #tile[:, :] = cv.equalizeHist(tile[:, :])

    if (wp_type == 'P1'):
        # rgb(47,79,79)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1])):.1f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - tile.shape[0] * tile.shape[1]) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), fill=(
            47, 79, 79, 125), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P2'):
        # rgb(128,0,0)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - (tile.shape[0] * tile.shape[1]) / 2) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((tile.shape[1], tile.shape[0] * 1.5, tile.shape[1] * 2, tile.shape[0] * 2), fill=(
            128, 0, 0, 125), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.rectangle(
            (tile.shape[1], tile.shape[0], tile.shape[1] * 2, tile.shape[0] * 2), outline=(255, 255, 0), width=2)

        # symmetry axes symbols
        
        # top left        
        alpha_mask__rec_draw.polygon(((tile.shape[1], tile.shape[0] - 9), (tile.shape[1] - 3, tile.shape[0]), (
            tile.shape[1], tile.shape[0] + 9), (tile.shape[1] + 3, tile.shape[0]), ((tile.shape[1], tile.shape[0] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0] - 10), (tile.shape[1] - 4, tile.shape[0]), (tile.shape[1], tile.shape[0] + 10),
                                    (tile.shape[1] + 4, tile.shape[0]), (tile.shape[1], tile.shape[0] - 10)), fill=(255, 255, 0, 255), width=1)
        # top center
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 1.5 + 4, tile.shape[0] - 7), (tile.shape[1] * 1.5 - 2, tile.shape[0] - 2), (
            tile.shape[1] * 1.5 - 4, tile.shape[0] + 7), (tile.shape[1] * 1.5 + 2, tile.shape[0] + 2), ((tile.shape[1] * 1.5 + 4, tile.shape[0] - 7))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5 + 5, tile.shape[0] - 8), (tile.shape[1] * 1.5 - 3, tile.shape[0] - 3), (tile.shape[1] * 1.5 - 5, tile.shape[0] + 8),
                                    (tile.shape[1] * 1.5 + 3, tile.shape[0] + 3), (tile.shape[1] * 1.5 + 5, tile.shape[0] - 8)), fill=(255, 255, 0, 255), width=1) 
        # top right
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 2, tile.shape[0] - 9), (tile.shape[1] * 2 - 3, tile.shape[0]), (
            tile.shape[1] * 2, tile.shape[0] + 9), (tile.shape[1] * 2 + 3, tile.shape[0]), ((tile.shape[1] * 2, tile.shape[0] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 2, tile.shape[0] - 10), (tile.shape[1] * 2 - 4, tile.shape[0]), (tile.shape[1] * 2, tile.shape[0] + 10),
                                    (tile.shape[1] * 2 + 4, tile.shape[0]), (tile.shape[1] * 2, tile.shape[0] - 10)), fill=(255, 255, 0, 255), width=1)
        # midddle left
        alpha_mask__rec_draw.polygon(((tile.shape[1] - 4, tile.shape[0] * 1.5 - 7), (tile.shape[1] - 2, tile.shape[0] * 1.5 + 2), (
            tile.shape[1] + 4, tile.shape[0] * 1.5 + 7), (tile.shape[1] + 2, tile.shape[0] * 1.5 - 2), ((tile.shape[1] - 4, tile.shape[0] * 1.5 - 7))), fill=(0,255,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1] - 5, tile.shape[0] * 1.5 - 8), (tile.shape[1] - 3, tile.shape[0] * 1.5 + 3), (tile.shape[1] + 5, tile.shape[0] * 1.5 + 8),
                                    (tile.shape[1] + 3, tile.shape[0] * 1.5 - 3), (tile.shape[1] - 5, tile.shape[0] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        # midddle center
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 1.5, tile.shape[0] * 1.5 - 3), (tile.shape[1] * 1.5 - 9, tile.shape[0] * 1.5), (
            tile.shape[1] * 1.5, tile.shape[0] * 1.5 + 3), (tile.shape[1] * 1.5 + 9, tile.shape[0] * 1.5), ((tile.shape[1] * 1.5, tile.shape[0] * 1.5 - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5, tile.shape[0] * 1.5 - 4), (tile.shape[1] * 1.5 - 10, tile.shape[0] * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5 + 4),
                                    (tile.shape[1] * 1.5 + 10, tile.shape[0] * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5 - 4)), fill=(255, 255, 0, 255), width=1) 
        # midddle right
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 2 - 4, tile.shape[0] * 1.5 - 7), (tile.shape[1] * 2 - 2, tile.shape[0] * 1.5 + 2), (
            tile.shape[1] * 2 + 4, tile.shape[0] * 1.5 + 7), (tile.shape[1] * 2 + 2, tile.shape[0] * 1.5 - 2), ((tile.shape[1] * 2 - 4, tile.shape[0] * 1.5 - 7))), fill=(0,255,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 2 - 5, tile.shape[0] * 1.5 - 8), (tile.shape[1] * 2 - 3, tile.shape[0] * 1.5 + 3), (tile.shape[1] * 2 + 5, tile.shape[0] * 1.5 + 8),
                                    (tile.shape[1] * 2 + 3, tile.shape[0] * 1.5 - 3), (tile.shape[1] * 2 - 5, tile.shape[0] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        # bottom left        
        alpha_mask__rec_draw.polygon(((tile.shape[1], tile.shape[0] * 2 - 9), (tile.shape[1] - 3, tile.shape[0] * 2), (
            tile.shape[1], tile.shape[0] * 2 + 9), (tile.shape[1] + 3, tile.shape[0] * 2), ((tile.shape[1], tile.shape[0] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0] * 2 - 10), (tile.shape[1] - 4, tile.shape[0] * 2), (tile.shape[1], tile.shape[0] * 2 + 10),
                                    (tile.shape[1] + 4, tile.shape[0] * 2), (tile.shape[1], tile.shape[0] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom center
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 1.5 + 4, tile.shape[0] * 2 - 7), (tile.shape[1] * 1.5 - 2, tile.shape[0] * 2 - 2), (
            tile.shape[1] * 1.5 - 4, tile.shape[0] * 2 + 7), (tile.shape[1] * 1.5 + 2, tile.shape[0] * 2 + 2), ((tile.shape[1] * 1.5 + 4, tile.shape[0] * 2 - 7))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5 + 5, tile.shape[0] * 2 - 8), (tile.shape[1] * 1.5 - 3, tile.shape[0] * 2 - 3), (tile.shape[1] * 1.5 - 5, tile.shape[0] * 2 + 8),
                                    (tile.shape[1] * 1.5 + 3, tile.shape[0] * 2 + 3), (tile.shape[1] * 1.5 + 5, tile.shape[0] * 2 - 8)), fill=(255, 255, 0, 255), width=1) 
        # bottom right
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 2, tile.shape[0] * 2 - 9), (tile.shape[1] * 2 - 3, tile.shape[0] * 2), (
            tile.shape[1] * 2, tile.shape[0] * 2 + 9), (tile.shape[1] * 2 + 3, tile.shape[0] * 2), ((tile.shape[1] * 2, tile.shape[0] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 2, tile.shape[0] * 2 - 10), (tile.shape[1] * 2 - 4, tile.shape[0] * 2), (tile.shape[1] * 2, tile.shape[0] * 2 + 10),
                                    (tile.shape[1] * 2 + 4, tile.shape[0] * 2), (tile.shape[1] * 2, tile.shape[0] * 2 - 10)), fill=(255, 255, 0, 255), width=1) 

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'PM'):
        # rgb(0,128,0)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        # fundamental region (horizontal mirror)
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((tile.shape[1], tile.shape[0] * 1.5, tile.shape[1] * 2, tile.shape[0] * 2), fill=(
            0, 128, 0, 125), outline=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'PG'):
        # rgb(0,0,205)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        # fundamental region (horizontal glide)
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((tile.shape[1] * 1.5, tile.shape[0], tile.shape[1] * 2, tile.shape[0] * 2), fill=(
            0, 0, 205, 125), outline=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'CM'):
        # rgb(255,20,147)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cma = plt.get_cmap("gray")
        tile_cm = cma(tile)
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cma(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        
        # fundamental region (horizontal mirror)
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1])), fill=(255,20,147, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 2), (
            tile.shape[0] * 2, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1])), fill=(255, 255, 0, 255), width=2, joint="curve")
        alpha_mask__rec_draw.line(
            ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'PMM'):
        # rgb(255,69,0)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1] * 1.5, tile.shape[0] * 1.5, tile.shape[1] * 2), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        # top left        
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] - 9), (tile.shape[0] - 3, tile.shape[1]), (
            tile.shape[0], tile.shape[1] + 9), (tile.shape[0] + 3, tile.shape[1]), ((tile.shape[0], tile.shape[1] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] - 10), (tile.shape[0] - 4, tile.shape[1]), (tile.shape[0], tile.shape[1] + 10),
                                    (tile.shape[0] + 4, tile.shape[1]), (tile.shape[0], tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        # top center
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5 + 4, tile.shape[1] - 7), (tile.shape[0] * 1.5 - 2, tile.shape[1] - 2), (
            tile.shape[0] * 1.5 - 4, tile.shape[1] + 7), (tile.shape[0] * 1.5 + 2, tile.shape[1] + 2), ((tile.shape[0] * 1.5 + 4, tile.shape[1] - 7))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5 + 5, tile.shape[1] - 8), (tile.shape[0] * 1.5 - 3, tile.shape[1] - 3), (tile.shape[0] * 1.5 - 5, tile.shape[1] + 8),
                                    (tile.shape[0] * 1.5 + 3, tile.shape[1] + 3), (tile.shape[0] * 1.5 + 5, tile.shape[1] - 8)), fill=(255, 255, 0, 255), width=1) 
        # top right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] - 9), (tile.shape[0] * 2 - 3, tile.shape[1]), (
            tile.shape[0] * 2, tile.shape[1] + 9), (tile.shape[0] * 2 + 3, tile.shape[1]), ((tile.shape[0] * 2, tile.shape[1] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] - 10), (tile.shape[0] * 2 - 4, tile.shape[1]), (tile.shape[0] * 2, tile.shape[1] + 10),
                                    (tile.shape[0] * 2 + 4, tile.shape[1]), (tile.shape[0] * 2, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        # midddle left
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 4, tile.shape[1] * 1.5 - 7), (tile.shape[0] - 2, tile.shape[1] * 1.5 + 2), (
            tile.shape[0] + 4, tile.shape[1] * 1.5 + 7), (tile.shape[0] + 2, tile.shape[1] * 1.5 - 2), ((tile.shape[0] - 4, tile.shape[1] * 1.5 - 7))), fill=(0,255,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 5, tile.shape[1] * 1.5 - 8), (tile.shape[0] - 3, tile.shape[1] * 1.5 + 3), (tile.shape[0] + 5, tile.shape[1] * 1.5 + 8),
                                    (tile.shape[0] + 3, tile.shape[1] * 1.5 - 3), (tile.shape[0] - 5, tile.shape[1] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        # midddle center
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 3), (tile.shape[0] * 1.5 - 9, tile.shape[1] * 1.5), (
            tile.shape[0] * 1.5, tile.shape[1] * 1.5 + 3), (tile.shape[0] * 1.5 + 9, tile.shape[1] * 1.5), ((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 4), (tile.shape[0] * 1.5 - 10, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5 + 4),
                                    (tile.shape[0] * 1.5 + 10, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 4)), fill=(255, 255, 0, 255), width=1) 
        # midddle right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2 - 4, tile.shape[1] * 1.5 - 7), (tile.shape[0] * 2 - 2, tile.shape[1] * 1.5 + 2), (
            tile.shape[0] * 2 + 4, tile.shape[1] * 1.5 + 7), (tile.shape[0] * 2 + 2, tile.shape[1] * 1.5 - 2), ((tile.shape[0] * 2 - 4, tile.shape[1] * 1.5 - 7))), fill=(0,255,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2 - 5, tile.shape[1] * 1.5 - 8), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.5 + 3), (tile.shape[0] * 2 + 5, tile.shape[1] * 1.5 + 8),
                                    (tile.shape[0] * 2 + 3, tile.shape[1] * 1.5 - 3), (tile.shape[0] * 2 - 5, tile.shape[1] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        # bottom left        
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 2 - 9), (tile.shape[0] - 3, tile.shape[1] * 2), (
            tile.shape[0], tile.shape[1] * 2 + 9), (tile.shape[0] + 3, tile.shape[1] * 2), ((tile.shape[0], tile.shape[1] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 2 - 10), (tile.shape[0] - 4, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 + 10),
                                    (tile.shape[0] + 4, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom center
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5 + 4, tile.shape[1] * 2 - 7), (tile.shape[0] * 1.5 - 2, tile.shape[1] * 2 - 2), (
            tile.shape[0] * 1.5 - 4, tile.shape[1] * 2 + 7), (tile.shape[0] * 1.5 + 2, tile.shape[1] * 2 + 2), ((tile.shape[0] * 1.5 + 4, tile.shape[1] * 2 - 7))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5 + 5, tile.shape[1] * 2 - 8), (tile.shape[0] * 1.5 - 3, tile.shape[1] * 2 - 3), (tile.shape[0] * 1.5 - 5, tile.shape[1] * 2 + 8),
                                    (tile.shape[0] * 1.5 + 3, tile.shape[1] * 2 + 3), (tile.shape[0] * 1.5 + 5, tile.shape[1] * 2 - 8)), fill=(255, 255, 0, 255), width=1) 
        # bottom right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 2 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 2), (
            tile.shape[0] * 2, tile.shape[1] * 2 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 2), ((tile.shape[0] * 2, tile.shape[1] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 2 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 2), (tile.shape[0] * 2, tile.shape[1] * 2 + 10),
                                    (tile.shape[0] * 2 + 4, tile.shape[1] * 2), (tile.shape[0] * 2, tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'PMG'):
        # rgb(255,165,0)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1] * 1.5, tile.shape[0] * 1.5, tile.shape[1] * 2), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        
        # top left
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.25 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.25), (
            tile.shape[0], tile.shape[1] * 1.25 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.25), (tile.shape[0], tile.shape[1] * 1.25 - 9)), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.25 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.25), (tile.shape[0], tile.shape[1] * 1.25 + 10),
                                   (tile.shape[0] + 4, tile.shape[1] * 1.25), (tile.shape[0], tile.shape[1] * 1.25 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom left
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.75 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.75), (
            tile.shape[0], tile.shape[1] * 1.75 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.75), (tile.shape[0], tile.shape[1] * 1.75 - 9)), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.75 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.75), (tile.shape[0], tile.shape[1] * 1.75 + 10),
                                   (tile.shape[0] + 4, tile.shape[1] * 1.75), (tile.shape[0], tile.shape[1] * 1.75 - 10)), fill=(255, 255, 0, 255), width=1)
        
        # middle top
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 1.25 - 3), (tile.shape[0] * 1.5 - 9, tile.shape[1] * 1.25), (
            tile.shape[0] * 1.5, tile.shape[1] * 1.25 + 3), (tile.shape[0] * 1.5 + 9, tile.shape[1] * 1.25), (tile.shape[0] * 1.5, tile.shape[1] * 1.25 - 3)), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 1.25 - 4), (tile.shape[0] * 1.5 - 10, tile.shape[1] * 1.25), (tile.shape[0] * 1.5, tile.shape[1] * 1.25 + 4),
                                   (tile.shape[0] * 1.5 + 10, tile.shape[1] * 1.25), (tile.shape[0] * 1.5, tile.shape[1] * 1.25 - 4)), fill=(255, 255, 0, 255), width=1)
        # middle bottom
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 1.75 - 3), (tile.shape[0] * 1.5 - 9, tile.shape[1] * 1.75), (
            tile.shape[0] * 1.5, tile.shape[1] * 1.75 + 3), (tile.shape[0] * 1.5 + 9, tile.shape[1] * 1.75), (tile.shape[0] * 1.5, tile.shape[1] * 1.75 - 3)), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 1.75 - 4), (tile.shape[0] * 1.5 - 10, tile.shape[1] * 1.75), (tile.shape[0] * 1.5, tile.shape[1] * 1.75 + 4),
                                   (tile.shape[0] * 1.5 + 10, tile.shape[1] * 1.75), (tile.shape[0] * 1.5, tile.shape[1] * 1.75 - 4)), fill=(255, 255, 0, 255), width=1)
        # top right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.25 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.25), (
            tile.shape[0] * 2, tile.shape[1] * 1.25 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 1.25), (tile.shape[0] * 2, tile.shape[1] * 1.25 - 9)), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.25 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 1.25), (tile.shape[0] * 2, tile.shape[1] * 1.25 + 10),
                                   (tile.shape[0] * 2 + 4, tile.shape[1] * 1.25), (tile.shape[0] * 2, tile.shape[1] * 1.25 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.75 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.75), (
            tile.shape[0] * 2, tile.shape[1] * 1.75 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 1.75), (tile.shape[0] * 2, tile.shape[1] * 1.75 - 9)), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.75 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 1.75), (tile.shape[0] * 2, tile.shape[1] * 1.75 + 10),
                                   (tile.shape[0] * 2 + 4, tile.shape[1] * 1.75), (tile.shape[0] * 2, tile.shape[1] * 1.75 - 10)), fill=(255, 255, 0, 255), width=1)
        

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'PGG'):
        # rgb(189,183,107)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 2), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 2), (tile.shape[0] * 2,
            tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0], tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        
        # top left        
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] - 9), (tile.shape[0] - 3, tile.shape[1]), (
            tile.shape[0], tile.shape[1] + 9), (tile.shape[0] + 3, tile.shape[1]), ((tile.shape[0], tile.shape[1] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] - 10), (tile.shape[0] - 4, tile.shape[1]), (tile.shape[0], tile.shape[1] + 10),
                                    (tile.shape[0] + 4, tile.shape[1]), (tile.shape[0], tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        # top right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] - 9), (tile.shape[0] * 2 - 3, tile.shape[1]), (
            tile.shape[0] * 2, tile.shape[1] + 9), (tile.shape[0] * 2 + 3, tile.shape[1]), ((tile.shape[0] * 2, tile.shape[1] - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] - 10), (tile.shape[0] * 2 - 4, tile.shape[1]), (tile.shape[0] * 2, tile.shape[1] + 10),
                                    (tile.shape[0] * 2 + 4, tile.shape[1]), (tile.shape[0] * 2, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        # midddle left
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 3), (tile.shape[0] - 9, tile.shape[1] * 1.5), (
            tile.shape[0], tile.shape[1] * 1.5 + 3), (tile.shape[0] + 9, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 4), (tile.shape[0] - 10, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 4),
                                    (tile.shape[0] + 10, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 4)), fill=(255, 255, 0, 255), width=1)
        # midddle right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 3), (tile.shape[0] * 2 - 9, tile.shape[1] * 1.5), (
            tile.shape[0] * 2, tile.shape[1] * 1.5 + 3), (tile.shape[0] * 2 + 9, tile.shape[1] * 1.5), ((tile.shape[0] * 2, tile.shape[1] * 1.5 - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 4), (tile.shape[0] * 2 - 10, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 + 4),
                                    (tile.shape[0] * 2 + 10, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 - 4)), fill=(255, 255, 0, 255), width=1)
        # midddle top
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 3), (tile.shape[0] * 1.5 - 9, tile.shape[1]), (
            tile.shape[0] * 1.5, tile.shape[1] + 3), (tile.shape[0] * 1.5 + 9, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 4), (tile.shape[0] * 1.5 - 10, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 4),
                                    (tile.shape[0] * 1.5 + 10, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 4)), fill=(255, 255, 0, 255), width=1)
        # midddle bottom
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 3), (tile.shape[0] * 1.5 - 9, tile.shape[1] * 2), (
            tile.shape[0] * 1.5, tile.shape[1] * 2 + 3), (tile.shape[0] * 1.5 + 9, tile.shape[1] * 2), ((tile.shape[0] * 1.5, tile.shape[1] * 2 - 3))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 4), (tile.shape[0] * 1.5 - 10, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 + 4),
                                    (tile.shape[0] * 1.5 + 10, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 - 4)), fill=(255, 255, 0, 255), width=1)
        # midddle center
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1] * 1.5), (
            tile.shape[0] * 1.5, tile.shape[1] * 1.5 + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1] * 1.5), ((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5 + 10),
                                    (tile.shape[0] * 1.5 + 4, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom left        
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 2 - 9), (tile.shape[0] - 3, tile.shape[1] * 2), (
            tile.shape[0], tile.shape[1] * 2 + 9), (tile.shape[0] + 3, tile.shape[1] * 2), ((tile.shape[0], tile.shape[1] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 2 - 10), (tile.shape[0] - 4, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 + 10),
                                    (tile.shape[0] + 4, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom right
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 2 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 2), (
            tile.shape[0] * 2, tile.shape[1] * 2 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 2), ((tile.shape[0] * 2, tile.shape[1] * 2 - 9))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 2 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 2), (tile.shape[0] * 2, tile.shape[1] * 2 + 10),
                                    (tile.shape[0] * 2 + 4, tile.shape[1] * 2), (tile.shape[0] * 2, tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)                
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'CMM'):
        # rgb(127,0,127)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((tile.shape[0], tile.shape[1]), (tile.shape[0] * 0.75, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2), (tile.shape[0], tile.shape[1])), fill=(127,0,127, 125))
        alpha_mask__rec_draw.line(
            ((tile.shape[0] * 0.75, tile.shape[1] * 2), (tile.shape[0] * 1.25, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], tile.shape[1]), (tile.shape[0], tile.shape[1] * 3)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1]), (tile.shape[0] * 0.75, tile.shape[1] * 2), (tile.shape[0],
             tile.shape[1] * 3), (tile.shape[0] * 1.25, tile.shape[1] * 2), (tile.shape[0], tile.shape[1])), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        
        # top middle
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] - 8), (tile.shape[0] - 3, tile.shape[1]), (
            tile.shape[0], tile.shape[1] + 8), (tile.shape[0] + 3, tile.shape[1]), ((tile.shape[0], tile.shape[1] - 8))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] - 9), (tile.shape[0] - 4, tile.shape[1]), (tile.shape[0], tile.shape[1] + 9),
                                    (tile.shape[0] + 4, tile.shape[1]), (tile.shape[0], tile.shape[1] - 9)), fill=(255, 255, 0, 255), width=1)
        # left middle
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 0.75, tile.shape[1] * 2 - 8), (tile.shape[0] * 0.75 - 3, tile.shape[1] * 2), (
            tile.shape[0] * 0.75, tile.shape[1] * 2 + 8), (tile.shape[0] * 0.75 + 3, tile.shape[1] * 2), ((tile.shape[0] * 0.75, tile.shape[1] * 2 - 8))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 0.75, tile.shape[1] * 2 - 9), (tile.shape[0] * 0.75 - 4, tile.shape[1] * 2), (tile.shape[0] * 0.75, tile.shape[1] * 2 + 9),
                                    (tile.shape[0] * 0.75 + 4, tile.shape[1] * 2), (tile.shape[0] * 0.75, tile.shape[1] * 2 - 9)), fill=(255, 255, 0, 255), width=1)
        # bottom middle
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 3 - 8), (tile.shape[0] - 3, tile.shape[1] * 3), (
            tile.shape[0], tile.shape[1] * 3 + 8), (tile.shape[0] + 3, tile.shape[1] * 3), ((tile.shape[0], tile.shape[1] * 3 - 8))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 3 - 9), (tile.shape[0] - 4, tile.shape[1] * 3), (tile.shape[0], tile.shape[1] * 3 + 9),
                                    (tile.shape[0] + 4, tile.shape[1] * 3), (tile.shape[0], tile.shape[1] * 3 - 9)), fill=(255, 255, 0, 255), width=1)
        # right middle
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.25, tile.shape[1] * 2 - 8), (tile.shape[0] * 1.25 - 3, tile.shape[1] * 2), (
            tile.shape[0] * 1.25, tile.shape[1] * 2 + 8), (tile.shape[0] * 1.25 + 3, tile.shape[1] * 2), ((tile.shape[0] * 1.25, tile.shape[1] * 2 - 8))), fill=(0,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.25, tile.shape[1] * 2 - 9), (tile.shape[0] * 1.25 - 4, tile.shape[1] * 2), (tile.shape[0] * 1.25, tile.shape[1] * 2 + 9),
                                    (tile.shape[0] * 1.25 + 4, tile.shape[1] * 2), (tile.shape[0] * 1.25, tile.shape[1] * 2 - 9)), fill=(255, 255, 0, 255), width=1)
        # center
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 8, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 - 3), (
            tile.shape[0] + 8, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 + 3), ((tile.shape[0] - 8, tile.shape[1] * 2))), fill=(255,0,0,125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 9, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2 - 4), (tile.shape[0] + 9, tile.shape[1] * 2),
                                    (tile.shape[0], tile.shape[1] * 2 + 4), (tile.shape[0] - 9, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=1)
        # right top
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.125 + 6, tile.shape[1] * 1.5 - 8), (tile.shape[0] * 1.125 + 3, tile.shape[1] * 1.5 + 3), (tile.shape[0] * 1.125 - 6, tile.shape[1] * 1.5 + 8),
                                    (tile.shape[0] * 1.125 - 3, tile.shape[1] * 1.5 - 3), (tile.shape[0] * 1.125 + 6, tile.shape[1] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.125 + 5, tile.shape[1] * 1.5 - 7), (tile.shape[0] * 1.125 + 2, tile.shape[1] * 1.5 + 2), (tile.shape[0] * 1.125 - 5, tile.shape[1] * 1.5 + 7),
                                    (tile.shape[0] * 1.125 - 2, tile.shape[1] * 1.5 - 2), (tile.shape[0] * 1.125 + 5, tile.shape[1] * 1.5 - 7)), fill=(0,255,0,125))
        # right bottom
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.125 - 6, tile.shape[1] * 2.5 - 8), (tile.shape[0] * 1.125 - 3, tile.shape[1] * 2.5 + 3), (tile.shape[0] * 1.125 + 6, tile.shape[1] * 2.5 + 8),
                                    (tile.shape[0] * 1.125 + 3, tile.shape[1] * 2.5 - 3), (tile.shape[0] * 1.125 - 6, tile.shape[1] * 2.5 - 8)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.125 - 5, tile.shape[1] * 2.5 - 7), (tile.shape[0] * 1.125 - 2, tile.shape[1] * 2.5 + 2), (tile.shape[0] * 1.125 + 6, tile.shape[1] * 2.5 + 7),
                                    (tile.shape[0] * 1.125 + 2, tile.shape[1] * 2.5 - 2), (tile.shape[0] * 1.125 - 5, tile.shape[1] * 2.5 - 7)), fill=(0,255,0,125))
        # left bottom
        alpha_mask__rec_draw.line(((tile.shape[0] * 0.875 + 6, tile.shape[1] * 2.5 - 8), (tile.shape[0] * 0.875 + 3, tile.shape[1] * 2.5 + 3), (tile.shape[0] * 0.875 - 6, tile.shape[1] * 2.5 + 8),
                                    (tile.shape[0] * 0.875 - 3, tile.shape[1] * 2.5 - 3), (tile.shape[0] * 0.875 + 6, tile.shape[1] * 2.5 - 8)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 0.875 + 5, tile.shape[1] * 2.5 - 7), (tile.shape[0] * 0.875 + 2, tile.shape[1] * 2.5 + 2), (tile.shape[0] * 0.875 - 5, tile.shape[1] * 2.5 + 7),
                                    (tile.shape[0] * 0.875 - 2, tile.shape[1] * 2.5 - 2), (tile.shape[0] * 0.875 + 5, tile.shape[1] * 2.5 - 7)), fill=(0,255,0,125))
        # left top
        alpha_mask__rec_draw.line(((tile.shape[0] * 0.875 - 6, tile.shape[1] * 1.5 - 8), (tile.shape[0] * 0.875 - 3, tile.shape[1] * 1.5 + 3), (tile.shape[0] * 0.875 + 6, tile.shape[1] * 1.5 + 8),
                                    (tile.shape[0] * 0.875 + 3, tile.shape[1] * 1.5 - 3), (tile.shape[0] * 0.875 - 6, tile.shape[1] * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)        
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 0.875 - 5, tile.shape[1] * 1.5 - 7), (tile.shape[0] * 0.875 - 2, tile.shape[1] * 1.5 + 2), (tile.shape[0] * 0.875 + 5, tile.shape[1] * 1.5 + 7),
                                    (tile.shape[0] * 0.875 + 2, tile.shape[1] * 1.5 - 2), (tile.shape[0] * 0.875 - 5, tile.shape[1] * 1.5 - 7)), fill=(0,255,0,125))
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        
        

    elif (wp_type == 'P4'):
        # rgb(124,252,0)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1] * 1.5, tile.shape[0] * 1.5, tile.shape[1] * 2), fill=(124, 252, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        
        # top left
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0], tile.shape[1]), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
        # bottom left
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0], tile.shape[1] * 2), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
        # bottom right
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] * 2, tile.shape[1] * 2), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
        # top right
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] * 2, tile.shape[1]), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
        # center
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] * 1.5, tile.shape[1] * 1.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
        # top center        
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1]), (
            tile.shape[0] * 1.5, tile.shape[1] + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 9))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 10),
                                    (tile.shape[0] * 1.5 + 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        # bottom center        
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1] * 2), (
            tile.shape[0] * 1.5, tile.shape[1] * 2 + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1] * 2), ((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 + 10),
                                    (tile.shape[0] * 1.5 + 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
        # left center        
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.5), (
            tile.shape[0], tile.shape[1] * 1.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 10),
                                    (tile.shape[0] + 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
        # right center        
        alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.5), (
            tile.shape[0] * 2, tile.shape[1] * 1.5 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 1.5), ((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 + 10),
                                    (tile.shape[0] * 2 + 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)       
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P4M'):
        # rgb(0,250,154)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        if ratio > 0.03125:
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1]), (
                tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 0.5, tile.shape[1] * 1.5)), fill=(0, 250, 154, 125))
            alpha_mask__rec_draw.rectangle(
                (tile.shape[0] * 0.5, tile.shape[1] * 0.5, tile.shape[0] * 1.5, tile.shape[1] * 1.5), outline=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 0.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0], tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 0.5, tile.shape[1] * 0.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 1.5, tile.shape[1] * 0.5), (tile.shape[0] * 0.5, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
    
            # symmetry axes symbols
            
            # top left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 0.5, tile.shape[1] * 0.5), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # bottom left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 0.5, tile.shape[1] * 1.5), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # bottom right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 1.5), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # top right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 0.5), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # center
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1]), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 0.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 0.5), (
                tile.shape[0], tile.shape[1] * 0.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 0.5), ((tile.shape[0], tile.shape[1] * 0.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 0.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 0.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 0.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # bottom center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.5), (
                tile.shape[0], tile.shape[1] * 1.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # left center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 0.5, tile.shape[1] - 9), (tile.shape[0] * 0.5 - 3, tile.shape[1]), (
                tile.shape[0] * 0.5, tile.shape[1] + 9), (tile.shape[0] * 0.5 + 3, tile.shape[1]), ((tile.shape[0] * 0.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 0.5, tile.shape[1] - 10), (tile.shape[0] * 0.5 - 4, tile.shape[1]), (tile.shape[0] * 0.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 0.5 + 4, tile.shape[1]), (tile.shape[0] * 0.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
            # right center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1]), (
                tile.shape[0] * 1.5, tile.shape[1] + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        else:
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 1.5), (
                tile.shape[0] * 1.5, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 2)), fill=(0, 250, 154, 125))
            alpha_mask__rec_draw.rectangle(
                (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0], tile.shape[1]), (tile.shape[0] * 2, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 2, tile.shape[1]), (tile.shape[0], tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
    
            # symmetry axes symbols
            
            # top left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1]), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # bottom left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1] * 2), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # bottom right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 2, tile.shape[1] * 2), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # top right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 2, tile.shape[1]), 6), 4, 45, fill=(255,0,0,125), outline=(255, 255, 0))
            # center
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 1.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1]), (
                tile.shape[0] * 1.5, tile.shape[1] + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
            # bottom center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1] * 2), (
                tile.shape[0] * 1.5, tile.shape[1] * 2 + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1] * 2), ((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
            # left center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.5), (
                tile.shape[0], tile.shape[1] * 1.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # right center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.5), (
                tile.shape[0] * 2, tile.shape[1] * 1.5 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 1.5), ((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] * 2 + 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P4G'):
        # rgb(65,105,225)
        if(sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1])):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]))) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        if ratio > 0.03125:
            alpha_mask__rec_draw.polygon(
                ((tile.shape[0] * 0.5, tile.shape[1]), (tile.shape[0], tile.shape[1]), (tile.shape[0], tile.shape[1] * 1.5)), fill=(65, 105, 225, 125))
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 0.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0], tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(((tile.shape[0] * 0.5, tile.shape[1]), (tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5,
                                        tile.shape[1]), (tile.shape[0], tile.shape[1] * 0.5), (tile.shape[0] * 0.5, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.rectangle(
                (tile.shape[0] * 0.5, tile.shape[1] * 0.5, tile.shape[0] * 1.5, tile.shape[1] * 1.5), outline=(255, 255, 0, 255), width=2)
    
            # symmetry axes symbols
            
            # top left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 0.5, tile.shape[1] * 0.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # bottom left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 0.5, tile.shape[1] * 1.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # bottom right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 1.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 0.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # center
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1]), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 0.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 0.5), (
                tile.shape[0], tile.shape[1] * 0.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 0.5), ((tile.shape[0], tile.shape[1] * 0.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 0.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 0.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 0.5), (tile.shape[0], tile.shape[1] * 0.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # bottom center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.5), (
                tile.shape[0], tile.shape[1] * 1.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # left center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 0.5, tile.shape[1] - 9), (tile.shape[0] * 0.5 - 3, tile.shape[1]), (
                tile.shape[0] * 0.5, tile.shape[1] + 9), (tile.shape[0] * 0.5 + 3, tile.shape[1]), ((tile.shape[0] * 0.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 0.5, tile.shape[1] - 10), (tile.shape[0] * 0.5 - 4, tile.shape[1]), (tile.shape[0] * 0.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 0.5 + 4, tile.shape[1]), (tile.shape[0] * 0.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
            # right center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1]), (
                tile.shape[0] * 1.5, tile.shape[1] + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
        else:
            alpha_mask__rec_draw.polygon(
                ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(65, 105, 225, 125))
            alpha_mask__rec_draw.line(
                ((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(
                ((tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] * 2)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1]), (tile.shape[0] * 2,
                                        tile.shape[1] * 1.5), (tile.shape[0] * 1.5, tile.shape[1] * 2), (tile.shape[0], tile.shape[1] * 1.5)), fill=(255, 255, 0, 255), width=2)
            alpha_mask__rec_draw.rectangle(
                (tile.shape[0], tile.shape[1], tile.shape[0] * 2, tile.shape[1] * 2), outline=(255, 255, 0, 255), width=2)
            # symmetry axes symbols
        
            # top left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1]), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # bottom left
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0], tile.shape[1] * 2), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # bottom right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 2, tile.shape[1] * 2), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top right
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 2, tile.shape[1]), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # center
            alpha_mask__rec_draw.regular_polygon(
                ((tile.shape[0] * 1.5, tile.shape[1] * 1.5), 6), 4, 0, fill=(0,255,0,125), outline=(255, 255, 0))
            # top center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1]), (
                tile.shape[0] * 1.5, tile.shape[1] + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1]), ((tile.shape[0] * 1.5, tile.shape[1] - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1]), (tile.shape[0] * 1.5, tile.shape[1] - 10)), fill=(255, 255, 0, 255), width=1)
            # bottom center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9), (tile.shape[0] * 1.5 - 3, tile.shape[1] * 2), (
                tile.shape[0] * 1.5, tile.shape[1] * 2 + 9), (tile.shape[0] * 1.5 + 3, tile.shape[1] * 2), ((tile.shape[0] * 1.5, tile.shape[1] * 2 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 1.5, tile.shape[1] * 2 - 10), (tile.shape[0] * 1.5 - 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 + 10),
                                        (tile.shape[0] * 1.5 + 4, tile.shape[1] * 2), (tile.shape[0] * 1.5, tile.shape[1] * 2 - 10)), fill=(255, 255, 0, 255), width=1)
            # left center        
            alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] * 1.5 - 9), (tile.shape[0] - 3, tile.shape[1] * 1.5), (
                tile.shape[0], tile.shape[1] * 1.5 + 9), (tile.shape[0] + 3, tile.shape[1] * 1.5), ((tile.shape[0], tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] * 1.5 - 10), (tile.shape[0] - 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] + 4, tile.shape[1] * 1.5), (tile.shape[0], tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
            # right center        
            alpha_mask__rec_draw.polygon(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9), (tile.shape[0] * 2 - 3, tile.shape[1] * 1.5), (
                tile.shape[0] * 2, tile.shape[1] * 1.5 + 9), (tile.shape[0] * 2 + 3, tile.shape[1] * 1.5), ((tile.shape[0] * 2, tile.shape[1] * 1.5 - 9))), fill=(255,0,255,125))
            alpha_mask__rec_draw.line(((tile.shape[0] * 2, tile.shape[1] * 1.5 - 10), (tile.shape[0] * 2 - 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 + 10),
                                        (tile.shape[0] * 2 + 4, tile.shape[1] * 1.5), (tile.shape[0] * 2, tile.shape[1] * 1.5 - 10)), fill=(255, 255, 0, 255), width=1)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P3'):
        # rgb(233,150,122)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 6):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 6)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0]), (tile.shape[1] * 1.5, tile.shape[0] * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * 0.5, tile.shape[0]  * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/6) + 2, tile.shape[0]  * 1.5 - 2), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * (7/6) - 2, tile.shape[0]  * 1.5 - 2), (tile.shape[1] - 2, tile.shape[0] + 2), (tile.shape[1] * (5/6) + 2, tile.shape[0]  * 1.5 - 2)), fill=(233, 150, 122, 125))
        # symmetry axes symbols
        # left center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), 6), 3, 20, fill=(0,0,255,125), outline=(255, 255, 0))
        # left center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # top center triangle
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1], tile.shape[0])), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # right center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 1.5, tile.shape[0] * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # bottom center triangle
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1], (tile.shape[0] * 2), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # right center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), 6), 3, 55, fill=(0,255,0,125), outline=(255, 255, 0))
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P3M1'):
        # rgb(0,191,255)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 12):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 12)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5, tile.shape[0]  * (4 / 6)), (tile.shape[1], tile.shape[0]  * (7 / 6)), (tile.shape[1] * 2, tile.shape[0]  * (7 / 6)), (tile.shape[1] * 2.5, tile.shape[0] * (4 / 6)), (tile.shape[1] * 1.5, tile.shape[0]  * (4 / 6))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5, tile.shape[0]  * (4 / 6)), (tile.shape[1] * 1.5, tile.shape[0]), (tile.shape[1] * 2, tile.shape[0]  * (7 / 6)), tile.shape[1] * 2, tile.shape[0] * (5 / 6), (tile.shape[1] * 1.5, tile.shape[0]  * (4 / 6))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0]  * (7 / 6)), tile.shape[1] * 2.5, tile.shape[0] * (4 / 6)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 2, tile.shape[0] * (5 / 6)), (tile.shape[1] * 2, tile.shape[0]  * (4 / 6))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5, tile.shape[0]), (tile.shape[1] * 1.5, tile.shape[0]  * (7 / 6))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 1.5, tile.shape[0]), (tile.shape[1] * 1.25, tile.shape[0]  * (11 / 12))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 2, tile.shape[0] * (5 / 6)), (tile.shape[1] * 2.25, tile.shape[0]  * (11 / 12))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.polygon(((tile.shape[1] * 1.5 + 2, tile.shape[0] - 2), (tile.shape[1] * 1.5 + 2, tile.shape[0]  * (4 / 6) + 2), (tile.shape[1] * 2 - 3, tile.shape[0] * (5 / 6)), (tile.shape[1] * 1.5 + 2, tile.shape[0] - 2)), fill=(0, 191, 255, 125))
        # symmetry axes symbols
        # left top triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 1.5, tile.shape[0]  * (4 / 6)), 6), 3, 0, fill=(255,0,0,125), outline=(255, 255, 0))
        # left bottom triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1], tile.shape[0]  * (7 / 6)), 6), 3, 0, fill=(255,0,0,125), outline=(255, 255, 0))
        # left center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] * 1.5, tile.shape[0])), 6), 3, 165, fill=(0,0,255,125), outline=(255, 255, 0))
        # right top triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 2.5, tile.shape[0] * (4 / 6)), 6), 3, 0, fill=(255,0,0,125), outline=(255, 255, 0))
        # right bottom triangle
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] * 2, tile.shape[0]  * (7 / 6), 6), 3, 0, fill=(255,0,0,125), outline=(255, 255, 0))
        # right center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] * 2, tile.shape[0] * (5 / 6)), 6), 3, 65, fill=(0,255,0,125), outline=(255, 255, 0))
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P31M'):
        # rgb(255,0,255)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 12):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 12)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2), tile.shape[1], (tile.shape[0] * 2), (tile.shape[1] * 0.5 + 2, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2)), fill=(255, 0, 255, 125))
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0]), (tile.shape[1] * 1.5, tile.shape[0] * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * 0.5, tile.shape[0]  * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0]), (tile.shape[1], (tile.shape[0] * 2))), fill=(255, 255, 0), width=2)
        
        # symmetry axes symbols
        # left center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), 6), 3, 30, fill=(0,0,255,255), outline=(255, 255, 0))
        # left center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # top center triangle
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1], tile.shape[0])), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # right center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 1.5, tile.shape[0] * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # bottom center triangle
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1], (tile.shape[0] * 2), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # right center 3-rot triangle
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), 6), 3, 30, fill=(0,0,255,255), outline=(255, 255, 0))
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P6'):
        # rgb(221,160,221)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 12):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 12)) / (N**2 * ratio)) * 100):.2f}%')
        else:
           print('Area of Lattice of ' + wp_type +
                 f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
           print('Area of Lattice Region of ' +
                 wp_type + ' should be = ', (N**2 * ratio))
           print(
               f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0]), (tile.shape[1] * 1.5, tile.shape[0] * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * 0.5, tile.shape[0]  * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0]), (tile.shape[1], (tile.shape[0] * 2))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2), tile.shape[1], (tile.shape[0] * 2), (tile.shape[1] * 0.5 + 2, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2)), fill=(221, 160, 221, 125))
        # symmetry axes symbols
        # left center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # left side bottom diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.75 - 7), (tile.shape[1] * (3/4) + 2, tile.shape[0]  * 1.75 + 2), (tile.shape[1] * (3/4) - 5, tile.shape[0]  * 1.75 + 7),
                                    (tile.shape[1] * (3/4) - 2, tile.shape[0]  * 1.75 - 2), (tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.75 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.75 - 8), (tile.shape[1] * (3/4) + 3, tile.shape[0]  * 1.75 + 3), (tile.shape[1] * (3/4) - 6, tile.shape[0]  * 1.75 + 8),
                                    (tile.shape[1] * (3/4) - 3, tile.shape[0]  * 1.75 - 3), (tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.75 - 8)), fill=(255, 255, 0, 255), width=1)
        # left side top diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.25 - 7), (tile.shape[1] * (3/4) + 2, tile.shape[0]  * 1.25 + 2), (tile.shape[1] * (3/4) - 5, tile.shape[0]  * 1.25 + 7),
                                    (tile.shape[1] * (3/4) - 2, tile.shape[0]  * 1.25 - 2), (tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.25 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.25 - 8), (tile.shape[1] * (3/4) + 3, tile.shape[0]  * 1.25 + 3), (tile.shape[1] * (3/4) - 6, tile.shape[0]  * 1.25 + 8),
                                    (tile.shape[1] * (3/4) - 3, tile.shape[0]  * 1.25 - 3), (tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.25 - 8)), fill=(255, 255, 0, 255), width=1)
        # right side bottom diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.75 - 7), (tile.shape[1] * (5/4) + 2, tile.shape[0]  * 1.75 + 2), (tile.shape[1] * (5/4) - 5, tile.shape[0]  * 1.75 + 7),
                                    (tile.shape[1] * (5/4) - 2, tile.shape[0]  * 1.75 - 2), (tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.75 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.75 - 8), (tile.shape[1] * (5/4) + 3, tile.shape[0]  * 1.75 + 3), (tile.shape[1] * (5/4) - 6, tile.shape[0]  * 1.75 + 8),
                                    (tile.shape[1] * (5/4) - 3, tile.shape[0]  * 1.75 - 3), (tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.75 - 8)), fill=(255, 255, 0, 255), width=1)  
        # right side top diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.25 - 7), (tile.shape[1] * (5/4) + 2, tile.shape[0]  * 1.25 + 2), (tile.shape[1] * (5/4) - 5, tile.shape[0]  * 1.25 + 7),
                                    (tile.shape[1] * (5/4) - 2, tile.shape[0]  * 1.25 - 2), (tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.25 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.25 - 8), (tile.shape[1] * (5/4) + 3, tile.shape[0]  * 1.25 + 3), (tile.shape[1] * (5/4) - 6, tile.shape[0]  * 1.25 + 8),
                                    (tile.shape[1] * (5/4) - 3, tile.shape[0]  * 1.25 - 3), (tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.25 - 8)), fill=(255, 255, 0, 255), width=1)     
        # left center hexagon
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # top center hexagon
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1], tile.shape[0])), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # right center hexagon
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 1.5, tile.shape[0] * 1.5), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # bottom center hexagon
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1], (tile.shape[0] * 2), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # right center triangle
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # center diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] + 5, tile.shape[0]  * 1.5 - 7), (tile.shape[1] + 2, tile.shape[0]  * 1.5 + 2), (tile.shape[1] - 5, tile.shape[0]  * 1.5 + 7),
                                    (tile.shape[1] - 2, tile.shape[0]  * 1.5 - 2), (tile.shape[1] + 5, tile.shape[0]  * 1.5 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] + 6, tile.shape[0]  * 1.5 - 8), (tile.shape[1] + 3, tile.shape[0]  * 1.5 + 3), (tile.shape[1] - 6, tile.shape[0]  * 1.5 + 8),
                                    (tile.shape[1] - 3, tile.shape[0]  * 1.5 - 3), (tile.shape[1] + 6, tile.shape[0]  * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

    elif (wp_type == 'P6M'):
        # rgb(143,188,143)
        if (sizing == 'fr'):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 24):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 24)) / (N**2 * ratio)) * 100):.2f}%')
        else:
            print('Area of Lattice of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Lattice Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')      
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        # resize tile to ensure it will fit wallpaper size properly
        if (tile.shape[0] > N):
            tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), :]
            N = tile.shape[0]
        if (tile.shape[1] > N):
            tile = tile[:, round((tile.shape[1] - N) / 2)
                                 : round((N + (tile.shape[1] - N) / 2))]
            N = tile.shape[1]
        
        if (tile.shape[0] % 2 != 0):
            tile = tile[:tile.shape[0] - 1, :]
        if (tile.shape[1] % 2 != 0):
            tile = tile[:, :tile.shape[1] - 1]
        dN = tuple(1 + (math.floor(N / ti)) for ti in np.shape(tile))
    
        row = dN[0]
        col = dN[1]
    
        # to avoid divide by zero errors
        if(dN[0] == 1):
            row = row + 1
        if(dN[1] == 1):
            col = col + 1
        tile_rep = numpy.matlib.repmat(tile, row, col)
        tile_cm = cm(tile_rep)
        dia_fr_im = Image.fromarray(
            (tile_cm[:, :, :] * 255).clip(0,255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0]), (tile.shape[1] * 1.5, tile.shape[0] * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * 0.5, tile.shape[0]  * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1], (tile.shape[0] * 2)), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1], tile.shape[0])), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1] * 1.5, tile.shape[0] * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1], tile.shape[0]), (tile.shape[1], (tile.shape[0] * 2))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (3/4), tile.shape[0]  * 1.25), (tile.shape[1] * (5/6), tile.shape[0]  * 1.5), (tile.shape[1] * (3/4), tile.shape[0]  * 1.75)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/4), tile.shape[0]  * 1.25), (tile.shape[1] * (7/6), tile.shape[0]  * 1.5), (tile.shape[1] * (5/4), tile.shape[0]  * 1.75)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2), (tile.shape[1] * (3/4) - 2, tile.shape[0]  * 1.75 - 2), (tile.shape[1] * 0.5 + 2, tile.shape[0]  * 1.5), (tile.shape[1] * (5/6) - 2, tile.shape[0]  * 1.5 + 2)), fill=(143,188,143, 125))
        # symmetry axes symbols
        # left center triangle
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * (5/6), tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # left side bottom diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.75 - 7), (tile.shape[1] * (3/4) + 2, tile.shape[0]  * 1.75 + 2), (tile.shape[1] * (3/4) - 5, tile.shape[0]  * 1.75 + 7),
                                    (tile.shape[1] * (3/4) - 2, tile.shape[0]  * 1.75 - 2), (tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.75 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.75 - 8), (tile.shape[1] * (3/4) + 3, tile.shape[0]  * 1.75 + 3), (tile.shape[1] * (3/4) - 6, tile.shape[0]  * 1.75 + 8),
                                    (tile.shape[1] * (3/4) - 3, tile.shape[0]  * 1.75 - 3), (tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.75 - 8)), fill=(255, 255, 0, 255), width=1)
        # left side top diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.25 - 7), (tile.shape[1] * (3/4) + 2, tile.shape[0]  * 1.25 + 2), (tile.shape[1] * (3/4) - 5, tile.shape[0]  * 1.25 + 7),
                                    (tile.shape[1] * (3/4) - 2, tile.shape[0]  * 1.25 - 2), (tile.shape[1] * (3/4) + 5, tile.shape[0]  * 1.25 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.25 - 8), (tile.shape[1] * (3/4) + 3, tile.shape[0]  * 1.25 + 3), (tile.shape[1] * (3/4) - 6, tile.shape[0]  * 1.25 + 8),
                                    (tile.shape[1] * (3/4) - 3, tile.shape[0]  * 1.25 - 3), (tile.shape[1] * (3/4) + 6, tile.shape[0]  * 1.25 - 8)), fill=(255, 255, 0, 255), width=1)
        # right side bottom diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.75 - 7), (tile.shape[1] * (5/4) + 2, tile.shape[0]  * 1.75 + 2), (tile.shape[1] * (5/4) - 5, tile.shape[0]  * 1.75 + 7),
                                    (tile.shape[1] * (5/4) - 2, tile.shape[0]  * 1.75 - 2), (tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.75 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.75 - 8), (tile.shape[1] * (5/4) + 3, tile.shape[0]  * 1.75 + 3), (tile.shape[1] * (5/4) - 6, tile.shape[0]  * 1.75 + 8),
                                    (tile.shape[1] * (5/4) - 3, tile.shape[0]  * 1.75 - 3), (tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.75 - 8)), fill=(255, 255, 0, 255), width=1)  
        # right side top diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.25 - 7), (tile.shape[1] * (5/4) + 2, tile.shape[0]  * 1.25 + 2), (tile.shape[1] * (5/4) - 5, tile.shape[0]  * 1.25 + 7),
                                    (tile.shape[1] * (5/4) - 2, tile.shape[0]  * 1.25 - 2), (tile.shape[1] * (5/4) + 5, tile.shape[0]  * 1.25 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.25 - 8), (tile.shape[1] * (5/4) + 3, tile.shape[0]  * 1.25 + 3), (tile.shape[1] * (5/4) - 6, tile.shape[0]  * 1.25 + 8),
                                    (tile.shape[1] * (5/4) - 3, tile.shape[0]  * 1.25 - 3), (tile.shape[1] * (5/4) + 6, tile.shape[0]  * 1.25 - 8)), fill=(255, 255, 0, 255), width=1)  
        # left center hexagon
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 0.5, tile.shape[0]  * 1.5), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # top center hexagon
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1], tile.shape[0])), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # right center hexagon
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] * 1.5, tile.shape[0] * 1.5), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # bottom center hexagon
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1], (tile.shape[0] * 2), 6), 6, 15, fill=(0,0,255,125), outline=(255, 255, 0))
        # right center triangle
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] * (7/6), tile.shape[0]  * 1.5), 6), 3, 90, fill=(255,0,0,125), outline=(255, 255, 0))
        # center diamond
        alpha_mask__rec_draw.polygon(((tile.shape[1] + 5, tile.shape[0]  * 1.5 - 7), (tile.shape[1] + 2, tile.shape[0]  * 1.5 + 2), (tile.shape[1] - 5, tile.shape[0]  * 1.5 + 7),
                                    (tile.shape[1] - 2, tile.shape[0]  * 1.5 - 2), (tile.shape[1] + 5, tile.shape[0]  * 1.5 - 7)), fill=(255,0,255,125))
        alpha_mask__rec_draw.line(((tile.shape[1] + 6, tile.shape[0]  * 1.5 - 8), (tile.shape[1] + 3, tile.shape[0]  * 1.5 + 3), (tile.shape[1] - 6, tile.shape[0]  * 1.5 + 8),
                                    (tile.shape[1] - 3, tile.shape[0]  * 1.5 - 3), (tile.shape[1] + 6, tile.shape[0]  * 1.5 - 8)), fill=(255, 255, 0, 255), width=1)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
    
    # resize diagnostic_fr image to actual size of wallpaper
    im_w, im_h = dia_fr_im.size
    left = round((im_w - N) / 2)
    right = round(N + ((im_w - N) / 2))
    top = round((im_h - N) / 2)
    bottom = round(N + ((im_h - N) / 2))
    dia_fr_im = dia_fr_im.crop((left, top, right, bottom))
    pattern_path = save_path + '/' + wp_type + '_FundamentalRegion_' + str(k + 1) + '.' + "png"
    dia_fr_im.save(pattern_path, "png")
    
    
    # diagnostic plots
    logging.getLogger('matplotlib.font_manager').disabled = True
    pattern_path = save_path + '/' + wp_type + '_diagnostic_all_' + str(k + 1) + '.' + "png"
    hidx_0 = int(img.shape[0] * (1 / 3))
    hidx_1 = int(img.shape[0] / 2)
    hidx_2 = int(img.shape[0] * (2 / 3))
    I = np.dstack([img, img, img])
    I[hidx_0 - 2:hidx_0 + 2, :] = np.array([1, 0, 0])
    I[hidx_1 - 2:hidx_1 + 2, :] = np.array([0, 1, 0])
    I[hidx_2 - 2:hidx_2 + 2, :] = np.array([0, 0, 1])
    I = clip_wallpaper(I, N)
    cm = plt.get_cmap("gray")
    cm(I)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 40))
    ax1.imshow(dia_fr_im)
    ax1.set_title('Fundamental Region for ' + wp_type)
    ax1.set(adjustable='box', aspect='auto')
    
    ax2.imshow(I)
    ax2.set_title(wp_type + ' diagnostic image 1')
    ax2.set(adjustable='box', aspect='auto')
    
    bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
    ax2_path = save_path + '/' + wp_type + '_diagnostic_1_' + str(k + 1) + '.' + "png"
    fig.savefig(ax2_path,bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))

    ax3.plot(img[hidx_0, :], c=[1, 0, 0])
    ax3.plot(img[hidx_1, :], c=[0, 1, 0])
    ax3.plot(img[hidx_2, :], c=[0, 0, 1])
    ax3.set_title('Sample values along the horizontal lines {} {} and {}'.format(
        hidx_0, hidx_1, hidx_2))
    
    bbox = ax3.get_tightbbox(fig.canvas.get_renderer())
    ax3_path = save_path + '/' + wp_type + '_diagnostic_2_' + str(k + 1) + '.' + "png"
    fig.savefig(ax3_path,bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))

    bins = np.linspace(0, 1, 100)
    ax4.hist(img[hidx_0, :], bins, color=[1, 0, 0])
    ax4.hist(img[hidx_1, :], bins, color=[0, 1, 0])
    ax4.hist(img[hidx_2, :], bins, color=[0, 0, 1])
    ax4.set_title('Frequency of sample values across the horizontal lines')
    
    bbox = ax4.get_tightbbox(fig.canvas.get_renderer())
    ax4_path = save_path + '/' + wp_type + '_diagnostic_3_' + str(k + 1) + '.' + "png"
    fig.savefig(ax4_path,bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))


    # add figure to pdf
    pdf.savefig(fig)

    plt.show()
    fig.savefig(pattern_path)

# array of coefficients(DO NOT CHANGE):
# tile sizes by groups:     tile_in     tile_out         square ratio       width ratio
# P1:                        (n, n)      (n, n)              1                  1
# P2:                        (n, n)      (n, 2n)             2                  0.5
# PM, PG:                    (n, n)      (2n, n)             2                  1
# PMG, PMM, P4, P4M:         (n, n)      (2n, 2n)            4                  0.5
# PGG:                       (n, 2n)     (2n, 2n)            4                  0.5
# CM:                        (n, n)      (n, 2n)             2                  0.25
# P4G:                       (n, 2n)     (4n, 4n)            16                 0.25
# CMM:                       (n, n)      (4n, 4n)            16                 0.25
# P3:                        (n, n)      (3n, n sqrt(3))     3sqrt(3)           1/sqrt(3)
# P31M: s = round(n*2.632):  (s, s)      (3s, s sqrt(3))     3s^2 sqrt(3)/n^2   1/(2.632*sqrt(3))
# P3M1, P6, P6M:             (s, s)      (s, s sqrt(3))      s^2 sqrt(3)/n^2    1/(2.632*sqrt(3))


def matlab_style_gauss2D(shape, sigma):
    """
    This code was taken from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# filter/mask every image


def filter_img(inImg, N):
    # make filter intensity adaptive (600 is empirical number)
    sigma = N / 600
    low_pass = matlab_style_gauss2D((9, 9), sigma)

    # filter
    image = scipy.ndimage.correlate(
        inImg, low_pass, mode='reflect')
    # normalize
    image = image / np.ptp(image)  # scale to unit range
    image = image - np.mean(image[:])  # bring mean luminance to zero
    image = image / np.max(np.abs(image[:]))  # Scale so max signed value is 1
    image = 125 * image + 127  # Scale into 2-252 range
    image = image / 255

    out_img = image
    return out_img

# apply mask


def mask_img(inImg, N):
    # define mask(circle)
    r = round(0.5 * N)
    mask = np.zeros((inImg.shape[0], inImg.shape[1]), np.uint8)
    cv.circle(mask, (round(inImg.shape[1] / 2),
                     round(inImg.shape[0] / 2)), r, 1, -1)
    mask = cv.bitwise_and(mask, mask, mask=mask)
    out_img = inImg[:np.shape(mask)[0], :np.shape(mask)[1]]
    out_img[mask == 0] = 0.5
    return out_img

def mask_imgs(imgs, radius=None):
    w = imgs.shape[-2]
    h = imgs.shape[-1]
    center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    assert 2*radius <= w , '{} > {}'.format(2*radius, w)
    assert 2*radius <= h , '{} > {}'.format(2*radius, h)
    x,y = np.ogrid[:w,:h]
    dists = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    mask = dists>radius
    imgs[np.broadcast_to(mask,imgs.shape)] = 0.5
    return imgs


# replace spectra


def replace_spectra(in_image,  n_phase_scrambles=1, ps_scrambling=False, use_magnitude=np.array([]), cmap="gray"):
    in_image = (in_image / np.max(in_image)) * 2.0 - 1.0

    if n_phase_scrambles > 0 and ps_scrambling:
        LOGGER.warning('n_phase_scambles is {} and ps_scrambling is True. Defaulting to phase scrambling'.format(n_phase_scrambles))
        ps_scrambling = False
    # phase scrambling
    if n_phase_scrambles >0:
        return minPhaseInterp(in_image, n_phase_scrambles)
    
    # Portilla-Simoncelli scrambling
    elif ps_scrambling:
        out_image = psScramble(in_image, cmap)
        out_image = out_image[:,:,np.newaxis]
    else: # TODO: What is happening here?
        in_spectrum = np.fft.rfft2( in_image )

        phase = np.fft.fftshift(np.angle(in_spectrum))
        mag = np.fft.fftshift(np.abs(in_spectrum))

        # use new magnitude instead
        if(use_magnitude.size != 0):
            mag = use_magnitude
        cmplx_im = mag * np.exp(1j * phase)

        # get the real parts and then take the absolute value of the real parts as this is the closest solution to be found to emulate matlab's ifft2 "symmetric" parameter
        # out_image = np.abs(np.real(np.fft.ifft2(cmplx_im)))
        # the above does not seem to work, instead use code below
        # cribbed from https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        out_image = np.real(np.fft.irfft2(np.fft.ifftshift(cmplx_im)))

    # standardize image
#    out_image = out_image - np.mean(out_image)
#    out_image = out_image / np.std(out_image)
#    out_image = out_image * 0.5 ## TO DO: MAKE THIS AN ADJUSTABLE INPUT VARIABLE
#    out_image = np.clip(out_image, a_min=-1.0, a_max=1.0)
#
#    out_image = (out_image + 1) / 2

    return out_image

def minPhaseInterp(in_image, n):
    """
    This code comes from Justin Ales' work on minimum phase image interpolation. 
    This python implementation comes from the original matlab implementation done by Justin Ales.
    All notes and comments are from Justin Ales unless otherwise stated.
    More information on the original implementation can be found at https://www.st-andrews.ac.uk/~jma23/code.html
    """
    
    image_start = np.random.rand(in_image.shape[0], in_image.shape[1])
    interp_vals = np.linspace(0, 1, n)
    
    image_start_size = image_start.shape
    in_image_size =   in_image.shape
    
    
    # Setup interpolation function
    x = []
    y = []
    z = []
    x = [a for a in range(image_start_size[0])]
    y = [b for b in range(image_start_size[1])]
    z = [0, 1]
    
    
    # Take fourier transform of input images and decompopse complex values
    # to phase and amplitude
    # Use amplitude spectrum for the end image on the first image.
    # This keeps the amplitude spectrum constant.
    
    start_fourier = np.fft.fft2(image_start)
    end_fourier   = np.fft.fft2(in_image)
    
    start_phase   = np.fft.fftshift(np.angle(start_fourier))
    # startAmp     = abs(startFourier);
    
    end_phase     = np.fft.fftshift(np.angle(end_fourier))
    end_amp       = np.fft.fftshift(np.abs(end_fourier))
    
    start_amp = end_amp;
    
    initial_sequence = np.stack((start_phase,end_phase), 2)
       
    # This is where I figure out the minimum phase direction.
    # I do this by chaining some trigonometry operations together.
    # D is the angle between the starting and ending phase
    # We know we want to change phase by this amount
    # We then redefine the starting phase so the interpolation always goes in
    # the correct direction
    
    D = np.squeeze(initial_sequence[:,:,0] - initial_sequence[:,:,1])
    delta = np.arctan2(np.sin(D),np.cos(D));
    initial_sequence[:,:,0] = initial_sequence[:,:,1] + delta
    
    # This is slow, but it's easy and I'm lazy.
    
    xi,yi,zi = numpy.meshgrid(x, y, interp_vals, indexing='ij')
    
    phase_sequence = interpn((x,y,z),(initial_sequence),(xi,yi,zi))
    
    phase_sequence = np.mod(phase_sequence + np.pi, 2 * np.pi) - np.pi
     
    amp_Sequence = np.tile(end_amp,[interp_vals.shape[0], 1, 1])
     
    amp_Sequence = np.transpose(amp_Sequence, (1,2,0))
   
    complex_sequence = amp_Sequence * np.exp(1j * phase_sequence)
    sequence = np.real(np.fft.ifft2(np.fft.ifftshift(complex_sequence, axes=(0,1)),axes=(0,1)))
    
    # Added specifically for the wallpaper_maker code to standardize all the images in the sequence
#    for i in range(sequence.shape[2]):
#            sequence[:, :, i] = sequence[:, :, i] - np.mean(sequence[:, :, i])
#            sequence[:, :, i] = sequence[:, :, i] / np.std(sequence[:, :, i])
#            sequence[:, :, i] = sequence[:, :, i] * 0.5 ## TO DO: MAKE THIS AN ADJUSTABLE INPUT VARIABLE
#            sequence[:, :, i] = np.clip(sequence[:, :, i], a_min=-1.0, a_max=1.0)
#            sequence[:, :, i] = (sequence[:, :, i] + 1) / 2
    return sequence

def psScramble(in_image, cmap):
    # make sure the image has a size of n*4*2**depth
    assert in_image.shape[0] % (4*2**5) == 0, ' image width not an integer multiple of 4*2**5' # hard wired for depth == 5
    assert in_image.shape[1] % (4*2**5) == 0, ' image height not an integer multiple of 4*2**5' # hard wired for depth == 5
    out_image = pss_g.synthesis(
        in_image, in_image.shape[0], in_image.shape[1], 5, 4, 7, 25) #synthesis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor, iter)
    return out_image

def next_power_2(x):
    x = x - 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x = x + 1
    return x

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def calc_radially_averaged_psd(img):
    spec = np.fft.rfft2(img)
    abs_spec = np.abs(spec)
    [X,Y]= np.meshgrid( np.arange(0,spec.shape[1]), np.fft.fftfreq(spec.shape[0])*spec.shape[0])
    [rho,phi] = cart2pol(X,Y)

    rho_ = np.round(rho).astype(np.int32)
    avg_spec = np.zeros(int(rho_.max())+1)
    for r in np.unique(rho_):
        avg_spec[int(r)] = np.mean(abs_spec[rho_==r])
    return avg_spec


def str2bool(v):
    # convert str to bool for commandline input
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    # convert str to list for commandline input
    if isinstance(v, list):
        return v
    else:
        return list(v.split(","))

def is_valid_ratio(groups, sizing, ratio):
    if sizing == 'lattice' and ratio > 0.5:
        raise SystemExit('Invalid ratio for lattice sizing. Ratio cannot be greater than 0.5')
    elif sizing == 'fr':
        for group in groups:
            if group == 'P1' and ratio > 0.5:
                raise SystemExit('Invalid ratio for fr sizing for P1. Ratio cannot be greater than 0.5')
            elif (group == 'P2' or group == 'PM' or group == 'PG' or group == 'CM') and ratio > 0.25:
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.25')
            elif (group == 'PMM' or group == 'PMG' or group == 'PGG' or group == 'CMM' or group == 'P4') and ratio > 0.125:
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.125')
            elif (group == 'P4M' or group == 'P4G') and ratio > 0.0625:
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.625')
            elif group == 'P3' and ratio > (1 / 18):
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.0555...')
            elif group == 'P3M1' and ratio > (1 / 24):
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.0416...')
            elif (group == 'P31M' or group == 'P6') and ratio > (1 / 36):
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.0277...')
            elif group == 'P6M' and ratio > (1 / 72):
                raise SystemExit('Invalid ratio for fr sizing for ' + group + '. Ratio cannot be greater than 0.0138...')

# for commandline input
if __name__ == "__main__":
    LOGGER.info('Generating Wallpapers')

    parser = argparse.ArgumentParser(
        description='Wallpaper Generator')
    parser.add_argument('--groups', '-g', default=['P1', 'P2', 'P4', 'P3', 'P6'], type=str2list,
                        help='Groups to create')
    parser.add_argument('--num_exemplars', '-n', default=10, type=int,
                        help='Number of images per group')
    parser.add_argument('--wp_size_dva', '-v', default=30.0, type=float,
                        help='Wallpaper size (visual angle)')
    parser.add_argument('--wallpaperSize', '-ws', default=500, type=int,
                        help='Side length of the wallpaper in pixels')
    parser.add_argument('--lattice_sizing', '-l', default=False, type=str2bool,
                        help='Size wallpaper as a ratio between the lattice and wallpaper size')
    parser.add_argument('--fr_sizing', '-fr', default=False, type=str2bool,
                        help='Size wallpaper as a ratio between the fundamental region and wallpaper size')
    parser.add_argument('--ratio', '-ra', default=1.0, type=float,
                        help='Size wallpaper as a ratio')
    parser.add_argument('--save_fmt', '-f', default="png", type=str,
                        help='Image save format')
    parser.add_argument('--filter_freq', '-f0fr', nargs='+',  default=[], type=str2list,
                        help='Center frequency (in cycle per degree) for dyadic bandpass filtering the fundamental region. [] does not invoke filtering. Might be extended for a multichannel filterbanks later.')
    parser.add_argument('--phase_scramble', '-phases', default="1", type=int,
                        help='Number of phases or minimum phase interpolated scrambling')
    parser.add_argument('--ps_scramble', '-ps_scramble', default=False, type=str2bool,
                        help='PS scrambling')
    parser.add_argument('--save_raw', '-r', default=False, type=str2bool,
                        help='save raw')
    parser.add_argument('--same_magnitude', '-m', default=False, type=str2bool,
                        help='New magnitude')
    parser.add_argument('--diagnostic', '-diag', default=True, type=str2bool,
                        help='Produce diagnostics for wallpapers')
    parser.add_argument('--mask', '-msk', default=True, type=str2bool,
                        help='Mask output wallpaper with circular aperture')

    args = parser.parse_args()

#    for ratio in [0.03, 0.015, 0.02]:
#    for ratio in [0.03, 0.015, 0.02]:
#        make_set(groups=['P31M'], num_exemplars=5, wp_size_dva=args.wp_size_dva,
#             wp_size_pix=args.wallpaperSize, sizing=args.sizing, ratio=ratio, filter_freqs=[1],
#             save_fmt=args.save_fmt, save_raw=True, ctrl_images='phase', ps_scramble = 5,
#             same_magnitude=False,
#             is_diagnostic=False, save_path='./wallpapers2', mask=args.mask)
#    make_set(groups=['P6'], num_exemplars=2, wp_size_dva=args.wp_size_dva,
#             wp_size_pix=args.wallpaperSize, sizing=args.sizing, ratio=ratio, filter_freqs=[1,2,4,6],
#             save_fmt=args.save_fmt, save_raw=True, ctrl_images='phase', ps_scramble = 13,
#             same_magnitude=True,
#             is_diagnostic=False, save_path='./wallpapers2', mask=args.mask)
#    make_set(groups=['P6'], num_exemplars=2, wp_size_dva=args.wp_size_dva,
#             wp_size_pix=args.wallpaperSize, sizing=args.sizing, ratio=ratio, filter_freqs=[],
#             save_fmt=args.save_fmt, save_raw=True, ctrl_images='phase', ps_scramble = 1,
#             same_magnitude=True,
#             is_diagnostic=False, save_path='./wallpapers2', mask=args.mask)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=True, phase_scramble=0, ps_scramble=True)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=512, sizing='fr', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=True, phase_scramble=0, ps_scramble=True)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=512, sizing='fr', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=True, phase_scramble=10, ps_scramble=True)

    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=True, phase_scramble=10, ps_scramble=False)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=True, phase_scramble=10, ps_scramble=False)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[], is_diagnostic=False, same_magnitude=True, phase_scramble=10, ps_scramble=False)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[1,3], is_diagnostic=False, same_magnitude=False, phase_scramble=10, ps_scramble=False)
#    make_set(groups = ['P31M'], num_exemplars=5, wp_size_dva=30, wp_size_pix=600, sizing='lattice', ratio=0.030, save_path='./wallpapers2', filter_freqs=[], is_diagnostic=False, same_magnitude=False, phase_scramble=10, ps_scramble=False)
