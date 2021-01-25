"""

The code will generate an arbitrary number of exemplars belonging to each of the 17 groups, as well as matched control exemplars using phase-scrambling and Portilla-Simoncelli scrambling.

To run the code use the following function with the available optional parameters: 

make_set(groups to create, number of images per group, wallpaper size (visual angle), distance beteween eye and wallpaper, .... 
tile area, image save format, save raw, print analysis, Portilla-Simoncelli scrambled, phase scrambled, new magnitude, color or greyscale map, debug parameters on/off)

Check function for expected data types for each argument.

"""
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

import sys
import cv2 as cv

import logging
import argparse

from scipy.stats import mode
import scipy.ndimage

from IPython.display import display, Markdown

np.set_printoptions(threshold=sys.maxsize)


SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))


def make_set(groups: list = ['P1', 'P2', 'P4', 'P3', 'P6'], num_exemplars: int = 10, wp_size_dva: float = 30.0, wp_size_pix: int = 500, lattice_sizing: bool = False,
             fr_sizing: bool = False, ratio: float = 1.0, is_dots: bool = False, filter_freq: list = [], save_fmt: str = "png", save_raw: bool = False, ctrl_images: str = 'False', same_magnitude: bool = False,
             cmap: str = "gray", is_diagnostic: bool = True, save_path: str = "", debug: bool = False):

    # save parameters
    if not save_path:
        save_path = os.path.join(os.path.expanduser('~'), 'wallpapers')
    today = datetime.today()
    time_str = today.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # define group to index mapping
    #key_set = groups

    # useful parameters for debugging
    if debug:
        #num_exemplars = 1
        #ratio = 1
        groups = ['P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG',
                   'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M']
        #key_set = ['P3']
        #wp_size_pix= 300
    key_set = ['P1', 'P2', 'PM', 'PG', 'CM', 'PMM', 'PMG', 'PGG',
                   'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M']
    value_set = np.arange(101, 101 + len(key_set), 1)
    map_group = {}
    for i in range(value_set.shape[0]):
        map_group[key_set[i]] = value_set[i]
    Groups = groups
    raw_path = ''
    if (not filter_freq):
        filter_freq = False;
        filter_freq_str = "No filtering applied"
    else:
        filter_freq_str = ','.join(str(x) for x in filter_freq)
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
    orig_wallpapers = []
    orig_wallpapers_group = []
    avg_mag = []
    mags = []
    cm = plt.get_cmap(cmap)
    for i in range(len(Groups)):
        # making regular images
        print('generating ', Groups[i])
        group = Groups[i]
        if lattice_sizing:
            n = size_lattice(ratio, wp_size_pix, group)
        elif fr_sizing:
            n = size_fundamental_region(ratio, wp_size_pix, group)
        else:
            n = size_tile(ratio, wp_size_pix, group)
        raw = []             
        for k in range(num_exemplars):
            raw = make_single(group, wp_size_pix, int(n), fr_sizing, lattice_sizing, ratio, wp_size_dva, is_diagnostic,
                                  filter_freq, fundamental_region_source_type, is_dots, cmap, save_path, k)
            group_number = map_group[group]
            if(save_raw):
                raw_path_tmp = raw_path + '/' + time_str + '_' + str(1000 * group_number + k + 1) + \
                    '_' + cmap + '.' + save_fmt
                display(Markdown(str(1000 * group_number + k + 1) +
                                 '_' + cmap + '_raw'))
                display(Image.fromarray(
                    (raw[:, :] * 255).astype(np.uint8)))
                Image.fromarray(
                    (raw[:, :] * 255).astype(np.uint8)).save(raw_path_tmp, save_fmt)
            
            # low-pass filtering + histeq
            if filter_freq:
                num_each_wallpaper = len(filter_freq)
            else:
                num_each_wallpaper = 1
            for o in range(num_each_wallpaper):
                filtered = (filter_img(raw[o], wp_size_pix))   
                orig_wallpapers_group.append(Groups[i])
                orig_wallpapers.append(filtered)
        if same_magnitude:
            for p in range(len(filter_freq)):
                for q in range(num_exemplars):
                    mags.append(np.fft.fftshift(np.abs( np.fft.fft2(orig_wallpapers[p + (q * len(filter_freq))]))))
                avg_mag.append(np.median(np.array(mags), 0))

    # image processing steps
    exemplar_index_increment = 0
    freq_index_increment = 0
    mag_index = 0
    for j in range(len(orig_wallpapers)):
        group = orig_wallpapers_group[j]
        group_number = map_group[group]
        if not is_dots:
            # replace each image's magnitude with the average   
            if freq_index_increment + exemplar_index_increment * len(filter_freq) == len(filter_freq) * num_exemplars + freq_index_increment:
                exemplar_index_increment = 0
                freq_index_increment = freq_index_increment + 1
                mag_index = mag_index + 1
            if (j % (len(filter_freq) * num_exemplars) == 0 and j != 0):
                freq_index_increment = j
            wallpaper_index = freq_index_increment + exemplar_index_increment * len(filter_freq)
            avg_raw = (replace_spectra(orig_wallpapers[wallpaper_index], use_magnitude=np.array(avg_mag[mag_index])))
            exemplar_index_increment = exemplar_index_increment + 1
            #orig_wallpapers[j] = avg_raw
            # masking the image (final step)
            masked = (mask_img(avg_raw, wp_size_pix))
            if filter_freq:
                filter_str = '_f0fr' + str(filter_freq[j % len(filter_freq)])
            else:
                filter_str = ''

            main_str = str(1000 * group_number + j + 1) + '_' + \
                cmap + filter_str

            pattern_path = "{0}/{1}_{2}.{3}".format(
                save_path, time_str, main_str, save_fmt)

            if (is_dots):
                Image.fromarray((raw[:, :]).astype(
                    np.uint32), 'RGBA').save(pattern_path, "png")
                display(Markdown(str(1000 * group_number + j + 1) +
                                 '_' + cmap))
                display(Image.fromarray(
                    (raw[:, :]).astype(np.uint32), 'RGBA'))
            else:
                Image.fromarray(
                    (masked[:, :] * 255).astype(np.uint8)).save(pattern_path, save_fmt)
                display(Markdown(str(1000 * group_number + j + 1) +
                                 '_' + cmap + filter_str))
                display(Image.fromarray((masked[:, :] * 255).astype(np.uint8)))    
                
    # generating wallpapers, saving freq. representations
    
    # scrambled image processing steps
    exemplar_index_increment = 0
    freq_index_increment = 0
    mag_index = 0
    
    for l in range(len(orig_wallpapers)):
        group = orig_wallpapers_group[l]
        # making scrambled images
        if filter_freq:
            filter_str = '_f0fr' + str(filter_freq[l % len(filter_freq)])
        else:
            filter_str = ''
           
        if (ctrl_images == 'ps'):
            # replace each image's magnitude with the average
            ps_raw = replace_spectra(orig_wallpapers[l], ctrl_images, cmap=cmap)
            #ps_filtered = (filter_img(ps_raw, wp_size_pix))
            # masking the image (final step)
            ps_masked = cm(mask_img(ps_raw, wp_size_pix))
        if (ctrl_images == 'phase'):
            # replace each image's magnitude with the average
            if freq_index_increment + exemplar_index_increment * len(filter_freq) == len(filter_freq) * num_exemplars + freq_index_increment:
                exemplar_index_increment = 0
                freq_index_increment = freq_index_increment + 1
                mag_index = mag_index + 1
            if (l % (len(filter_freq) * num_exemplars) == 0 and l != 0):
                freq_index_increment = l
            wallpaper_index = freq_index_increment + exemplar_index_increment * len(filter_freq)
            scrambled_raw = replace_spectra(orig_wallpapers[wallpaper_index], ctrl_images, cmap=cmap, use_magnitude=avg_mag[mag_index])
            exemplar_index_increment = exemplar_index_increment + 1
            #scrambled_filtered = (filter_img(scrambled_raw, wp_size_pix))
            # masking the image (final step)
            scrambled_masked = cm(mask_img(scrambled_raw, wp_size_pix))
            #Image.fromarray(np.hstack(((masked[:, :, :3] * 255).astype(np.uint8), (scrambled_masked[:, :, :3] * 255).astype(np.uint8)))).show()
        group_number = map_group[group]

        # saving averaged and scrambled images
            
        if (ctrl_images == 'phase'):
            if (filter_freq):
                scramblePath = save_path + '/' + time_str + '_' + str(1000 * (group_number + 17) + l + 1) + '_' + \
                    cmap + '_f0fr' + filter_str + '.' + save_fmt
            else:
                scramblePath = save_path + '/' + \
                    time_str + '_' + str(1000 * (group_number + 17) + l + 1) + '_' + \
                    cmap + '.' + save_fmt
            display(Markdown(str(1000 * (group_number + 17) + l + 1) +
                             '_' + cmap + filter_str))
            display(Image.fromarray(
                (scrambled_masked[:, :, :3] * 255).astype(np.uint8)))
            Image.fromarray(
                (scrambled_masked[:, :, :3] * 255).astype(np.uint8)).save(scramblePath, save_fmt)

        if (ctrl_images == 'ps'):
            if (filter_freq):
                scramblePath = save_path + '/' + time_str + '_' + str(1000 * (group_number + 34) + l + 1) + '_' + \
                    cmap + '_f0fr' + \
                    filter_str + '.' + save_fmt
            else:
                scramblePath = save_path + '/' + \
                    time_str + '_' + str(1000 * (group_number + 34) + l + 1) + '_' + \
                    cmap + '.' + save_fmt
            display(Markdown(str(1000 * (group_number + 34) + l + 1) +
                             '_' + cmap + filter_str))
            display(Image.fromarray(
                (ps_masked[:, :, :3] * 255).astype(np.uint8)))
            Image.fromarray(
                (ps_masked[:, :, :3] * 255).astype(np.uint8)).save(scramblePath, save_fmt)

            
            

        # all_in_one = cellfun(@(x,y,z) cat(2,x(1:wp_size_pix,1:wp_size_pix),y(1:wp_size_pix,1:wp_size_pix),z(1:wp_size_pix,1:wp_size_pix)),raw,avg_raw,filtered,'uni',false)

        # variables for saving in a mat file
        #symAveraged[:,i]= np.concatenate((avg_raw, scrambled_raw))
        #symFiltered[:,i]= np.concatenate((filtered, scrambled_filtered))
        #symMasked[:,i]= np.concatenate((masked, scrambled_masked))
    # save([save_path,timeStr,'.mat'],'symAveraged','symFiltered','symMasked','Groups');


def dot_texture(size, min_rad, max_rad, num_of_dots, wp_type):
    # auxiliary function for generating a dots texture for use with wallpaper generation code make_single.py

    height = 0
    width = 0
    # get correct size of dots tile based on wallpaper chosen
    if (wp_type == "P1"):
        height = size
        width = size
    elif (wp_type == "P2" or wp_type == "PM" or wp_type == "PG"):
        width = round(size / 2)
        height = 2 * width
    elif (wp_type == "PMM" or wp_type == "CM" or wp_type == "PMG" or wp_type == 'PGG' or wp_type == 'P4' or wp_type == 'P4M' or wp_type == 'P4G'):
        height = round(size / 2)
        width = height
    elif (wp_type == 'P3'):
        alpha = np.pi / 3
        s = size / math.sqrt(3 * np.tan(alpha))
        height = math.floor(s * 1.5) * 6
        width = size * 6
    elif (wp_type == 'P3M1'):
        alpha = np.pi / 3
        s = size / math.sqrt(3 * np.tan(alpha))
        height = round(s) * 10
        width = size * 10
    elif (wp_type == 'P31M' or wp_type == 'P6'):
        s = size / math.sqrt(math.sqrt(3))
        height = round(s) * 6
        width = size * 6
    elif (wp_type == 'P6M'):
        s = size / math.sqrt(math.sqrt(3))
        height = round(s / 2) * 6
        width = size * 6
    elif (wp_type == 'CMM'):
        width = round(size / 4)
        height = 2 * width

    surface = cr.ImageSurface(cr.FORMAT_ARGB32, width, height)
    ctx = cr.Context(surface)
    ctx.scale(width, height)

    pat = cr.SolidPattern(0.5, 0.5, 0.5, 1.0)

    # ctx.set_source_rgb(0, 0, 0)  # Solid color
    ctx.rectangle(0, 0, width, height)  # Rectangle(x0, y0, x1, y1)
    ctx.set_source(pat)
    ctx.fill()

    # generate random dots
    #num_of_dots = 10
    start_time = time.time()
    end_time = time.time()
    previous_dots = []
    x = 0
    while x < num_of_dots:
        is_possible_dot = False
        colour = np.linspace(0., 1., num_of_dots)
        ctx.set_source_rgb(colour[x], colour[x], colour[x])  # Solid color
        while is_possible_dot is False:
            # attempt to regenerate dots if current dots cannot all be placed
            if math.floor(end_time - start_time) % 2 == 0 and math.floor(end_time - start_time) != 0:
                pat = cr.SolidPattern(0.5, 0.5, 0.5, 1.0)
                ctx.rectangle(0, 0, width, height)  # Rectangle(x0, y0, x1, y1)
                ctx.set_source(pat)
                ctx.fill()
                x = 0
                start_time = time.time()
                end_time = time.time()
                ctx.set_source_rgb(colour[x], colour[x], colour[x])
                previous_dots = []
                print("Could not create dots with current values. Starting again.")
            radius = np.random.uniform(min_rad, max_rad)
            xc = np.random.uniform(radius, 1 - radius)
            yc = np.random.uniform(radius, 1 - radius)

            # place dots only places where it won't get cut off in wallpaper construction
            if (wp_type == 'P3'):
                xc = np.random.uniform(radius, 1 - max(0.75, radius * 2))
                yc = np.random.uniform(0.30 + radius, 1 - max(0.35, radius * 2))
            elif (wp_type == 'P4G'):
                xc = np.random.uniform(0.45 + radius, 1 - max(0.25, radius * 2))
                yc = np.random.uniform(0.45 + radius, 1 - max(0.25, radius * 2))
            elif (wp_type == 'P4M'):
                xc = np.random.uniform(0.05 + radius, 1 - max(0.05, radius * 2))
                yc = np.random.uniform(
                    0.05 + radius, 1 - max(0.7 + radius, radius * 2))
            elif (wp_type == 'P3M1'):
                xc = np.random.uniform(0.025 + radius, 1 - max(0.93, radius * 2))
                yc = np.random.uniform(0.175 + radius, 1 - max(0.35, radius * 2))
            elif (wp_type == 'P31M'):
                xc = np.random.uniform(0.025 + radius, 1 - max(0.85, radius * 2))
                yc = np.random.uniform(0.30 + radius, 1 - max(0.35, radius * 2))
            elif (wp_type == 'P6'):
                xc = np.random.uniform(0.025 + radius, 1 - max(0.93, radius * 2))
                yc = np.random.uniform(0.30 + radius, 1 - max(0.05, radius * 2))
            elif (wp_type == 'P6M'):
                xc = np.random.uniform(0.025 + radius, 1 - max(0.93, radius * 2))
                yc = np.random.uniform(0.30 + radius, 1 - max(0.05, radius * 2))

            if x == 0:
                #ctx.set_source_rgb(0, 0, 0)
                #ctx.arc(xc, yc, 0.008, 0, 2*math.pi)
                # ctx.fill()
                # ctx.set_source_rgb(0, 0, 0)  # Solid color
                # ctx.arc(xc, yc, radius, 0, 2*math.pi) #circle
                # ctx.fill()
                #previous_dots.append([xc,yc, radius])
                is_possible_dot = True
            else:
                # generate radius not touching other dots
                for y in previous_dots:
                    d = (xc - y[0])**2 + (yc - y[1])**2
                    rad_sum_sq = (radius + y[2])**2
                    if (d > rad_sum_sq):
                        is_possible_dot = True
                    else:
                        is_possible_dot = False
                        break
            # if dot is okay to be placed will generate a blob constrained in the dot
            if is_possible_dot:
                #blobs = np.random.randint(1, 15)
                blobs = 5
                previous_dots.append([xc, yc, radius])
                x = x + 1
                for i in range(blobs):
                    xc1 = np.random.uniform(xc - radius, radius + xc)
                    yc1 = np.random.uniform(yc - radius, radius + yc)
                    radius1 = radius / 2
                    ctx.arc(xc1, yc1, radius1, 0, 2 * math.pi)
                    ctx.fill()
            end_time = time.time()

    # surface.write_to_png("example.png")
    buf = surface.get_data()
    if (wp_type == 'P3' or wp_type == 'P3M1' or wp_type == 'P31M' or wp_type == 'P6' or wp_type == 'P6M'):
        result = np.ndarray(shape=(height, width), dtype=np.uint32, buffer=buf)
    else:
        result = np.ndarray(shape=(width, height), dtype=np.uint32, buffer=buf)
    return result


def new_p3(tile, is_dots):
    # Generate p3 wallpaper

    # For mag_factor, use a multiple of 3 to avoid the rounding error
    # when stacking two_tirds, one_third tiles together

    #saveStr = os.path.join(os.path.expanduser('~'),'wallpapers')
    #today = datetime.today()
    #timeStr = today.strftime("%Y%m%d_%H%M%S")
    #save_path = saveStr + timeStr;
    #pattern_path = save_path + "_P3_Start_2"  + '.' + "png"
    mag_factor = 6
    if (is_dots):
        height = tile.shape[0]
        s1 = round((height / 3))
        s = 2 * s1
        width = round(height / math.sqrt(3)) - 1
        xy = np.array([[0, 0], [s1, width], [height, width],
                       [2 * s1, 0], [0, 0]]).astype(np.uint32)
        mask = skd.polygon2mask((height, width), xy).astype(np.uint32)
        tile0 = (mask * tile[:, :width]).astype(np.uint32)

        # rotate rectangle by 120, 240 degs

        # note on 120deg rotation: 120 deg rotation of rhombus-shaped
        # texture preserves the size, so 'crop' option can be used to
        # tile size.

        tile0Im = Image.fromarray(tile0, 'I')
        tile0Im2 = Image.fromarray(tile0, 'I')
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand=False)
        tile120 = np.array(tile0Im_rot120, np.uint32)
        tile0Im_rot240 = tile0Im2.rotate(240, Image.NEAREST, expand=True)
        tile240 = np.array(tile0Im_rot240, np.uint32)

        # manually trim the tiles:

        # tile120 should have the same size as rectangle: [heigh x width]
        # tile120 = tile120(1:height, (floor(0.5*width) + 1):(floor(0.5*width) + width))

        # tile240 should have the size [s x 2*width]
        # find how much we need to cut from both sides
        diff = round(0.5 * (np.size(tile240, 1) - 2 * width))
        row_start = round(0.25 * s)
        row_end = round(0.25 * s) + s
        col_start = diff
        col_end = 2 * width + diff
        tile240 = tile240[row_start:row_end,
                          col_start:col_end].astype(np.uint32)

        # Start to pad tiles and glue them together
        # Resulting tile will have the size [3*height x 2* width]

        two_thirds1 = np.concatenate(
            (tile0, tile120), axis=1).astype(np.uint32)
        two_thirds2 = np.concatenate(
            (tile120, tile0), axis=1).astype(np.uint32)

        two_thirds = np.concatenate(
            (two_thirds1, two_thirds2)).astype(np.uint32)

        # lower half of tile240 on the top, zero-padded to [height x 2 width]
        row_start = int(0.5 * s)
        col_end = 2 * width
        one_third11 = np.concatenate(
            (tile240[row_start:, :], np.zeros((s, col_end)))).astype(np.uint32)

        # upper half of tile240 on the bottom, zero-padded to [height x 2 width]
        row_end = int(0.5 * s)
        one_third12 = np.concatenate(
            (np.zeros((s, col_end)), tile240[:row_end, :])).astype(np.uint32)

        # right half of tile240 in the middle, zero-padded to [height x 2 width]
        col_start = width
        one_third21 = np.concatenate((np.zeros(
            (s, width)), tile240[:, col_start:], np.zeros((s, width)))).astype(np.uint32)

        # left half of tile240in the middle, zero-padded to [height x 2 width]
        one_third22 = np.concatenate(
            (np.zeros((s, width)), tile240[:, :width], np.zeros((s, width)))).astype(np.uint32)

        # cat them together
        one_third1 = np.concatenate(
            (one_third11, one_third12)).astype(np.uint32)
        one_third2 = np.concatenate(
            (one_third21, one_third22), axis=1).astype(np.uint32)

        # glue everything together, shrink and replicate
        one_third = np.maximum(one_third1, one_third2).astype(np.uint32)

        # size(whole) = [3xheight 2xwidth]
        whole = np.maximum(two_thirds, one_third).astype(np.uint32)
        whole[np.where(whole == np.min(whole))] = mode(
            whole, axis=None)[0].astype(np.uint32)

        whole_im = Image.fromarray(whole, 'I')

        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        whole_im_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                                  for i in reversed(whole.shape))

        p3 = np.array(whole_im.resize(whole_im_new_size,
                                      Image.NEAREST)).astype(np.uint32)
        #Image.fromarray((whole[:, :]).astype(np.uint32), 'RGBA').save(pattern_path, "png")
        return p3
    else:

        #cm = plt.get_cmap("gray")
        #saveStr = os.getcwd() + '\\WPSet\\'
        #today = datetime.today()
        #timeStr = today.strftime("%Y%m%d_%H%M%S")
        #save_path = saveStr + timeStr;

        tile_im = Image.fromarray(tile)

        # (tuple(i * mag_factor for i in reversed(tile.shape)) to calculate the (width, height) of the image

        tile1 = np.array(tile_im.resize(
            (tuple(i * mag_factor for i in reversed(tile.shape))), Image.BICUBIC))

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
        # width = min(round(height/sqrt(3)), size(tile1, 2)) - 1

        # define rhombus-shaped mask

        xy = np.array(
            [[0, 0], [s1, width], [height, width], [2 * s1, 0], [0, 0]])

        mask = skd.polygon2mask((height, width), xy)
        tile0 = mask * tile1[:, :width]

        # rotate rectangle by 120, 240 degs

        # note on 120deg rotation: 120 deg rotation of rhombus-shaped
        # texture preserves the size, so 'crop' option can be used to
        # tile size.

        tile0Im = Image.fromarray(tile0)
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand=False)
        tile120 = np.array(tile0Im_rot120)
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand=True)
        tile240 = np.array(tile0Im_rot240)

        # manually trim the tiles:

        # tile120 should have the same size as rectangle: [heigh x width]
        # tile120 = tile120(1:height, (floor(0.5*width) + 1):(floor(0.5*width) + width))

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

        whole_im = Image.fromarray(whole)
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        whole_im_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                                  for i in reversed(whole.shape))

        p3 = np.array(whole_im.resize(whole_im_new_size, Image.BICUBIC))
        return p3


def new_p3m1(tile, is_dots):
    # Generate p3m1 wallpaper
    mag_factor = 10
    if (is_dots):
        height = tile.shape[0]
        # fundamental region is equlateral triangle with side length = height
        width = round(0.5 * height * math.sqrt(3))
        y1 = round(height / 2)
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [y1, 0], [0, 0]]
        # Create half of the mask
        # reflect and concatenate, to get the full mask:
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32)
        mask = np.concatenate(
            (mask_half, np.flipud(mask_half))).astype(np.uint32)

        tile0 = (tile[:, :width] * mask).astype(np.uint32)

        # continue to modify the tile

        # reflect and rotate
        tile1_mirror = np.fliplr(tile0).astype(np.uint32)
        tile1_mirrorIm = Image.fromarray(tile1_mirror, 'I')
        tile240_Im = tile1_mirrorIm.rotate(240, Image.NEAREST, expand=True)
        tile240 = np.array(tile240_Im, np.uint32)
        # AY: I directly cut the tiles, because trim will
        # return slightly different size

        t_r1x = np.shape(tile240)[0]
        start_row = t_r1x - height
        tile240 = tile240[start_row:, :width].astype(np.uint32)

        # AY: rotating mirrored tile(as opposed to tileR1) will cause less
        # border effects when we'll add it to two other tiles.
        tile120_Im = tile1_mirrorIm.rotate(120, Image.NEAREST, expand=True)
        tile120 = np.array(tile120_Im, np.uint32)
        tile120 = tile120[:height, :width].astype(np.uint32)

        # Assembling the tiles

        # We have 3 tiles with the same triangle rotated 0, 120 and 240
        # pad them and put them together
        zero_tile = np.zeros((y1, width))
        tile2 = np.concatenate((zero_tile, tile0, zero_tile)).astype(np.uint32)
        tile240 = np.concatenate(
            (zero_tile, zero_tile, tile240)).astype(np.uint32)
        tile120 = np.concatenate(
            (tile120, zero_tile, zero_tile)).astype(np.uint32)

        # Using max() will give us smoother edges, as opppose to sum()
        half1 = np.maximum(tile2, tile240).astype(np.uint32)
        half = np.maximum(half1, tile120).astype(np.uint32)

        # By construction, size(whole) = [4*y1 2*x1]
        whole = np.concatenate((half, np.fliplr(half)),
                               axis=1).astype(np.uint32)

        # Shifting by 2 pix (as oppose to zero), we'll smoothly glue tile together.
        # Set delta_pix value to zero and see the difference
        delta_pix = 2
        start_row1 = 3 * y1 - delta_pix
        start_row2 = 3 * y1
        end_row1 = 4 * y1 - delta_pix
        end_row2 = y1 + delta_pix
        end_row3 = 4 * y1
        end_col1 = 2 * width

        top_bit = np.concatenate((whole[start_row1:end_row1, width:end_col1],
                                  whole[start_row1:end_row1, :width]), axis=1).astype(np.uint32)
        bot_bit = np.concatenate((whole[delta_pix:end_row2, width:end_col1],
                                  whole[delta_pix:end_row2, :width]), axis=1).astype(np.uint32)

        whole[:y1, :] = np.maximum(
            whole[delta_pix:end_row2, :], top_bit).astype(np.uint32)
        whole[start_row2:end_row3, :] = np.maximum(
            whole[start_row1:end_row1, :], bot_bit).astype(np.uint32)
        #whole[np.where(whole == 0)] = np.max(whole).astype(np.uint32)

        # cutting middle piece of tile
        mid_tile = whole[y1:start_row2, :width].astype(np.uint32)
        # reflecting middle piece and glueing both pieces to the bottom
        # size(big_tile)  = [6*y1 2*x1]
        cat_mid_flip = np.concatenate(
            (mid_tile, np.fliplr(mid_tile)), axis=1).astype(np.uint32)
        big_tile = np.concatenate((whole, cat_mid_flip)).astype(np.uint32)
        big_tile[np.where(big_tile == np.min(big_tile))] = mode(
            big_tile, axis=None)[0].astype(np.uint32)

        big_tile_im = Image.fromarray(big_tile, 'I')
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        big_tile_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                                  for i in reversed(big_tile.shape))
        p3m1 = np.array(big_tile_im.resize(
            big_tile_new_size, Image.NEAREST)).astype(np.uint32)

        return p3m1

    else:
        tile_im = Image.fromarray(tile)
        # (tuple(i * mag_factor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tile_im.resize(
            (tuple(i * mag_factor for i in reversed(tile.shape))), Image.BICUBIC))
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
        tile1_mirrorIm = Image.fromarray(tile1_mirror)
        tile240_Im = tile1_mirrorIm.rotate(240, Image.BILINEAR, expand=True)
        tile240 = np.array(tile240_Im)
        # AY: I directly cut the tiles, because trim will
        # return slightly different size

        t_r1x = np.shape(tile240)[0]
        start_row = t_r1x - height
        tile240 = tile240[start_row:, :width]

        # AY: rotating mirrored tile(as opposed to tileR1) will cause less
        # border effects when we'll add it to two other tiles.
        tile120_Im = tile1_mirrorIm.rotate(120, Image.BILINEAR, expand=True)
        tile120 = np.array(tile120_Im)

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
        cat_mid_flip = np.concatenate((mid_tile, np.fliplr(mid_tile)), axis=1)
        big_tile = np.concatenate((whole, cat_mid_flip))
        big_tile_im = Image.fromarray(big_tile)
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        big_tile_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                                  for i in reversed(big_tile.shape))
        p3m1 = np.array(big_tile_im.resize(big_tile_new_size, Image.BICUBIC))
        return p3m1


def new_p31m(tile, is_dots):
    # Generate p31m wallpaper
    mag_factor = 6

    if (is_dots):

        tile0 = tile.astype(np.uint32)
        height = np.shape(tile0)[0]
        width = round(0.5 * height / math.sqrt(3))
        y1 = round(height / 2)

        # fundamental region is an isosceles triangle with angles(30, 120, 30)

        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]]

        # make half of the mask
        # reflect and concatenate, to get the full mask:
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32)

        mask = np.concatenate(
            (mask_half, np.flipud(mask_half))).astype(np.uint32)

        # size(tile0) = [height  width]
        tile0 = (mask * tile0[:, :width]).astype(np.uint32)

        # rotate the tile
        tile0_Im = Image.fromarray(tile0, 'I')
        tile120_Im = tile0_Im.rotate(120, Image.NEAREST, expand=True)
        tile120 = np.array(tile120_Im).astype(np.uint32)

        tile240_Im = tile0_Im.rotate(240, Image.NEAREST, expand=True)
        tile240 = np.array(tile240_Im).astype(np.uint32)

        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from
        tile0 = np.concatenate(
            (tile0, np.zeros((height, width * 2))), axis=1).astype(np.uint32)
        delta = np.shape(tile0)[1]

        # ideally we would've used
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);
        x120 = np.shape(tile120)[1] - delta
        y120 = np.shape(tile120)[0] - y1

        # size(tile120, tile240) = [height 3width]

        tile120 = tile120[y120:, x120:].astype(np.uint32)
        tile240 = tile240[:y1, x120:].astype(np.uint32)

        # we have 3 tiles that will comprise
        # equilateral triangle together
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile1 already padded
        tile120 = np.concatenate(
            (np.zeros((y1, width * 3)), tile120)).astype(np.uint32)

        tile240 = np.concatenate(
            (tile240, np.zeros((y1, width * 3)))).astype(np.uint32)

        # size(tri) = [height 3width]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32)
        mirror_tri = np.fliplr(tri).astype(np.uint32)

        # use shift overlap, to smooth the edges
        delta_pix = 3
        row_start = y1 - delta_pix
        row_end1 = mirror_tri.shape[0] - delta_pix
        row_end2 = y1 + delta_pix
        shifted = np.concatenate(
            (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32)
        tile2 = np.maximum(shifted, tri).astype(np.uint32)

        # size(tile3) = [height 6width]
        tile3 = np.concatenate((tile2, np.fliplr(tile2)),
                               axis=1).astype(np.uint32)
        tile3[np.where(tile3 == np.min(tile3))] = mode(
            tile3, axis=None)[0].astype(np.uint32)
        tile3_Im = Image.fromarray(tile3, 'I')
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p31m = np.array(tile3_Im.resize(
            tile3_new_size, Image.NEAREST)).astype(np.uint32)

    else:
        tile = tile.astype('float32')

        tile_im = Image.fromarray(tile)
        # (tuple(i * mag_factor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile0 = np.array(tile_im.resize(
            (tuple(i * mag_factor for i in reversed(tile.shape))), Image.BILINEAR))

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
        tile0_Im = Image.fromarray(tile0)
        tile120_Im = tile0_Im.rotate(120, Image.BILINEAR, expand=True)
        tile120 = np.array(tile120_Im)

        tile240_Im = tile0_Im.rotate(240, Image.BILINEAR, expand=True)
        tile240 = np.array(tile240_Im)

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
        tile3_Im = Image.fromarray(tile3)
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p31m = np.array(tile3_Im.resize(tile3_new_size, Image.BILINEAR))
    return p31m


def new_p6(tile, is_dots):
    # Generate p6 wallpaper
    mag_factor = 6

    if (is_dots):
        tile1 = tile.astype(np.uint32)
        height = np.shape(tile1)[0]
        width = int(round(0.5 * height * np.tan(np.pi / 6)))
        y1 = round(height / 2)

        # fundamental region is an isosceles triangle with angles(30, 120, 30)

        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]]

        # half of the mask
        # reflect and concatenate, to get the full mask:
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32)

        mask = np.concatenate(
            (mask_half, np.flipud(mask_half))).astype(np.uint32)

        # size(tile0) = [height x width]
        tile0 = (mask * tile1[:, :width]).astype(np.uint32)

        # rotate tile1
        tile0Im = Image.fromarray(tile0, 'I')
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand=True)
        tile120 = np.array(tile0Im_rot120).astype(np.uint32)
        tile0Im_rot240 = tile0Im.rotate(240, Image.NEAREST, expand=True)
        tile240 = np.array(tile0Im_rot240).astype(np.uint32)

        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from
        tile0 = np.concatenate(
            (tile0, np.zeros((height, width * 2))), axis=1).astype(np.uint32)
        delta = np.shape(tile0)[1]

        # ideally we would've used
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2
        x120 = np.shape(tile120)[1] - delta
        y120 = np.shape(tile120)[0] - y1

        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:].astype(np.uint32)
        tile240 = tile240[:y1, x120:].astype(np.uint32)

        # we have 3 tiles that will comprise
        # equilateral triangle together

        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate(
            (np.zeros((y1, width * 3)), tile120)).astype(np.uint32)
        tile240 = np.concatenate(
            (tile240, np.zeros((y1, width * 3)))).astype(np.uint32)

        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32)

        # mirror_tri = fliplr(tri); --wrong! should be (fliplr(flipud(tri)))
        triIm = Image.fromarray(tri, 'I')
        triIm_rot180 = triIm.rotate(180, expand=True)
        mirror_tri = np.array(triIm_rot180).astype(np.uint32)

        # shifw w.slight overlap,
        delta_pix = 3
        row_start = y1 - delta_pix
        row_end1 = mirror_tri.shape[0] - delta_pix
        row_end2 = y1 + delta_pix
        shifted = np.concatenate(
            (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32)

        tile2 = np.maximum(tri, shifted).astype(np.uint32)
        t2 = int(np.floor(0.5 * np.shape(tile2)[0]))

        tile2_flipped = np.concatenate(
            (tile2[t2:, :], tile2[:t2, :])).astype(np.uint32)

        # size(tile3) = [2y1 x 6x1]
        tile3 = np.concatenate((tile2, tile2_flipped),
                               axis=1).astype(np.uint32)
        tile3[np.where(tile3 == np.min(tile3))] = mode(
            tile3, axis=None)[0].astype(np.uint32)
        tile3_Im = Image.fromarray(tile3, 'I')
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p6 = np.array(tile3_Im.resize(tile3_new_size,
                                      Image.NEAREST)).astype(np.uint32)

    else:
        tile_im = Image.fromarray(tile)
        # (tuple(i * mag_factor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tile_im.resize(
            (tuple(i * mag_factor for i in reversed(tile.shape))), Image.BICUBIC))

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
        tile0Im = Image.fromarray(tile0)
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand=True)
        tile120 = np.array(tile0Im_rot120)
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand=True)
        tile240 = np.array(tile0Im_rot240)

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
        triIm = Image.fromarray(tri)
        triIm_rot180 = triIm.rotate(180, expand=True)
        mirror_tri = np.array(triIm_rot180)

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
        tile3_Im = Image.fromarray(tile3)
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p6 = np.array(tile3_Im.resize(tile3_new_size, Image.BICUBIC))
    return p6


def new_p6m(tile, is_dots):
    # Generate p6m wallpaper

    mag_factor = 6
    if (is_dots):
        tile1 = tile.astype(np.uint32)

        height = np.shape(tile1)[0]

        width = round(height / math.sqrt(3))

        # fundamental region is right triangle with angles (30, 60)

        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [height, width], [height, 0], [0, 0]]

        # half of the mask
        # reflect and concatenate, to get the full mask:
        mask = skd.polygon2mask((height, width), mask_xy).astype(np.uint32)

        # right triangle inscribed into rectangle
        tile0 = (tile1[:, :width] * mask).astype(np.uint32)

        # size(tile0) = [height x width]
        tile0 = np.concatenate((tile0, np.flipud(tile0))).astype(np.uint32)

        # rotate tile1
        tile0Im = Image.fromarray(tile0, 'I')
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand=True)
        tile120 = np.array(tile0Im_rot120).astype(np.uint32)
        tile0Im_rot240 = tile0Im.rotate(240, Image.NEAREST, expand=True)
        tile240 = np.array(tile0Im_rot240).astype(np.uint32)

        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from
        tile0 = np.concatenate(
            (tile0, np.zeros((height * 2, width * 2))), axis=1).astype(np.uint32)
        delta = np.shape(tile0)[1]

        # ideally we would've used
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);
        x120 = np.shape(tile120)[1] - delta
        y120 = np.shape(tile120)[0] - height

        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:].astype(np.uint32)
        tile240 = tile240[:height, x120:].astype(np.uint32)

        # we have 3 tiles that will comprise
        # equilateral triangle together

        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate(
            (np.zeros((height, width * 3)), tile120)).astype(np.uint32)
        tile240 = np.concatenate(
            (tile240, np.zeros((height, width * 3)))).astype(np.uint32)

        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32)
        triIm = Image.fromarray(tri, 'I')
        triIm_rot180 = triIm.rotate(180, expand=True)
        mirror_tri = np.array(triIm_rot180).astype(np.uint32)

        # shifw w.slight overlap,
        delta_pix = 3
        row_start = height - delta_pix
        row_end1 = mirror_tri.shape[0] - delta_pix
        row_end2 = height + delta_pix
        shifted = np.concatenate(
            (mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32)

        tile2 = np.maximum(tri, shifted).astype(np.uint32)
        t2 = int(np.floor(0.5 * np.shape(tile2)[0]))

        tile2_flipped = np.concatenate(
            (tile2[t2:, :], tile2[:t2, :])).astype(np.uint32)
        # size(tile3) = [2y1 x 6x1]
        tile3 = np.concatenate((tile2, tile2_flipped),
                               axis=1).astype(np.uint32)
        tile3[np.where(tile3 == np.min(tile3))] = mode(
            tile3, axis=None)[0].astype(np.uint32)
        tile3_Im = Image.fromarray(tile3, 'I')
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image

        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p6m = np.array(tile3_Im.resize(
            tile3_new_size, Image.NEAREST)).astype(np.uint32)
    else:
        tile_im = Image.fromarray(tile)
        # (tuple(i * mag_factor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tile_im.resize(
            (tuple(i * mag_factor for i in reversed(tile.shape))), Image.BICUBIC))

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
        tile0Im = Image.fromarray(tile0)
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand=True)
        tile120 = np.array(tile0Im_rot120)
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand=True)
        tile240 = np.array(tile0Im_rot240)

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
        triIm = Image.fromarray(tri)
        triIm_rot180 = triIm.rotate(180, expand=True)
        mirror_tri = np.array(triIm_rot180)

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
        tile3_Im = Image.fromarray(tile3)
        # tuple(int(np.ceil(i * (1 / mag_factor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image

        tile3_new_size = tuple(int(np.ceil(i * (1 / mag_factor)))
                               for i in reversed(tile3.shape))
        p6m = np.array(tile3_Im.resize(tile3_new_size, Image.BICUBIC))
    return p6m


def make_single(wp_type, N, n, is_fr, is_lattice, ratio, angle, is_diagnostic, filter_freq,
                fundamental_region_source_type, is_dots, cmap, save_path, k, opt_texture=None):
    #  make_single(type,N,n,opt_texture)
    # generates single wallaper group image
    # wp_type defines the wallpaper group
    # N is the size of the output image. For complex groups
    #   returned image will have size at least NxN.
    # n the size of repeating pattern for all groups.
    # is_diagnostic whether to generate diagnostic images (outlining fundamental region and lattice)
    # isSpatFreqFilt generate a spatial frequency filtered wallpaper
    # fwhm full width at half maximum of spatial frequency filter
    # whether spatialfrequency filter is lowpass or highpass
    # is_dots generate wallpaper using dots rather than random noise

    # default
    # save paths for debugging

    if fundamental_region_source_type == 'uniform_noise' and is_dots is False:
        print('uniform noise')
        texture = np.random.rand(n, n)
    elif is_dots:
        print('random dots')
        texture = dot_texture(n, 0.05, 0.05, 5, wp_type)
    elif isinstance(fundamental_region_source_type, np.ndarray):
        print('texture was passed explicitly')
        opt_texture = fundamental_region_source_type
        min_dim = np.min(np.shape(opt_texture))
        # stretch user-defined texture, if it is too small for sampling
        if min_dim < n:
            ratio_texture = round(n / min_dim)
            opt_texture = np.array(Image.resize(
                reversed((opt_texture.shape * ratio_texture)), Image.NEAREST))
        texture = opt_texture
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
            fundamental_region_filter = filter.Cosine_filter(filter_freq[i], n, angle / N * n)
        else:
            fundamental_region_filter = None
        if fundamental_region_filter:
            if isinstance(fundamental_region_filter, filter.Cosine_filter):
                texture = fundamental_region_filter.filter_image(texture)
                # scale texture into range 0...1
                texture = (texture - texture.min()) / (texture.max() - texture.min())
            else:
                raise Exception('this filter type ({}) is not implemented'.format(
                    type(fundamental_region_filter)))
        # else:
           # TODO: not exactly sure, what this lowpass filter is supposed to do. in any case:
           #       it should be adapted to this structure that separates the noise generation from the filtering
        
        try:
            # generate the wallpapers
            if wp_type == 'P0':
                p0 = np.array(Image.resize(
                    reversed((texture.shape * round(N / n))), Image.NEAREST))
                image.append(p0)
            elif wp_type == 'P1':
                width = n
                height = width
                p1 = texture[:height, :width]
                p1_image = cat_tiles(p1, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p1_image, wp_type, p1, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p1_image)
            elif wp_type == 'P2':
                height = round(n / 2)
                width = 2 * height
                start_tile = texture[:height, :width]
                tileR180 = np.rot90(start_tile, 2)
                p2 = np.concatenate((start_tile, tileR180))
                p2_image = cat_tiles(p2, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p2_image, wp_type, p2, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p2_image)
            elif wp_type == 'PM':
                height = round(n / 2)
                width = 2 * height
                start_tile = texture[:height, :width]
                mirror = np.flipud(start_tile)
                pm = np.concatenate((start_tile, mirror))
                pm_image = cat_tiles(pm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pm_image, wp_type, pm, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(pm_image)
            elif wp_type == 'PG':
                height = round(n / 2)
                width = 2 * height
                start_tile = texture[:height, :width]
                tile = np.rot90(start_tile, 3)
                glide = np.flipud(tile)
                pg = np.concatenate((tile, glide), axis=1)
                pg_image = cat_tiles(pg.T, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pg_image, wp_type, pg, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(pg_image)
            elif wp_type == 'CM':
                height = round(n / 2)
                width = height
                start_tile = texture[:height, :width]
                mirror = np.fliplr(start_tile)
                tile1 = np.concatenate((start_tile, mirror), axis=1)
                tile2 = np.concatenate((mirror, start_tile), axis=1)
                cm = np.concatenate((tile1, tile2))
                cm_image = cat_tiles(cm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(cm_image, wp_type, cm, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(cm_image)
            elif wp_type == 'PMM':
                height = round(n / 2)
                width = height
                start_tile = texture[:height, :width]
                mirror = np.fliplr(start_tile)
                concat_tmp1 = np.concatenate((start_tile, mirror), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.flipud(start_tile), np.flipud(mirror)), axis=1)
                pmm = np.concatenate((concat_tmp1, concat_tmp2))
                pmm_image = cat_tiles(pmm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pmm_image, wp_type, pmm, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(pmm_image)
            elif wp_type == 'PMG':
                height = round(n / 2)
                width = height
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                concat_tmp1 = np.concatenate(
                    (start_tile, start_tile_rot180), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.flipud(start_tile), np.fliplr(start_tile)), axis=1)
                pmg = np.concatenate((concat_tmp1, concat_tmp2))
                pmg_image = cat_tiles(pmg, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pmg_image, wp_type, pmg, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(pmg_image)
            elif wp_type == 'PGG':
                height = round(n / 2)
                width = height
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                concat_tmp1 = np.concatenate(
                    (start_tile, np.flipud(start_tile)), axis=1)
                concat_tmp2 = np.concatenate(
                    (np.fliplr(start_tile), start_tile_rot180), axis=1)
                pgg = np.concatenate((concat_tmp1, concat_tmp2))
                pgg_image = cat_tiles(pgg, N, wp_type)
                if (is_diagnostic):
                    diagnostic(pgg_image, wp_type, pgg, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(pgg_image)
            elif wp_type == 'CMM':
                height = round(n / 4)
                width = 2 * height
                start_tile = texture[:height, :width]
                start_tile_rot180 = np.rot90(start_tile, 2)
                tile1 = np.concatenate((start_tile, start_tile_rot180))
                tile2 = np.flipud(tile1)
                concat_tmp1 = np.concatenate((tile1, tile2), axis=1)
                concat_tmp2 = np.concatenate((tile2, tile1), axis=1)
                cmm = np.concatenate((concat_tmp1, concat_tmp2))
                cmm_image = cat_tiles(cmm, N, wp_type)
                if (is_diagnostic):
                    diagnostic(cmm_image, wp_type, cmm, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(cmm_image)
            elif wp_type == 'P4':
                height = round(n / 2)
                width = height
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
                    diagnostic(p4_image, wp_type, p4, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p4_image)
            elif wp_type == 'P4M':
                height = round(n / 2)
                width = height
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
                    diagnostic(p4m_image, wp_type, p4m, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p4m_image)
            elif wp_type == 'P4G':
                height = round(n / 2)
                width = height
                if (is_dots):
                    start_tile = texture[:height, :width].astype(np.uint32)
                    xy = np.array(
                        [[0, 0], [width, 0], [width, height], [0, 0]]).astype(np.uint32)
                    mask = skd.polygon2mask((height, width), xy).astype(np.uint32)
                    tile1 = (mask.astype(np.uint32) *
                             start_tile.astype(np.uint32)).astype(np.uint32)
                    tile1 = start_tile - tile1
                    tile2 = np.fliplr(tile1).astype(np.uint32)
                    tile2 = np.rot90(tile2, 1).astype(np.uint32)
                    tile = np.maximum(tile1, tile2).astype(np.uint32)
                    tile_rot90 = np.rot90(tile, 1).astype(np.uint32)
                    tile_rot180 = np.rot90(tile, 2).astype(np.uint32)
                    tile_rot270 = np.rot90(tile, 3).astype(np.uint32)
                    concat_tmp1 = np.concatenate(
                        (tile_rot270, tile_rot180), axis=1).astype(np.uint32)
                    concat_tmp2 = np.concatenate(
                        (tile, tile_rot90), axis=1).astype(np.uint32)
                    p4g = np.concatenate(
                        (concat_tmp1, concat_tmp2)).astype(np.uint32)
                else:
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
                    diagnostic(p4g_image, wp_type, p4g, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p4g_image)
            elif wp_type == 'P3':
                alpha = np.pi / 3
                s = n / math.sqrt(3 * np.tan(alpha))
                height = math.floor(s * 1.5)
    
                start_tile = texture[:height, :]
                if (is_dots):
                    p3 = new_p3(texture, is_dots)
                else:
                    p3 = new_p3(start_tile, is_dots)
                p3_image = cat_tiles(p3, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p3_image, wp_type, p3, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p3_image)
            elif wp_type == 'P3M1':
                alpha = np.pi / 3
                s = n / math.sqrt(3 * np.tan(alpha))
                height = round(s)
                start_tile = texture[:height, :]
                if (is_dots):
                    p3m1 = new_p3m1(texture, is_dots)
                else:
                    p3m1 = new_p3m1(start_tile, is_dots)
                p3m1_image = cat_tiles(p3m1, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p3m1_image, wp_type, p3m1, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p3m1_image)
            elif wp_type == 'P31M':
                s = n / math.sqrt(math.sqrt(3))
                height = round(s)
                start_tile = texture[:height, :]
                if (is_dots):
                    p31m = new_p31m(texture, is_dots)
                else:
                    p31m = new_p31m(start_tile, is_dots)
    
                # ugly trick
                p31m_1 = np.fliplr(np.transpose(p31m))
                p31m_image = cat_tiles(p31m_1, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p31m_image, wp_type, p31m_1, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p31m_image)
            elif wp_type == 'P6':
                s = n / math.sqrt(math.sqrt(3))
                height = round(s)
                start_tile = texture[:height, :]
                if (is_dots):
                    p6 = new_p6(texture, is_dots)
                else:
                    p6 = new_p6(start_tile, is_dots)
                p6_image = cat_tiles(p6, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p6_image, wp_type, p6, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p6_image)
            elif wp_type == 'P6M':
                s = n / math.sqrt(math.sqrt(3))
                height = round(s / 2)
                start_tile = texture[:height, :]
                if (is_dots):
                    p6m = new_p6m(texture, is_dots)
                else:
                    p6m = new_p6m(start_tile, is_dots)
                p6m_image = cat_tiles(p6m, N, wp_type)
                if (is_diagnostic):
                    diagnostic(p6m_image, wp_type, p6m, is_fr, is_lattice,
                               N, ratio, cmap, is_dots, save_path, k)
                image.append(p6m_image)
            else:
                warnings.warn(
                    'Unexpected Wallpaper Group type. Returning random noise.', UserWarning)
                image = np.matlib.repmat(texture, [np.ceil(N / n),  np.ceil(N / n)])
                return image
        except Exception as err:
            print('new_SymmetricNoise:Error making ' + wp_type)
            print(err.args)
    return image


def cat_tiles(tile, N, wp_type):
    # disp tile square
    sq = np.shape(tile)[0] * np.shape(tile)[1]
    print(wp_type + ' area of tile = ', sq)

    # write tile
    #tile_im = Image.fromarray(tile)
    #tile_im.save('~/Documents/PYTHON/tiles/' + wp_type + '_tile.jpeg', 'JPEG')

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
    img = numpy.matlib.repmat(tile, row - 1, col - 1)

    row = math.floor(img.shape[0] + tile.shape[0] *
                     ((1 + (N / tile.shape[0])) - dN[0]))
    col = math.floor(img.shape[1] + tile.shape[1] *
                     ((1 + (N / tile.shape[1])) - dN[1]))
    if (math.floor(img.shape[0] + tile.shape[0] * ((1 + (N / tile.shape[0])) - dN[0])) % 2 != 0):
        row = row + 1

    if (math.floor(img.shape[1] + tile.shape[1] * ((1 + (N / tile.shape[1])) - dN[1])) % 2 != 0):
        col = col + 1

    img_final = np.zeros((row, col))

    # centers the evenly created tile and then even distributes the rest of the tile around the border s.t. the total size of the wallpaper = the desired input size of the wallpaper
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2),
              math.ceil((img_final.shape[1] - img.shape[1]) / 2): img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[:, :]
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2),
              : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[:, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):]
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2),
              img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):] = img[:, : math.ceil((img_final.shape[1] - img.shape[1]) / 2)]
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), math.ceil((img_final.shape[1] - img.shape[1]) / 2): img_final.shape[1] -
              math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, :]
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, math.ceil((img_final.shape[1] - img.shape[1]) / 2)
                                             : img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), :]
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[img.shape[0] -
                                                                                                                                math.ceil((img_final.shape[0] - img.shape[0]) / 2):, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):]
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, : math.ceil((img_final.shape[1] - img.shape[1]) / 2)
              ] = img[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):]
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, img_final.shape[1] - math.ceil(
        (img_final.shape[1] - img.shape[1]) / 2):] = img[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), :math.ceil((img_final.shape[1] - img.shape[1]) / 2)]
    img_final[:math.ceil((img_final.shape[0] - img.shape[0]) / 2),  img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2)              :] = img[img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, :math.ceil((img_final.shape[1] - img.shape[1]) / 2)]

    return img_final


def diagcat_tiles(tile, N, diag_tile, wp_type):
    # Create diagnostic wallpaper
    # resize tile to ensure it will fit wallpaper size properly
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
    if (wp_type == 'P31M' or wp_type == 'P3M1' or wp_type == 'P6' or wp_type == 'P6M'):
            img = np.tile(tile, (1 + (math.floor(row / 20)),
                         1 + (math.floor(col / 20)), 1))
    else:
        img = np.tile(tile, (1 + (math.floor(row / 2)),
                             1 + (math.floor(col / 2)), 1))
    if (diag_tile.shape[0] % 2 != 0):
        diag_tile = diag_tile[:diag_tile.shape[0] - 1, :]
    if (diag_tile.shape[1] % 2 != 0):
        diag_tile = diag_tile[:, :diag_tile.shape[1] - 1]
    img = np.rot90(img, 1)
    diag_tile = np.rot90(diag_tile, 1)
    img[:diag_tile.shape[0], :diag_tile.shape[1], :] = diag_tile[:, :, :]
    return img


def diagnostic(img, wp_type, tile, is_fr, is_lattice, N, ratio, cmap, is_dots, save_path, k):
    # function to take care of all diagnostic tasks related to the wallpaper generation
    # img is the full wallpaper
    # wp_type is the wallpaper type
    # tile is a single tile of the wallpaper
    # is_fr is if the wallpaper is sized as a ratio of the fundamental region
    # is_lattice is if the wallpaper is sized as a ratio of the lattice
    # N is the overall size of the wallpaper
    # ratio is the ratio of the FR/lattice sizing

    if not is_dots:
        tile = np.array(tile * 255, dtype=np.uint8)
        tile[:, :] = cv.equalizeHist(tile[:, :])

    #img = np.array(img * 255, dtype=np.uint8);
    #img[:,:] = cv.equalizeHist(img[:,:])

    if (wp_type == 'P1'):
        # rgb(47,79,79)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1])):.1f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - tile.shape[0] * tile.shape[1]) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)

        draw.rectangle((0, 0, tile.shape[0], tile.shape[1]), outline=(
            255, 255, 0), width=2)
        dia_lat_im.save(diag_path1, "png")
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), fill=(
            47, 79, 79, 125), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P2'):
        # rgb(128,0,0)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - (tile.shape[0] * tile.shape[1]) / 2) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0], tile.shape[1]), fill=(
            128, 0, 0, 125), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.polygon(
            ((2, -1), (-1, 3), (2, 8), (5, 3), (2, -1)), fill=(128, 0, 0, 125))
        alpha_mask__rec_draw.line(
            ((1, -2), (-1, 4), (3, 9), (6, 4), (1, -2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, -1), (tile.shape[0] + 1, 3), (
            tile.shape[0] - 2, 8), (tile.shape[0] - 5, 3), ((tile.shape[0] - 2, -1))), fill=(128, 0, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, -2), (tile.shape[0] + 1, 4), (tile.shape[0] - 3, 9),
                                   (tile.shape[0] - 6, 4), (tile.shape[0] - 1, -2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, tile.shape[1] + 1), (tile.shape[0] + 1, tile.shape[1] - 3), (tile.shape[0] - 2,
                                                                                                                       tile.shape[1] - 8), (tile.shape[0] - 5, tile.shape[1] - 3), ((tile.shape[0] - 2, tile.shape[1] + 1))), fill=(128, 0, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, tile.shape[1] + 2), (tile.shape[0] + 1, tile.shape[1] - 4), (tile.shape[0] - 3,
                                                                                                                    tile.shape[1] - 9), (tile.shape[0] - 6, tile.shape[1] - 4), (tile.shape[0] - 1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((2, tile.shape[1] + 1), (-1, tile.shape[1] - 3), (
            2, tile.shape[1] - 8), (5, tile.shape[1] - 3), (2, tile.shape[1] + 1)), fill=(128, 0, 0, 125))
        alpha_mask__rec_draw.line(((1, tile.shape[1] + 2), (-1, tile.shape[1] - 4), (3, tile.shape[1] - 9),
                                   (6, tile.shape[1] - 4), (1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 2), (tile.shape[0] / 2,
                                                                                                                               tile.shape[1] / 2 - 2), (tile.shape[0] / 2 + 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 2)), fill=(128, 0, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 2 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] /
                                                                                                                            2 - 3), (tile.shape[0] / 2 + 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 3)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] / 2), 4), 4, 345, fill=(128, 0, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 4, tile.shape[1] / 2), 4), 4, 345, fill=(128, 0, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 4), 4), 4, 15, fill=(128, 0, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 4), 4), 4, 15, fill=(128, 0, 0, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'PM'):
        # rgb(0,128,0)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0], tile.shape[1]), fill=(
            0, 128, 0, 125), outline=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'PG'):
        # rgb(127,0,127)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle((tile.shape[0] / 2, 0, tile.shape[0], tile.shape[1]), fill=(
            127, 0, 127, 125), outline=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'CM'):
        # rgb(143,188,143)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cma = plt.get_cmap("gray")
        tile_cm = cma(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]),
                   (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve")
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2)), fill=(143, 188, 143, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (
            tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0)), fill=(255, 255, 0, 255), width=2, joint="curve")
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'PMM'):
        # rgb(255,69,0)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.polygon(
            ((2, -1), (-1, 3), (2, 8), (5, 3), (2, -1)), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.line(
            ((1, -2), (-1, 4), (3, 9), (6, 4), (1, -2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, -1), (tile.shape[0] + 1, 3), (tile.shape[0] - 2, 8),
                                      (tile.shape[0] - 5, 3), ((tile.shape[0] - 2, -1))), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, -2), (tile.shape[0] + 1, 4), (tile.shape[0] - 3, 9),
                                   (tile.shape[0] - 6, 4), (tile.shape[0] - 1, -2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, tile.shape[1] + 1), (tile.shape[0] + 1, tile.shape[1] - 3), (tile.shape[0] - 2,
                                                                                                                       tile.shape[1] - 8), (tile.shape[0] - 5, tile.shape[1] - 3), ((tile.shape[0] - 2, tile.shape[1] + 1))), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, tile.shape[1] + 2), (tile.shape[0] + 1, tile.shape[1] - 4), (tile.shape[0] - 3,
                                                                                                                    tile.shape[1] - 9), (tile.shape[0] - 6, tile.shape[1] - 4), (tile.shape[0] - 1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((2, tile.shape[1] + 1), (-1, tile.shape[1] - 3), (
            2, tile.shape[1] - 8), (5, tile.shape[1] - 3), (2, tile.shape[1] + 1)), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.line(((1, tile.shape[1] + 2), (-1, tile.shape[1] - 4), (3, tile.shape[1] - 9),
                                   (6, tile.shape[1] - 4), (1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 2), (tile.shape[0] / 2,
                                                                                                                               tile.shape[1] / 2 - 2), (tile.shape[0] / 2 + 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 2)), fill=(255, 69, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 2 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] /
                                                                                                                            2 - 3), (tile.shape[0] / 2 + 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 3)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] / 2), 4), 4, 345, fill=(255, 69, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 4, tile.shape[1] / 2), 4), 4, 345, fill=(255, 69, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 4), 4), 4, 15, fill=(255, 69, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 4), 4), 4, 15, fill=(255, 69, 0, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'PMG'):
        # rgb(255,165,0)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.polygon(((1, tile.shape[1] / 4), (3, tile.shape[1] / 4 - 5), (
            6, tile.shape[1] / 4), (3, tile.shape[1] / 4 + 5), (1, tile.shape[1] / 4)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 4), (4, tile.shape[1] / 4 - 6), (7, tile.shape[1] / 4),
                                   (4, tile.shape[1] / 4 + 6), (0, tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((1, tile.shape[1] - tile.shape[1] / 4), (3, tile.shape[1] - tile.shape[1] / 4 - 5), (6, tile.shape[1] -
                                                                                                                           tile.shape[1] / 4), (3, tile.shape[1] - tile.shape[1] / 4 + 5), (1, tile.shape[1] - tile.shape[1] / 4)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((0, tile.shape[1] - tile.shape[1] / 4), (4, tile.shape[1] - tile.shape[1] / 4 - 6), (7, tile.shape[1] -
                                                                                                                        tile.shape[1] / 4), (4, tile.shape[1] - tile.shape[1] / 4 + 6), (0, tile.shape[1] - tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 4 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 4 - 1), (tile.shape[0] / 2,
                                                                                                                                   tile.shape[1] / 4 - 4), (tile.shape[0] / 2 + 5, tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] / 4 + 2)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 4 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] /
                                                                                                                                4 - 5), (tile.shape[0] / 2 + 6, tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] / 4 + 3)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] - tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] -
                                                                                                                                                                   tile.shape[1] / 4 - 4), (tile.shape[0] / 2 + 5, tile.shape[1] - tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 2)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] - tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] -
                                                                                                                                                                tile.shape[1] / 4 - 5), (tile.shape[0] / 2 + 6, tile.shape[1] - tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 3)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 1, tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] / 4 - 5), (tile.shape[0] - 6,
                                                                                                                           tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] / 4 + 5), (tile.shape[0] - 1, tile.shape[1] / 4)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] / 4 - 6), (tile.shape[0] - 7,
                                                                                                                    tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] / 4 + 6), (tile.shape[0], tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 1, tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] - tile.shape[1] / 4 - 5), (tile.shape[0] - 6, tile.shape[1] -
                                                                                                                                                           tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] - tile.shape[1] / 4 + 5), (tile.shape[0] - 1, tile.shape[1] - tile.shape[1] / 4)), fill=(255, 165, 0, 125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] - tile.shape[1] / 4 - 6), (tile.shape[0] - 7, tile.shape[1] -
                                                                                                                                                    tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] - tile.shape[1] / 4 + 6), (tile.shape[0], tile.shape[1] - tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1)

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'PGG'):
        # rgb(189,183,107)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2)), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0],
                                                                                                tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((4, 4), 6), 4, 45, fill=(189, 183, 107, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] - 5), 6), 4, 45, fill=(189, 183, 107, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, tile.shape[1] - 5), 6), 4, 45, fill=(189, 183, 107, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, 4), 6), 4, 45, fill=(189, 183, 107, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 45, fill=(189, 183, 107, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (4, tile.shape[1] / 2 - 3), (
            10, tile.shape[1] / 2), (4, tile.shape[1] / 2 + 3), (0, tile.shape[1] / 2)), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (5, tile.shape[1] / 2 - 4), (11, tile.shape[1] / 2),
                                   (5, tile.shape[1] / 2 + 4), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, 0), (tile.shape[0] / 2 - 5, 3), (tile.shape[0] / 2, 6),
                                      (tile.shape[0] / 2 + 5, 3), (tile.shape[0] / 2, 0)), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2 - 6, 4), (tile.shape[0] / 2, 7),
                                   (tile.shape[0] / 2 + 6, 4), (tile.shape[0] / 2, 0)), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1]), (tile.shape[0] / 2 - 5, tile.shape[1] - 3), (tile.shape[0] / 2,
                                                                                                                       tile.shape[1] - 6), (tile.shape[0] / 2 + 5, tile.shape[1] - 3), (tile.shape[0] / 2, tile.shape[1])), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1]), (tile.shape[0] / 2 - 6, tile.shape[1] - 4), (tile.shape[0] / 2,
                                                                                                                    tile.shape[1] - 7), (tile.shape[0] / 2 + 6, tile.shape[1] - 4), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=1)
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] / 2), (tile.shape[0] - 4, tile.shape[1] / 2 - 3), (tile.shape[0] - 10,
                                                                                                                       tile.shape[1] / 2), (tile.shape[0] - 4, tile.shape[1] / 2 + 3), (tile.shape[0], tile.shape[1] / 2)), fill=(189, 183, 107, 125))
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] / 2), (tile.shape[0] - 5, tile.shape[1] / 2 - 4), (tile.shape[0] - 11,
                                                                                                                    tile.shape[1] / 2), (tile.shape[0] - 5, tile.shape[1] / 2 + 4), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=1)

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'CMM'):
        # rgb(0,0,205)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]),
                   (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve")
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1])), fill=(0, 0, 205, 125))
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[1] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0],
                                                                                                tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 30, fill=(0, 0, 205, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((6, tile.shape[1] / 2), 6), 4, 345, fill=(0, 0, 205, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 6, tile.shape[1] / 2), 6), 4, 345, fill=(0, 0, 205, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 6), 6), 4, 345, fill=(0, 0, 205, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 6), 6), 4, 345, fill=(0, 0, 205, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P4'):
        # rgb(124,252,0)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.rectangle(
            (0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(124, 252, 0, 125))
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((4, 4), 6), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] - 5), 6), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, tile.shape[1] - 5), 6), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, 4), 6), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] / 2), 4), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 4, tile.shape[1] / 2), 4), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 4), 4), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 4), 4), 4, 45, fill=(124, 252, 0, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P4M'):
        # rgb(0,250,154)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0), width=2)
        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2), (
            tile.shape[0] / 2, tile.shape[1]), (0, tile.shape[1])), fill=(0, 250, 154, 125))
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((0, 0), (tile.shape[0], tile.shape[1])), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[0], 0), (0, tile.shape[1])), fill=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((4, 4), 6), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] - 5), 6), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, tile.shape[1] - 5), 6), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, 4), 6), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] / 2), 4), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 4, tile.shape[1] / 2), 4), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 4), 4), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 4), 4), 4, 45, fill=(0, 250, 154, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P4G'):
        # rgb(65,105,225)
        if(is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]),
                   (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve")

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(
            ((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1])), fill=(65, 105, 225, 125))
        alpha_mask__rec_draw.line(
            ((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(
            ((tile.shape[1] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0],
                                                                                                tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2)
        alpha_mask__rec_draw.rectangle(
            (0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255, 255, 0, 255), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((4, 4), 6), 4, 0, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((4, tile.shape[1] - 5), 6), 4, 0, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, tile.shape[1] - 5), 6), 4, 0, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 5, 4), 6), 4, 0, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((6, tile.shape[1] / 2), 6), 4, 45, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] - 6, tile.shape[1] / 2), 6), 4, 45, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[0] / 2, tile.shape[1] - 6), 6), 4, 45, fill=(65, 105, 225, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, 6), 6), 4, 45, fill=(65, 105, 225, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P3'):
        # rgb(233,150,122)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 18):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 18)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2),
                   (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2,
                                                                                                    tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=3)
        alpha_mask__rec_draw.polygon(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(233, 150, 122, 125))

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(233, 150, 122, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(233, 150, 122, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 0, fill=(233, 150, 122, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 60, fill=(233, 150, 122, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] - 5, 3), 5), 3, 210, fill=(233, 150, 122, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(233, 150, 122, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P3M1'):
        # rgb(0,191,255)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2),
                   (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(0, 191, 255, 125))
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2,
                                                                                                    tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] / 1.5, tile.shape[0] / 3), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(0, 191, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(0, 191, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 0, fill=(0, 191, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 60, fill=(0, 191, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] - 5, 3), 5), 3, 210, fill=(0, 191, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(0, 191, 255, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P31M'):
        # rgb(255,0,255)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2),
                   (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), (
            tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(255, 0, 255, 125))
        alpha_mask__rec_draw.line(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), (
            tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2,
                                                                                                    tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((((tile.shape[1] / 2, tile.shape[0] / 6), ((
            tile.shape[1] - 1), tile.shape[0] / 3))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(
            (((tile.shape[1] - 1, 0), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75))), fill=(255, 255, 0), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(255, 0, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(255, 0, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 30, fill=(255, 0, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 30, fill=(255, 0, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] - 5, 3), 5), 3, 210, fill=(255, 0, 255, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            ((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(255, 0, 255, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P6'):
        # rgb(221,160,221)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2),
                   (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2)

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(221, 160, 221, 125))
        alpha_mask__rec_draw.line(((tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2,
                                                                                                                                      tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((((tile.shape[1] - ((tile.shape[1] - 1) / 3)), tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] - ((tile.shape[1] - 1) / 5.75), (tile.shape[0] / 1.25)), (tile.shape[1] - ((tile.shape[1] - 1) / 3), tile.shape[0] - 1)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), tile.shape[1] - (
            tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), tile.shape[1] - (
            tile.shape[1] / 6), tile.shape[0] - (tile.shape[0] / 2))), fill=(255, 255, 0), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5), 5), 3, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - (
            tile.shape[0] / 2), 3), 4, 45, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] / 2, (tile.shape[0] / 2), 5), 6, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2), 5), 6, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - 5, tile.shape[0] - 5, 5), 6, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 5, 5), 6, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, (
            tile.shape[0] / 1.25), 5), 3, 0, fill=(221, 160, 221, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 3, 3), 4, 45, fill=(221, 160, 221, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)
        display(Markdown('Fundamental Region for ' + wp_type))

    elif (wp_type == 'P6M'):
        # rgb(255,20,147)
        if (is_fr):
            print('Area of Fundamental Region of ' + wp_type +
                  f' =  {((tile.shape[0] * tile.shape[1]) / 72):.2f}')
            print('Area of Fundamental Region of ' +
                  wp_type + ' should be = ', (N**2 * ratio))
            print(
                f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 72)) / (N**2 * ratio)) * 100):.2f}%')
        diag_path1 = save_path + "_DIAGNOSTIC_LATTICE_" + wp_type + '.' + "png"
        cm = plt.get_cmap("gray")
        tile_cm = cm(tile)
        if(is_dots):
            dia_lat_im = Image.fromarray(
                (tile[:, :]).astype(np.uint32), 'RGBA')
            draw = ImageDraw.Draw(dia_lat_im, 'RGBA')
        else:
            dia_lat_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(dia_lat_im)
        draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2),
                   (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2)

        diag_path2 = save_path + "_DIAGNOSTIC_FR_" + wp_type + '.' + "png"
        if(is_dots):
            dia_fr_im = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA')
        else:
            dia_fr_im = Image.fromarray(
                (tile_cm[:, :, :] * 255).astype(np.uint8))
        alpha_mask_rec = Image.new('RGBA', dia_fr_im.size, (0, 0, 0, 0))
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec)
        alpha_mask__rec_draw.polygon(((tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255, 20, 147, 125))
        alpha_mask__rec_draw.line((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - (tile.shape[0] / 2), ((
            tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2,
                                                                                                                                      tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((((tile.shape[1] - ((tile.shape[1] - 1) / 3)), tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2)), (tile.shape[1] - ((tile.shape[1] - 1) / 5.75), (tile.shape[0] / 1.25)), (tile.shape[1] - ((tile.shape[1] - 1) / 3), tile.shape[0] - 1)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), tile.shape[1] - (
            tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5), tile.shape[1] - (
            tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line(((tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 1), tile.shape[1] - (
            tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2)
        alpha_mask__rec_draw.line((((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), tile.shape[1] - (
            tile.shape[1] / 6), tile.shape[0] - (tile.shape[0] / 2))), fill=(255, 255, 0), width=2)

        # symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5), 5), 3, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - (
            tile.shape[0] / 2), 3), 4, 45, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] / 2, (tile.shape[0] / 2), 5), 6, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (
            tile.shape[0] / 2), 5), 6, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - 5, tile.shape[0] - 5, 5), 6, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 5, 5), 6, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, (
            tile.shape[0] / 1.25), 5), 3, 0, fill=(255, 20, 147, 125), outline=(255, 255, 0))
        alpha_mask__rec_draw.regular_polygon(
            (tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 3, 3), 4, 45, fill=(255, 20, 147, 125), outline=(255, 255, 0))

        dia_fr_im = Image.alpha_composite(dia_fr_im, alpha_mask_rec)

        display(Markdown('Fundamental Region for ' + wp_type))

    dia_fr_im2 = Image.fromarray((tile_cm[:, :, :] * 255).astype(np.uint8))
    alpha_mask_rec2 = Image.new('RGBA', dia_fr_im2.size, (0, 0, 0, 0))
    alpha_mask__rec2_draw = ImageDraw.Draw(alpha_mask_rec2)
    dia_fr_im2 = Image.alpha_composite(dia_fr_im2, alpha_mask_rec2)

    diag_wallpaper = diagcat_tiles(np.array(dia_fr_im2).astype(
        np.uint32), N, np.array(dia_fr_im).astype(np.uint32), wp_type)

    if is_dots:
        pattern_path = save_path + '/' + wp_type + '_FundamentalRegion_' + str(k + 1) + '.' + "png"
        Image.fromarray((diag_wallpaper[:, :]).astype(
            np.uint32), 'RGBA').save(pattern_path, "png")
        display(Image.fromarray(
            (diag_wallpaper[:, :]).astype(np.uint32), 'RGBA'))
    else:
        display(Image.fromarray((diag_wallpaper[:, :]).astype(np.uint8)))
    if not is_dots:
        pattern_path = save_path + '/' + wp_type + '_FundamentalRegion_' + str(k + 1) + '.' + "png"
        Image.fromarray((diag_wallpaper[:, :]).astype(
            np.uint8)).save(pattern_path, "png")
        # diagnostic plots
        logging.getLogger('matplotlib.font_manager').disabled = True
        pattern_path = save_path + '/' + wp_type + '_diagnostic_' + str(k + 1) + '.' + "png"
        hidx_0 = int(img.shape[0] * (1 / 3))
        hidx_1 = int(img.shape[0] / 2)
        hidx_2 = int(img.shape[0] * (2 / 3))
        I = np.dstack([np.rot90(img, 1), np.rot90(img, 1), np.rot90(img, 1)])
        I[hidx_0 - 2:hidx_0 + 2, :] = np.array([1, 0, 0])
        I[hidx_1 - 2:hidx_1 + 2, :] = np.array([0, 1, 0])
        I[hidx_2 - 2:hidx_2 + 2, :] = np.array([0, 0, 1])
        cm = plt.get_cmap("gray")
        cm(I)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 30))

        ax1.imshow(I)
        ax1.set_title(wp_type + ' diagnostic image 1')
        ax1.set(adjustable='box', aspect='auto')

        ax2.plot(img[hidx_0, :], c=[1, 0, 0])
        ax2.plot(img[hidx_1, :], c=[0, 1, 0])
        ax2.plot(img[hidx_2, :], c=[0, 0, 1])
        ax2.set_title('Sample values along the horizontal lines {} {} and {}'.format(
            hidx_0, hidx_1, hidx_2))

        bins = np.linspace(0, 1, 100)
        ax3.hist(img[hidx_0, :], bins, color=[1, 0, 0])
        ax3.hist(img[hidx_1, :], bins, color=[0, 1, 0])
        ax3.hist(img[hidx_2, :], bins, color=[0, 0, 1])
        ax3.set_title('Frequency of sample values across the horizontal lines')

        plt.show()
        fig.savefig(pattern_path)

# old cat function
# def cat_tiles(tile, N, wp_type):
#     #disp tile square
#     sq = np.shape(tile)[0] * np.shape(tile)[1]
#     print(wp_type + ' area of tile = ', sq);

#     #write tile
#     #tile_im = Image.fromarray(tile)
#     #tile_im.save('~/Documents/PYTHON/tiles/' + wp_type + '_tile.jpeg', 'JPEG')
#     dN = tuple((1 + math.floor(N / ti)) for ti in np.shape(tile))
#     #dN = 1 + math.floor(N / np.shape(tile))
#     img = numpy.matlib.repmat(tile, dN[0], dN[1])
#     return img

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
        inImg, low_pass, mode='constant').transpose()

    # histeq
    # changed to inImg from image to stop low pass
    image = np.array(image * 255, dtype=np.uint8)
    image[:, :] = cv.equalizeHist(image[:, :])

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

# replace spectra


def replace_spectra(in_image, ctrl_images='False', use_magnitude=np.array([]), cmap="gray"):

    in_image = (in_image / np.max(in_image)) * 2.0 - 1.0

    if ctrl_images == 'ps':
        # Portilla-Simoncelli scrambling
        out_image = psScramble(in_image, cmap)
    else:
        in_spectrum = np.fft.fft2( in_image )
    
        phase = np.fft.fftshift(np.angle(in_spectrum))
        mag = np.fft.fftshift(np.abs(in_spectrum))
    
        # phase scrambling
        if ctrl_images == 'phase':
            rand_img = np.random.rand(in_image.shape[0], in_image.shape[1])
            rand_img = (rand_img / np.max(rand_img)) * 2.0 - 1.0
            rand_phase = np.fft.fft2(rand_img)
            phase = np.fft.fftshift(np.angle(rand_phase))
            # NO NEED TO RANDOMIZE THE PHASE IF YOU ARE USING RANDOM PHASE
            #rng = np.random.default_rng()
            #[rng.shuffle(x) for x in phase]

        # use new magnitude instead
        if(use_magnitude.size != 0):
            mag = use_magnitude

        cmplx_im = mag * np.exp(1j * phase)

        # get the real parts and then take the absolute value of the real parts as this is the closest solution to be found to emulate matlab's ifft2 "symmetric" parameter
        # out_image = np.abs(np.real(np.fft.ifft2(cmplx_im)))
        # the above does not seem to work, instead use code below
        # cribbed from https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html
        out_image = np.real(np.fft.ifft2(np.fft.ifftshift(cmplx_im)))

    # standardize image
    out_image = out_image - np.mean(out_image)
    out_image = out_image / np.std(out_image)
    out_image = out_image * 0.5 ## TO DO: MAKE THIS AN ADJUSTABLE INPUT VARIABLE
    out_image = np.clip(out_image, a_min=-1.0, a_max=1.0)

    out_image = (out_image + 1) / 2 

    return out_image

def psScramble(in_image, cmap):
    image_tmp = Image.fromarray(in_image)
    # resize image to nearest power of 2 to make use of the steerable pyramid for PS
    new_size = next_power_2(in_image.shape[0])
    image_tmp = image_tmp.resize((new_size, new_size), Image.BICUBIC)
    in_image = np.array(image_tmp)
    out_image = pss_g.synthesis(
        in_image, in_image.shape[0], in_image.shape[1], 5, 4, 7, 25)
    #out_image = None
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


def size_fundamental_region(ratio, n, cell_struct):
    # 0...1:1 aspect ratio
    if (cell_struct == "P1"):
        return round(np.sqrt((n**2 * ratio)))
    elif (cell_struct == "P2" or cell_struct == "PG" or cell_struct == "PM"):
        return round(np.sqrt((n**2 * ratio)) * np.sqrt(2))
    elif (cell_struct == "PMM" or cell_struct == "CMM" or cell_struct == "PMG" or cell_struct == "PGG" or cell_struct == "P4" or cell_struct == "CM"):
        return round(np.sqrt((n**2 * ratio) * 4))
    elif (cell_struct == "P4M" or cell_struct == "P4G"):
        return round(np.sqrt((n**2 * ratio) * 8))
    elif (cell_struct == "P3"):
        # equilateral rhombus
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * (n**2 * ratio * 3)))
    elif (cell_struct == "P3M1"):
        # equilateral triangle
        return round(np.sqrt(((n**2 * ratio) * 3 / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)))
    elif (cell_struct == "P31M" or cell_struct == "P6"):
        # isosceles triangle
        # solved symbolically using mathematical software (maple)
        return round(6 * np.sqrt((n**2 * ratio)))
    elif (cell_struct == "P6M"):
        # right angle triangle
        # solved symbolically using mathematical software (maple)
        return round(6 * np.sqrt(2) * np.sqrt((n**2 * ratio)))
    else:
        return round(n * ratio * 2)


def size_lattice(ratio, n, cell_struct):
    # 0...1:1 aspect ratio
    if (cell_struct == "P1" or cell_struct == "PMM" or cell_struct == "PMG" or cell_struct == "PGG" or cell_struct == "P4" or cell_struct == "P4M" or cell_struct == "P4G" or cell_struct == "P2" or cell_struct == "PM" or cell_struct == "PG"):
        # square and rectangular
        return round(np.sqrt((n**2 * ratio)))
    elif (cell_struct == "CM" or cell_struct == "CMM"):
        # rhombic
        return round(np.sqrt((n**2 * ratio) * 2))
    elif (cell_struct == "P3"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * ((n**2 * ratio))))
    elif (cell_struct == "P3M1"):
        # hexagonal
        return round(np.sqrt((((n**2 * ratio) / 6) / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)))
    elif (cell_struct == "P31M" or cell_struct == "P6" or cell_struct == "P6M"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(np.sqrt(2) * np.sqrt((n**2 * ratio)))
    else:
        return round(n * ratio * 2)


def size_tile(ratio, n, cell_struct):
    # 0...1:1 aspect ratio
    if (cell_struct == "P1" or cell_struct == "PMM" or cell_struct == "PMG" or cell_struct == "PGG" or cell_struct == "P4" or cell_struct == "P4M" or cell_struct == "P4G" or cell_struct == "P2" or cell_struct == "PM" or cell_struct == "PG" or cell_struct == "CM" or cell_struct == "CMM"):
        # square and rectangular and rhombic
        return round(np.sqrt((n**2 * ratio)))
    elif (cell_struct == "P3"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * ((n**2 * ratio) / 6)))
    elif (cell_struct == "P3M1"):
        # hexagonal
        return round(np.sqrt((((n**2 * ratio) / 12) / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)))
    elif (cell_struct == "P31M" or cell_struct == "P6" or cell_struct == "P6M"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(np.sqrt(2) * np.sqrt((n**2 * ratio) / 2))
    else:
        return round(n * ratio * 2)

# returns average mag of the group


def mean_mag(freq_group, n_images):
    #n_images = 1
    mag = np.empty((freq_group[0].shape[0], freq_group[0].shape[1], n_images))
    for n in range(n_images):
        freq_group[n] = np.fft.fft2(freq_group[n], (freq_group[n].shape[0], freq_group[n].shape[1]))
    print(freq_group[0].shape)
    for n in range(n_images):
        mag[:, :, n] = np.abs(freq_group[n])
    out = np.median(mag, 2)
    print(out)
    return out


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
    parser.add_argument('--dots', '-d', default=False, type=str2bool,
                        help='Replace the fundamental region with random dot style patterns')
    parser.add_argument('--save_fmt', '-f', default="png", type=str,
                        help='Image save format')
    parser.add_argument('--filter_freq', '-f0fr', nargs='+',  default=[], type=str2list,
                        help='Center frequency (in cycle per degree) for dyadic bandpass filtering the fundamental region. [] does not invoke filtering. Might be extended for a multichannel filterbanks later.')
    parser.add_argument('--save_raw', '-r', default=False, type=str2bool,
                        help='save raw')
    parser.add_argument('--ctrl_images', '-cimg', default='False', type=str,
                        help='ps, phase, or no scrambling')
    parser.add_argument('--same_magnitude', '-m', default=False, type=str2bool,
                        help='New magnitude')
    parser.add_argument('--cmap', '-c', default="gray", type=str,
                        help='Color or greyscale map (hsv or gray)')
    parser.add_argument('--diagnostic', '-diag', default=True, type=str2bool,
                        help='Produce diagnostics for wallpapers')
    parser.add_argument('--debug', '-b', default=False, type=str2bool,
                        help='Debugging default parameters on')

    args = parser.parse_args()

    # need to investigate error in eval function
    make_set(args.groups, args.num_exemplars, args.wp_size_dva, args.wallpaperSize, args.lattice_sizing, args.fr_sizing, args.ratio, args.dots, args.filter_freq, args.save_fmt, args.save_raw,
                 args.scramble, args.same_magnitude, args.cmap, args.diagnostic, args.debug)
