import os
from datetime import datetime

import numpy as np
import numpy.matlib
import math
from PIL import Image, ImageDraw
from skimage import draw as skd
import warnings
import matplotlib.pyplot as plt
import dotsWallpaper as dW
import filter

import sys
import cv2 as cv

import logging

from scipy.stats import mode

from IPython.display import display, Markdown


def filterTile(inTile, filterIntensity):
    # generate random noise tile

    mu = 0.5;
    nx = np.size(inTile, 0);
    ny = np.size(inTile, 1);
    
    # make adaptive filtering
    
    sigma_x = 10*filterIntensity/nx;
    sigma_y = 10*filterIntensity/ny;
    
    x = np.linspace(0, 1, nx);
    y = np.linspace(0, 1, ny);
    
    gx = np.exp(-1 * (np.power((x - mu), 2)) / (2*(sigma_x**2))) / (sigma_x * math.sqrt(2 * math.pi));
    gy = np.exp(-1 * (np.power((y - mu), 2)) / (2*(sigma_y**2))) / (sigma_y * math.sqrt(2 * math.pi));

    gauss2 = np.matmul(gx.reshape(gx.shape[0], 1),gy.reshape(1, gy.shape[0]));
    gauss2 = gauss2 - gauss2.min();
    gauss2 = gauss2 / gauss2.max();
    gauss2 = gauss2 * 5;
    
    filtered = np.abs(np.fft.ifft2(np.fft.fft2(inTile) * gauss2));
    
    # normalize tile

    outTile = filtered - filtered.min();
    outTile = outTile / outTile.max();
    return outTile;

# generate random noise tile filtered using a spatial frequency filter
def spatial_filterTile(im_deg, inTile, lowpass=True, fwhm=1, f_center=0):
    #def make_noise(im_deg=10, rimg=False, lowpass=True, fwhm=1, f_center=0):
    #if not rimg:
    #    rimg = np.random.randn(500, 500)
    #    rimg = (rimg-rimg.min())/(rimg.max()-rimg.min())
    
    w = inTile.shape[1];
    h = inTile.shape[0];
    pix_per_degree = h/im_deg;
    
    # get into the frequency domain
    inTile_ = np.fft.fftshift(np.fft.fft2(inTile));
    
    # get frequency domain units
    cyc_per_pix_hor = np.linspace(-0.5, 0.5, w);
    cyc_per_pix_ver = np.linspace(-0.5, 0.5, h);
    cyc_per_deg_hor = cyc_per_pix_hor*pix_per_degree;
    cyc_per_deg_ver = cyc_per_pix_ver*pix_per_degree;
    
    # define mask in cyc per deg
    sig = fwhm/(2*np.sqrt(2*np.log2(2)));
    xx,yy = np.meshgrid(cyc_per_deg_hor,cyc_per_deg_ver);
    mask = np.exp(- ((xx**2+yy**2)**0.5-f_center)**2/(2*sig**2));
    if not lowpass:
        mask = 1-mask;
    
    # multiply the mask with the Fourier domain representation of our image
    filt_inTile_ = mask*inTile_;
    
    # back to image space
    filt_inTile = np.fft.ifft2(np.fft.ifftshift(filt_inTile_));
    # throw away imaginary parts
    filt_inTile = np.real(filt_inTile);
    # rescale image
    filt_inTile = (filt_inTile-filt_inTile.min())/(filt_inTile.max()-filt_inTile.min());
    return filt_inTile;

def new_p3(tile, isDots):
    # Generate p3 wallpaper
    
    # For magfactor, use a multiple of 3 to avoid the rounding error
    # when stacking two_tirds, one_third tiles together
    
    saveStr = os.getcwd() + '\\WPSet\\';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr; 
    patternPath = sPath + "_P3_Start_2"  + '.' + "png";
    magfactor = 6;
    if (isDots):
        height = tile.shape[0];
        s1 = round((height / 3));
        s = 2 * s1;
        width = round(height / math.sqrt(3)) - 1;
        xy = np.array ([[0, 0], [s1, width], [height, width], [2 * s1, 0], [0, 0]]).astype(np.uint32);
        mask = skd.polygon2mask((height, width), xy).astype(np.uint32);
        tile0 = (mask * tile[:, :width]).astype(np.uint32);
        
        # rotate rectangle by 120, 240 degs
   
        # note on 120deg rotation: 120 deg rotation of rhombus-shaped 
        # texture preserves the size, so 'crop' option can be used to 
        # tile size.
    
        tile0Im = Image.fromarray(tile0, 'I');
        tile0Im2 = Image.fromarray(tile0, 'I');
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand = False);
        tile120 = np.array(tile0Im_rot120, np.uint32);
        tile0Im_rot240 = tile0Im2.rotate(240, Image.NEAREST, expand = True);
        tile240 = np.array(tile0Im_rot240, np.uint32);
    
        # manually trim the tiles:
   
        # tile120 should have the same size as rectangle: [heigh x width]
        # tile120 = tile120(1:height, (floor(0.5*width) + 1):(floor(0.5*width) + width));
   
        # tile240 should have the size [s x 2*width]
        # find how much we need to cut from both sides
        diff = round(0.5 * (np.size(tile240, 1) - 2 * width));
        rowStart = round(0.25 * s);
        rowEnd = round(0.25 * s) + s;
        colStart = diff;
        colEnd = 2 * width + diff;
        tile240 = tile240[rowStart:rowEnd, colStart:colEnd].astype(np.uint32);

        # Start to pad tiles and glue them together
        # Resulting tile will have the size [3*height x 2* width]
    
        two_thirds1 = np.concatenate((tile0, tile120), axis=1).astype(np.uint32);
        two_thirds2 = np.concatenate((tile120, tile0), axis=1).astype(np.uint32);
    
        two_thirds = np.concatenate((two_thirds1, two_thirds2)).astype(np.uint32);
        
        
        
        #lower half of tile240 on the top, zero-padded to [height x 2 width]
        rowStart = int(0.5 * s);
        colEnd = 2 * width;
        one_third11 = np.concatenate((tile240[rowStart:,:], np.zeros((s, colEnd)))).astype(np.uint32);

        #upper half of tile240 on the bottom, zero-padded to [height x 2 width]
        rowEnd = int(0.5 * s); 
        one_third12 = np.concatenate((np.zeros((s, colEnd)), tile240[:rowEnd,:])).astype(np.uint32);

        # right half of tile240 in the middle, zero-padded to [height x 2 width]
        colStart = width;
        one_third21 = np.concatenate((np.zeros((s, width)), tile240[:,colStart:], np.zeros((s, width)))).astype(np.uint32);

        # left half of tile240in the middle, zero-padded to [height x 2 width]
        one_third22 = np.concatenate((np.zeros((s, width)), tile240[:,:width], np.zeros((s, width)))).astype(np.uint32);

        # cat them together
        one_third1 = np.concatenate((one_third11, one_third12)).astype(np.uint32);
        one_third2 = np.concatenate((one_third21, one_third22), axis=1).astype(np.uint32);


        # glue everything together, shrink and replicate
        one_third = np.maximum(one_third1,one_third2).astype(np.uint32);

        #size(whole) = [3xheight 2xwidth]
        whole = np.maximum(two_thirds, one_third).astype(np.uint32);
        whole[np.where(whole == np.min(whole))] = mode(whole, axis=None)[0].astype(np.uint32);
       
        wholeIm = Image.fromarray(whole, 'I');
    
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        wholeIm_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape));
    
        p3 = np.array(wholeIm.resize(wholeIm_new_size, Image.NEAREST)).astype(np.uint32);
        #Image.fromarray((whole[:, :]).astype(np.uint32), 'RGBA').save(patternPath, "png");
        return p3;
    else:

        #cm = plt.get_cmap("gray");
        #saveStr = os.getcwd() + '\\WPSet\\';
        #today = datetime.today();
        #timeStr = today.strftime("%Y%m%d_%H%M%S");
        #sPath = saveStr + timeStr; 
    
        tileIm = Image.fromarray(tile);

        # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        
        tile1 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BICUBIC));
        
    
        height = np.size(tile1, 0);

        # fundamental region is equlateral rhombus with side length = s
    
        s1 = round((height / 3));
        s = 2 * s1;
    
        # NOTE on 'ugly' way of calculating the widt = h
        # after magnification width(tile1) > tan(pi/6)*height(tile1) (should be equal)
        # and after 240 deg rotation width(tile240) < 2*width(tile1) (should be
        # bigger or equal, because we cat them together).
        # subtract one, to avoid screwing by imrotate(240)
    
        width = round(height / math.sqrt(3)) - 1;
        # width = min(round(height/sqrt(3)), size(tile1, 2)) - 1;

        # define rhombus-shaped mask

        xy = np.array ([[0, 0], [s1, width], [height, width], [2 * s1, 0], [0, 0]]);
    
        mask = skd.polygon2mask((height, width), xy);
        tile0 = mask * tile1[:, :width];

        # rotate rectangle by 120, 240 degs
   
        # note on 120deg rotation: 120 deg rotation of rhombus-shaped 
        # texture preserves the size, so 'crop' option can be used to 
        # tile size.
    
    
        tile0Im = Image.fromarray(tile0);
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand = False);
        tile120 = np.array(tile0Im_rot120);
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand = True);
        tile240 = np.array(tile0Im_rot240);
    
        # manually trim the tiles:
   
        # tile120 should have the same size as rectangle: [heigh x width]
        # tile120 = tile120(1:height, (floor(0.5*width) + 1):(floor(0.5*width) + width));
   
        # tile240 should have the size [s x 2*width]
        # find how much we need to cut from both sides
        diff = round(0.5 * (np.size(tile240, 1) - 2 * width));
        rowStart = round(0.25 * s);
        rowEnd = round(0.25 * s) + s;
        colStart = diff;
        colEnd = 2 * width + diff;
        tile240 = tile240[rowStart:rowEnd, colStart:colEnd];
    
        # Start to pad tiles and glue them together
        # Resulting tile will have the size [3*height x 2* width]
    
        two_thirds1 = np.concatenate((tile0, tile120), axis=1);
        two_thirds2 = np.concatenate((tile120, tile0), axis=1);
    
        two_thirds = np.concatenate((two_thirds1, two_thirds2));
    
        #lower half of tile240 on the top, zero-padded to [height x 2 width]
        rowStart = int(0.5 * s);
        colEnd = 2 * width;
        one_third11 = np.concatenate((tile240[rowStart:,:], np.zeros((s, colEnd))));
    
        #upper half of tile240 on the bottom, zero-padded to [height x 2 width]
        rowEnd = int(0.5 * s); 
        one_third12 = np.concatenate((np.zeros((s, colEnd)), tile240[:rowEnd,:]));
    
        # right half of tile240 in the middle, zero-padded to [height x 2 width]
        colStart = width;
        one_third21 = np.concatenate((np.zeros((s, width)), tile240[:,colStart:], np.zeros((s, width))));
    
        # left half of tile240in the middle, zero-padded to [height x 2 width]
        one_third22 = np.concatenate((np.zeros((s, width)), tile240[:,:width], np.zeros((s, width))));
    
        # cat them together
        one_third1 = np.concatenate((one_third11, one_third12));
        one_third2 = np.concatenate((one_third21, one_third22), axis=1);
    
        # glue everything together, shrink and replicate
        one_third = np.maximum(one_third1,one_third2);
    
        #size(whole) = [3xheight 2xwidth]
        whole = np.maximum(two_thirds, one_third);

        wholeIm = Image.fromarray(whole);
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        wholeIm_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape));

        p3 = np.array(wholeIm.resize(wholeIm_new_size, Image.BICUBIC));

        return p3;

def new_p3m1(tile, isDots):
    # Generate p3m1 wallpaper
    magfactor = 10;
    if (isDots):
        height = tile.shape[0];
        # fundamental region is equlateral triangle with side length = height 
        width = round(0.5 * height * math.sqrt(3));
        y1 = round(height/2);
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [y1, 0], [0, 0]];
        # Create half of the mask
        # reflect and concatenate, to get the full mask:   
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32);
        mask = np.concatenate((mask_half, np.flipud(mask_half))).astype(np.uint32);
        
        tile0 = (tile[:, :width] * mask).astype(np.uint32);
        
        # continue to modify the tile
        
        # reflect and rotate
        tile1_mirror = np.fliplr(tile0).astype(np.uint32);
        tile1_mirrorIm = Image.fromarray(tile1_mirror, 'I');
        tile240_Im = tile1_mirrorIm.rotate(240, Image.NEAREST, expand = True);
        tile240 = np.array(tile240_Im, np.uint32);
        # AY: I directly cut the tiles, because trim will
        # return slightly different size
        
        t_r1x = np.shape(tile240)[0];
        start_row = t_r1x - height;
        tile240 = tile240[start_row:, :width].astype(np.uint32);
        
        # AY: rotating mirrored tile(as opposed to tileR1) will cause less
        # border effects when we'll add it to two other tiles.
        tile120_Im = tile1_mirrorIm.rotate(120, Image.NEAREST, expand = True);
        tile120 = np.array(tile120_Im, np.uint32);
        tile120 = tile120[:height, :width].astype(np.uint32);
        
        # Assembling the tiles
        
        # We have 3 tiles with the same triangle rotated 0, 120 and 240
        # pad them and put them together
        zero_tile = np.zeros((y1, width));
        tile2 = np.concatenate((zero_tile, tile0, zero_tile)).astype(np.uint32);
        tile240 = np.concatenate((zero_tile, zero_tile, tile240)).astype(np.uint32);
        tile120 = np.concatenate((tile120, zero_tile, zero_tile)).astype(np.uint32);


        # Using max() will give us smoother edges, as opppose to sum()
        half1 = np.maximum(tile2, tile240).astype(np.uint32);
        half = np.maximum(half1, tile120).astype(np.uint32);
    
        # By construction, size(whole) = [4*y1 2*x1]
        whole = np.concatenate((half, np.fliplr(half)), axis=1).astype(np.uint32);
        
        # Shifting by 2 pix (as oppose to zero), we'll smoothly glue tile together. 
        # Set delta_pix value to zero and see the difference
        delta_pix = 2;
        start_row1 = 3 * y1 - delta_pix;
        start_row2 = 3 * y1;
        end_row1 = 4 * y1 - delta_pix;
        end_row2 = y1 + delta_pix;
        end_row3 = 4 * y1;
        end_col1 = 2 * width;
        
        topbit = np.concatenate((whole[start_row1:end_row1, width:end_col1], whole[start_row1:end_row1, :width]), axis=1).astype(np.uint32);
        botbit = np.concatenate((whole[delta_pix:end_row2, width:end_col1], whole[delta_pix:end_row2, :width]), axis=1).astype(np.uint32);
        
        whole[:y1, :] = np.maximum(whole[delta_pix:end_row2, :], topbit).astype(np.uint32);
        whole[start_row2:end_row3, :] = np.maximum(whole[start_row1:end_row1, :], botbit).astype(np.uint32);
        #whole[np.where(whole == 0)] = np.max(whole).astype(np.uint32);

        # cutting middle piece of tile
        mid_tile = whole[y1:start_row2, :width].astype(np.uint32);
        # reflecting middle piece and glueing both pieces to the bottom
        # size(bigTile)  = [6*y1 2*x1]  
        cat_mid_flip = np.concatenate((mid_tile, np.fliplr(mid_tile)), axis=1).astype(np.uint32);
        bigTile = np.concatenate((whole, cat_mid_flip)).astype(np.uint32);
        bigTile[np.where(bigTile == np.min(bigTile))] = mode(bigTile, axis=None)[0].astype(np.uint32);

        bigTileIm = Image.fromarray(bigTile, 'I');
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        bigTile_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(bigTile.shape));
        p3m1 = np.array(bigTileIm.resize(bigTile_new_size, Image.NEAREST)).astype(np.uint32);

        return p3m1;

    else:
        tileIm = Image.fromarray(tile);
        # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BICUBIC));
        height = np.shape(tile1)[0];
        
        
        # fundamental region is equlateral triangle with side length = height 
        width = round(0.5 * height * math.sqrt(3));
    
        y1 = round(height/2);
       
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [y1, 0], [0, 0]];
        
        # Create half of the mask
        # reflect and concatenate, to get the full mask:   
    
        mask_half = skd.polygon2mask((y1, width), mask_xy);
        mask = np.concatenate((mask_half, np.flipud(mask_half)));
    
        # equilateral triangle inscribed into rectangle 
        tile0 = tile1[:, :width] * mask;
     
        # continue to modify the tile
        
        # reflect and rotate
        tile1_mirror = np.fliplr(tile0);
        tile1_mirrorIm = Image.fromarray(tile1_mirror);
        tile240_Im = tile1_mirrorIm.rotate(240, Image.BILINEAR, expand = True);
        tile240 = np.array(tile240_Im);
        # AY: I directly cut the tiles, because trim will
        # return slightly different size
        
        t_r1x = np.shape(tile240)[0];
        start_row = t_r1x - height;
        tile240 = tile240[start_row:, :width];
        
        # AY: rotating mirrored tile(as opposed to tileR1) will cause less
        # border effects when we'll add it to two other tiles.
        tile120_Im = tile1_mirrorIm.rotate(120, Image.BILINEAR, expand = True);
        tile120 = np.array(tile120_Im);
        
        tile120 = tile120[:height, :width];
        # Assembling the tiles
        
        # We have 3 tiles with the same triangle rotated 0, 120 and 240
        # pad them and put them together
        zero_tile = np.zeros((y1, width));
        tile2 = np.concatenate((zero_tile, tile0, zero_tile));
        tile240 = np.concatenate((zero_tile, zero_tile, tile240));
        tile120 = np.concatenate((tile120, zero_tile, zero_tile));
        
        # Using max() will give us smoother edges, as opppose to sum()
        half1 = np.maximum(tile2, tile240);
        half = np.maximum(half1, tile120);
    
        # By construction, size(whole) = [4*y1 2*x1]
        whole = np.concatenate((half, np.fliplr(half)), axis=1);
        # Shifting by 2 pix (as oppose to zero), we'll smoothly glue tile together. 
        # Set delta_pix value to zero and see the difference
        delta_pix = 2;
        start_row1 = 3 * y1 - delta_pix;
        start_row2 = 3 * y1;
        end_row1 = 4 * y1 - delta_pix;
        end_row2 = y1 + delta_pix;
        end_row3 = 4 * y1;
        end_col1 = 2 * width;
        
        topbit = np.concatenate((whole[start_row1:end_row1, width:end_col1], whole[start_row1:end_row1, :width]), axis=1);
        botbit = np.concatenate((whole[delta_pix:end_row2, width:end_col1], whole[delta_pix:end_row2, :width]), axis=1);
        
        whole[:y1, :] = np.maximum(whole[delta_pix:end_row2, :], topbit);
        whole[start_row2:end_row3, :] = np.maximum(whole[start_row1:end_row1, :], botbit);
        # cutting middle piece of tile
        mid_tile = whole[y1:start_row2, :width];
        # reflecting middle piece and glueing both pieces to the bottom
        # size(bigTile)  = [6*y1 2*x1]  
        cat_mid_flip = np.concatenate((mid_tile, np.fliplr(mid_tile)), axis=1);
        bigTile = np.concatenate((whole, cat_mid_flip));
        bigTileIm = Image.fromarray(bigTile);
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        bigTile_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(bigTile.shape));
        p3m1 = np.array(bigTileIm.resize(bigTile_new_size, Image.BICUBIC));
        return p3m1;

def  new_p31m(tile, isDots):
    # Generate p31m wallpaper
    magfactor = 6;
    
    if (isDots):
        
        tile0 = tile.astype(np.uint32);
        height = np.shape(tile0)[0];
        width = round(0.5 * height / math.sqrt(3));
        y1 = round(height / 2);
        
        # fundamental region is an isosceles triangle with angles(30, 120, 30)
        
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]];
    
        # make half of the mask
        # reflect and concatenate, to get the full mask:    
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32);
    
        mask = np.concatenate((mask_half, np.flipud(mask_half))).astype(np.uint32);
    
        
        # size(tile0) = [height  width]
        tile0 = (mask * tile0[:, :width]).astype(np.uint32);
        
        # rotate the tile
        tile0_Im = Image.fromarray(tile0, 'I');
        tile120_Im = tile0_Im.rotate(120, Image.NEAREST, expand = True);
        tile120 = np.array(tile120_Im).astype(np.uint32);
    
        tile240_Im = tile0_Im.rotate(240, Image.NEAREST, expand = True);
        tile240 = np.array(tile240_Im).astype(np.uint32);
        
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from 
        tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1).astype(np.uint32);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);   
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - y1;
        
    
        # size(tile120, tile240) = [height 3width] 
        
        tile120 = tile120[y120:, x120:].astype(np.uint32);
        tile240 = tile240[:y1, x120:].astype(np.uint32);
        
        # we have 3 tiles that will comprise
        # equilateral triangle together
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile1 already padded
        tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120)).astype(np.uint32);
        
        tile240 = np.concatenate((tile240, np.zeros((y1, width * 3)))).astype(np.uint32);
        
        # size(tri) = [height 3width]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32);
        mirror_tri = np.fliplr(tri).astype(np.uint32);
        
        # use shift overlap, to smooth the edges
        delta_pix = 3;
        row_start = y1 - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = y1 + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32);
        tile2 = np.maximum(shifted, tri).astype(np.uint32);
    
        # size(tile3) = [height 6width]
        tile3 = np.concatenate((tile2, np.fliplr(tile2)), axis=1).astype(np.uint32);
        tile3[np.where(tile3 == np.min(tile3))] = mode(tile3, axis=None)[0].astype(np.uint32);
        tile3_Im = Image.fromarray(tile3, 'I');
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p31m = np.array(tile3_Im.resize(tile3_new_size, Image.NEAREST)).astype(np.uint32); 
    
    else:
        tile = tile.astype('float32');
        
    
        tileIm = Image.fromarray(tile);
        # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile0 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BILINEAR));
    
        height = np.shape(tile0)[0];
        width = round(0.5 * height / math.sqrt(3));
        y1 = round(height / 2);
        
        # fundamental region is an isosceles triangle with angles(30, 120, 30)
        
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]];
    
        # make half of the mask
        # reflect and concatenate, to get the full mask:    
        mask_half = skd.polygon2mask((y1, width), mask_xy);
    
        mask = np.concatenate((mask_half, np.flipud(mask_half)));
    
        
        # size(tile0) = [height  width]
        tile0 = mask * tile0[:, :width];
        
        # rotate the tile
        tile0_Im = Image.fromarray(tile0);
        tile120_Im = tile0_Im.rotate(120, Image.BILINEAR, expand = True);
        tile120 = np.array(tile120_Im);
    
        tile240_Im = tile0_Im.rotate(240, Image.BILINEAR, expand = True);
        tile240 = np.array(tile240_Im);
        
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from 
        tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2);   
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - y1;
        
    
        # size(tile120, tile240) = [height 3width] 
        
        tile120 = tile120[y120:, x120:];
        tile240 = tile240[:y1, x120:];
        
        # we have 3 tiles that will comprise
        # equilateral triangle together
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile1 already padded
        tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120));
        
        tile240 = np.concatenate((tile240, np.zeros((y1, width * 3))));
        
        # size(tri) = [height 3width]
        tri = np.maximum(np.maximum(tile0, tile120), tile240);
        mirror_tri = np.fliplr(tri);
        
        # use shift overlap, to smooth the edges
        delta_pix = 3;
        row_start = y1 - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = y1 + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]));
        tile2 = np.maximum(shifted, tri);
    
        # size(tile3) = [height 6width]
        tile3 = np.concatenate((tile2, np.fliplr(tile2)), axis=1);
        tile3_Im = Image.fromarray(tile3);
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p31m = np.array(tile3_Im.resize(tile3_new_size, Image.BILINEAR)); 
    return p31m;

def new_p6(tile, isDots):
    # Generate p6 wallpaper
    magfactor = 6;
    
    if (isDots):
        tile1 = tile.astype(np.uint32);
        height = np.shape(tile1)[0];
        width = int(round(0.5 * height * np.tan(np.pi / 6)));
        y1 = round(height/2);
    
        
        # fundamental region is an isosceles triangle with angles(30, 120, 30)
    
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]];
        
        # half of the mask
        # reflect and concatenate, to get the full mask:
        mask_half = skd.polygon2mask((y1, width), mask_xy).astype(np.uint32);
    
        mask = np.concatenate((mask_half, np.flipud(mask_half))).astype(np.uint32);
        
        # size(tile0) = [height x width]
        tile0 = (mask * tile1[:, :width]).astype(np.uint32);
        
        # rotate tile1
        tile0Im = Image.fromarray(tile0, 'I');
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand = True);
        tile120 = np.array(tile0Im_rot120).astype(np.uint32);
        tile0Im_rot240 = tile0Im.rotate(240, Image.NEAREST, expand = True);
        tile240 = np.array(tile0Im_rot240).astype(np.uint32);
    
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from
        tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1).astype(np.uint32);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - y1;
        
        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:].astype(np.uint32);
        tile240 = tile240[:y1, x120:].astype(np.uint32);
    
        # we have 3 tiles that will comprise
        # equilateral triangle together
        
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120)).astype(np.uint32);
        tile240 = np.concatenate((tile240, np.zeros((y1, width * 3)))).astype(np.uint32);
        
        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32);
        
        # mirror_tri = fliplr(tri); --wrong! should be (fliplr(flipud(tri)))
        triIm = Image.fromarray(tri, 'I');
        triIm_rot180 = triIm.rotate(180, expand = True);
        mirror_tri = np.array(triIm_rot180).astype(np.uint32);
        
        # shifw w.slight overlap, 
        delta_pix = 3;
        row_start = y1 - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = y1 + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32);
    
        tile2 = np.maximum(tri, shifted).astype(np.uint32);
        t2 = int(np.floor(0.5*np.shape(tile2)[0]));
        
        tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :])).astype(np.uint32); 
        
        # size(tile3) = [2y1 x 6x1]
        tile3 =  np.concatenate((tile2, tile2_flipped),axis=1).astype(np.uint32);
        tile3[np.where(tile3 == np.min(tile3))] = mode(tile3, axis=None)[0].astype(np.uint32);
        tile3_Im = Image.fromarray(tile3, 'I');
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p6 = np.array(tile3_Im.resize(tile3_new_size, Image.NEAREST)).astype(np.uint32);
        
    else:
        tileIm = Image.fromarray(tile);
        # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BICUBIC));
    
        height = np.shape(tile1)[0];
        width = int(round(0.5 * height * np.tan(np.pi / 6)));
        y1 = round(height/2);
    
        
        # fundamental region is an isosceles triangle with angles(30, 120, 30)
    
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [y1, width], [height, 0], [0, 0]];
        
        # half of the mask
        # reflect and concatenate, to get the full mask:
        mask_half = skd.polygon2mask((y1, width), mask_xy);
    
        mask = np.concatenate((mask_half, np.flipud(mask_half)));
        
        # size(tile0) = [height x width]
        tile0 = mask * tile1[:, :width];
        
        # rotate tile1
        tile0Im = Image.fromarray(tile0);
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand = True);
        tile120 = np.array(tile0Im_rot120);
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand = True);
        tile240 = np.array(tile0Im_rot240);
    
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from
        tile0 = np.concatenate((tile0, np.zeros((height, width * 2))), axis=1);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - y1;
        
        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:];
        tile240 = tile240[:y1, x120:];
    
        # we have 3 tiles that will comprise
        # equilateral triangle together
        
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate((np.zeros((y1, width * 3)), tile120));
        tile240 = np.concatenate((tile240, np.zeros((y1, width * 3))));
        
        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240);
        
        # mirror_tri = fliplr(tri); --wrong! should be (fliplr(flipud(tri)))
        triIm = Image.fromarray(tri);
        triIm_rot180 = triIm.rotate(180, expand = True);
        mirror_tri = np.array(triIm_rot180);
        
        # shifw w.slight overlap, 
        delta_pix = 3;
        row_start = y1 - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = y1 + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]));
    
        tile2 = np.maximum(tri, shifted);
        t2 = int(np.floor(0.5*np.shape(tile2)[0]));
        
        tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :])); 
        
        # size(tile3) = [2y1 x 6x1]
        tile3 =  np.concatenate((tile2, tile2_flipped),axis=1);
        tile3_Im = Image.fromarray(tile3);
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p6 = np.array(tile3_Im.resize(tile3_new_size, Image.BICUBIC)); 
    return p6;

def new_p6m(tile, isDots):
    # Generate p6m wallpaper
    
    magfactor = 6;
    if (isDots):
        tile1 = tile.astype(np.uint32);
    
        height = np.shape(tile1)[0];
        
        width = round( height / math.sqrt(3));
    
        # fundamental region is right triangle with angles (30, 60)
        
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [height, width], [height, 0], [0, 0]];
        
        # half of the mask
        # reflect and concatenate, to get the full mask: 
        mask = skd.polygon2mask((height, width), mask_xy).astype(np.uint32);
        
        # right triangle inscribed into rectangle 
        tile0 = (tile1[:, :width] * mask).astype(np.uint32);
        
        # size(tile0) = [height x width]
        tile0 = np.concatenate((tile0, np.flipud(tile0))).astype(np.uint32);
        
        # rotate tile1
        tile0Im = Image.fromarray(tile0, 'I');
        tile0Im_rot120 = tile0Im.rotate(120, Image.NEAREST, expand = True);
        tile120 = np.array(tile0Im_rot120).astype(np.uint32);
        tile0Im_rot240 = tile0Im.rotate(240, Image.NEAREST, expand = True);
        tile240 = np.array(tile0Im_rot240).astype(np.uint32);
    
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from 
        tile0 = np.concatenate((tile0, np.zeros((height * 2, width * 2))), axis=1).astype(np.uint32);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2); 
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - height;
        
        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:].astype(np.uint32);
        tile240 = tile240[:height, x120:].astype(np.uint32);
        
        # we have 3 tiles that will comprise
        # equilateral triangle together
        
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate((np.zeros((height, width * 3)), tile120)).astype(np.uint32);
        tile240 = np.concatenate((tile240, np.zeros((height, width * 3)))).astype(np.uint32);
        
        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240).astype(np.uint32);
        triIm = Image.fromarray(tri, 'I');
        triIm_rot180 = triIm.rotate(180, expand = True);
        mirror_tri = np.array(triIm_rot180).astype(np.uint32);
        
        # shifw w.slight overlap, 
        delta_pix = 3;
        row_start = height - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = height + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :])).astype(np.uint32);
        
        tile2 = np.maximum(tri, shifted).astype(np.uint32);
        t2 = int(np.floor(0.5 * np.shape(tile2)[0]));
        
        tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :])).astype(np.uint32); 
        # size(tile3) = [2y1 x 6x1]
        tile3 =  np.concatenate((tile2, tile2_flipped),axis=1).astype(np.uint32);
        tile3[np.where(tile3 == np.min(tile3))] = mode(tile3, axis=None)[0].astype(np.uint32);
        tile3_Im = Image.fromarray(tile3, 'I');
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p6m = np.array(tile3_Im.resize(tile3_new_size, Image.NEAREST)).astype(np.uint32); 
    else:
        tileIm = Image.fromarray(tile);
        # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
        tile1 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BICUBIC));
    
        height = np.shape(tile1)[0];
        
        width = round( height / math.sqrt(3));
    
        # fundamental region is right triangle with angles (30, 60)
        
        # vetrices of the triangle (closed polygon => four points)
        mask_xy = [[0, 0], [height, width], [height, 0], [0, 0]];
        
        # half of the mask
        # reflect and concatenate, to get the full mask: 
        mask = skd.polygon2mask((height, width), mask_xy);
        
        # right triangle inscribed into rectangle 
        tile0 = tile1[:, :width] * mask;
        
        # size(tile0) = [height x width]
        tile0 = np.concatenate((tile0, np.flipud(tile0)));
        
        # rotate tile1
        tile0Im = Image.fromarray(tile0);
        tile0Im_rot120 = tile0Im.rotate(120, Image.BILINEAR, expand = True);
        tile120 = np.array(tile0Im_rot120);
        tile0Im_rot240 = tile0Im.rotate(240, Image.BILINEAR, expand = True);
        tile240 = np.array(tile0Im_rot240);
    
        # trim the tiles manually, using trigonometric laws
        # NOTE: floor and round give us values that differ by 1 pix.
        # to trim right, we'll have to derive the trim value from 
        tile0 = np.concatenate((tile0, np.zeros((height * 2, width * 2))), axis=1);   
        delta = np.shape(tile0)[1];
        
        # ideally we would've used  
        # delta = floor(sqrt(3)*s/2) OR round(sqrt(3)*s/2); 
        x120 = np.shape(tile120)[1] - delta;
        y120 = np.shape(tile120)[0] - height;
        
        # size(tile120, 240) = [y1 x 3x1]
        tile120 = tile120[y120:, x120:];
        tile240 = tile240[:height, x120:];
        
        # we have 3 tiles that will comprise
        # equilateral triangle together
        
        # glue them together by padding and smoothing edges (max instead of sum)
        # tile0 already padded
        tile120 = np.concatenate((np.zeros((height, width * 3)), tile120));
        tile240 = np.concatenate((tile240, np.zeros((height, width * 3))));
        
        # size(tri) = [2y1 x 3x1]
        tri = np.maximum(np.maximum(tile0, tile120), tile240);
        triIm = Image.fromarray(tri);
        triIm_rot180 = triIm.rotate(180, expand = True);
        mirror_tri = np.array(triIm_rot180);
        
        # shifw w.slight overlap, 
        delta_pix = 3;
        row_start = height - delta_pix;
        row_end1 = mirror_tri.shape[0] - delta_pix;
        row_end2 = height + delta_pix;
        shifted = np.concatenate((mirror_tri[row_start:row_end1, :], mirror_tri[delta_pix:row_end2, :]));
        
        tile2 = np.maximum(tri, shifted);
        t2 = int(np.floor(0.5 * np.shape(tile2)[0]));
        
        tile2_flipped = np.concatenate((tile2[t2:, :], tile2[:t2, :])); 
        # size(tile3) = [2y1 x 6x1]
        tile3 =  np.concatenate((tile2, tile2_flipped),axis=1);
        tile3_Im = Image.fromarray(tile3);
        # tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(whole.shape)) to calculate the (width, height) of the image
        
        tile3_new_size = tuple(int(np.ceil(i * (1 / magfactor))) for i in reversed(tile3.shape));
        p6m = np.array(tile3_Im.resize(tile3_new_size, Image.BICUBIC));
    return p6m;

def generateWPimage(wptype, N, n, isFR, isLattice, ratio, angle, isDiagnostic, fundemental_region_filter,
                    fundamental_region_source_type, isDots, cmap, optTexture = None):
    #  generateWPimage(type,N,n,optTexture)
    # generates single wallaper group image
    # wptype defines the wallpaper group
    # N is the size of the output image. For complex groups
    #   returned image will have size at least NxN.
    # n the size of repeating pattern for all groups.
    # isDiagnostic whether to generate diagnostic images (outlining fundamental region and lattice)
    # isSpatFreqFilt generate a spatial frequency filtered wallpaper
    # fwhm full width at half maximum of spatial frequency filter
    # whether spatialfrequency filter is lowpass or highpass
    # isDots generate wallpaper using dots rather than random noise
    
    # default
    # save paths for debugging
    saveStr = os.getcwd() + '_WPSet_';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr; 
        
    if fundamental_region_source_type == 'uniform_noise' and isDots == False:
        # TODO: do we  need white noise here as well?
        print('uniform noise');
        #texture = np.random.rand(n,n);
        grain = 1;
        texture = filterTile(np.random.rand(n, n), grain);
    elif isDots:
        print('random dots');
        texture = dW.genDotsFund(n, 0.05, 0.05, 5, wptype);
    elif isinstance(fundamental_region_source_type,np.ndarray):
        print('texture was passed explicitly');
        optTexture = fundamental_region_source_type; 
        minDim = np.min(np.shape(optTexture));
        # stretch user-defined texture, if it is too small for sampling
        if minDim < n:
            ratioTexture = round(n / minDim);
            optTexture = np.array(Image.resize(reversed((optTexture.shape * ratioTexture)), Image.NEAREST));
        texture = optTexture;
    else:
        raise Exception ('this source type ({})is not implemented'.format(type(fundamental_region_source_type)));
    # do filtering
    if fundemental_region_filter:
        if isinstance(fundemental_region_filter, filter.Cosine_filter):
            texture = fundemental_region_filter.filter_image(texture)
            #scale texture into range 0...1
            texture = (texture-texture.min())/(texture.max()-texture.min())
        else:
            raise Exception ('this filter type ({}) is not implemented'.format(type(fundemental_region_filter)))
    #else:
       # TODO: not exactly sure, what this lowpass filter is supposed to do. in any case:
       #       it should be adapted to this structure that separates the noise generation from the filtering

    try:
        # generate the wallpapers
        if wptype == 'P0':
                p0 = np.array(Image.resize(reversed((texture.shape * round(N/n))), Image.NEAREST));
                image = p0;
                return image;                                
        elif wptype == 'P1':
                width = n;
                height = width;
                p1 = texture[:height, :width];
                image = catTiles(p1, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p1, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                
        elif wptype == 'P2':
                height = round(n/2);
                width = 2*height;
                start_tile = texture[:height, :width];
                tileR180 = np.rot90(start_tile, 2)
                p2 = np.concatenate((start_tile, tileR180));
                image = catTiles(p2, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p2, isFR, isLattice, N, ratio, cmap, isDots);     
                return image;
        elif wptype == 'PM':
                height = round(n/2);
                width = 2*height;
                start_tile = texture[:height, :width];
                mirror = np.flipud(start_tile);
                pm = np.concatenate((start_tile, mirror));
                image = catTiles(pm, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, pm, isFR, isLattice, N, ratio, cmap, isDots); 
                return image;                
        elif wptype == 'PG':
                height = round(n/2);
                width = 2*height;
                start_tile = texture[:height, :width];
                tile = np.rot90(start_tile, 3);
                glide = np.flipud(tile);
                pg = np.concatenate((tile, glide), axis=1);
                image = catTiles(pg.T, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, pg, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                  
        elif wptype == 'CM':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];                
                mirror = np.fliplr(start_tile);
                tile1 = np.concatenate((start_tile, mirror), axis=1);
                tile2 = np.concatenate((mirror, start_tile), axis=1);
                cm = np.concatenate((tile1, tile2));
                image = catTiles(cm, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, cm, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                
        elif wptype == 'PMM':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];                    
                mirror = np.fliplr(start_tile);
                concatTmp1 = np.concatenate((start_tile, mirror), axis=1);
                concatTmp2 = np.concatenate((np.flipud(start_tile), np.flipud(mirror)), axis=1);
                pmm = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(pmm, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, pmm, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                 
        elif wptype == 'PMG':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tile_rot180 = np.rot90(start_tile, 2);
                concatTmp1 = np.concatenate((start_tile, start_tile_rot180), axis=1);
                concatTmp2 = np.concatenate((np.flipud(start_tile), np.fliplr(start_tile)), axis=1);
                pmg = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(pmg, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, pmg, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                 
        elif wptype == 'PGG':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tile_rot180 = np.rot90(start_tile, 2);
                concatTmp1 = np.concatenate((start_tile, np.flipud(start_tile)), axis=1);
                concatTmp2 = np.concatenate((np.fliplr(start_tile), start_tile_rot180), axis=1);
                pgg = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(pgg, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, pgg, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                 
        elif wptype == 'CMM':
                height = round(n/4);
                width = 2*height;
                start_tile = texture[:height, :width];
                start_tile_rot180 = np.rot90(start_tile, 2);
                tile1 = np.concatenate((start_tile, start_tile_rot180));               
                tile2 = np.flipud(tile1);
                concatTmp1 = np.concatenate((tile1, tile2), axis=1);
                concatTmp2 = np.concatenate((tile2, tile1), axis=1);
                cmm = np.concatenate((concatTmp1, concatTmp2)); 
                image = catTiles(cmm, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, cmm, isFR, isLattice, N, ratio, cmap, isDots);
                return image;                 
        elif wptype == 'P4':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tile_rot90 = np.rot90(start_tile, 1);
                start_tile_rot180 = np.rot90(start_tile, 2);
                start_tile_rot270 = np.rot90(start_tile, 3);
                concatTmp1 = np.concatenate((start_tile, start_tile_rot270), axis=1);
                concatTmp2 = np.concatenate((start_tile_rot90, start_tile_rot180), axis=1);                
                p4 = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p4, isFR, isLattice, N, ratio, cmap, isDots);                
                return image; 
        elif wptype == 'P4M':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                xy = np.array ([[0, 0], [width, height], [0, height], [0, 0]]);    
                mask = skd.polygon2mask((height, width), xy);
                tile1 = mask * start_tile;
                tile2 = np.fliplr(tile1);
                tile2 = np.rot90(tile2, 1);
                tile = np.maximum(tile1, tile2);
                tile_rot90 = np.rot90(tile, 1);
                tile_rot180 = np.rot90(tile, 2);
                tile_rot270 = np.rot90(tile, 3);
                concatTmp1 = np.concatenate((tile, tile_rot270), axis=1);
                concatTmp2 = np.concatenate((tile_rot90, tile_rot180), axis=1); 
                p4m = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4m, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p4m, isFR, isLattice, N, ratio, cmap, isDots); 
                return image; 
        elif wptype == 'P4G':
                height = round(n/2);
                width = height;
                if (isDots):
                    start_tile = texture[:height, :width].astype(np.uint32);
                    xy = np.array ([[0, 0], [width, 0], [width, height], [0, 0]]).astype(np.uint32);
                    mask = skd.polygon2mask((height, width), xy).astype(np.uint32);
                    tile1 = (mask.astype(np.uint32) * start_tile.astype(np.uint32)).astype(np.uint32);
                    tile1 = start_tile - tile1;
                    tile2 = np.fliplr(tile1).astype(np.uint32);
                    tile2 = np.rot90(tile2, 1).astype(np.uint32);
                    tile = np.maximum(tile1, tile2).astype(np.uint32);
                    tile_rot90 = np.rot90(tile, 1).astype(np.uint32);
                    tile_rot180 = np.rot90(tile, 2).astype(np.uint32);
                    tile_rot270 = np.rot90(tile, 3).astype(np.uint32);
                    concatTmp1 = np.concatenate((tile_rot270, tile_rot180), axis=1).astype(np.uint32);
                    concatTmp2 = np.concatenate((tile, tile_rot90), axis=1).astype(np.uint32); 
                    p4g = np.concatenate((concatTmp1, concatTmp2)).astype(np.uint32);
                else:
                    start_tile = texture[:height, :width];
                    xy = np.array ([[0, 0], [width, 0], [width, height], [0, 0]]);
                    mask = skd.polygon2mask((height, width), xy);
                    tile1 = mask * start_tile;
                    tile2 = np.fliplr(tile1);
                    tile2 = np.rot90(tile2, 1);
                    tile = np.maximum(tile1, tile2);
                    tile_rot90 = np.rot90(tile, 1);
                    tile_rot180 = np.rot90(tile, 2);
                    tile_rot270 = np.rot90(tile, 3);
                    concatTmp1 = np.concatenate((tile_rot270, tile_rot180), axis=1);
                    concatTmp2 = np.concatenate((tile, tile_rot90), axis=1); 
                    p4g = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4g, N, wptype);   
                if (isDiagnostic):
                    diagnostic(image, wptype, p4g, isFR, isLattice, N, ratio, cmap, isDots);               
                return image;
        elif wptype == 'P3':
                alpha = np.pi/3;
                s = n  / math.sqrt(3 * np.tan(alpha));
                height = math.floor(s * 1.5);

                start_tile = texture[:height,:];
                if (isDots):
                    p3 = new_p3(texture, isDots);
                else:
                    p3 = new_p3(start_tile, isDots);
                image = catTiles(p3, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p3, isFR, isLattice, N, ratio, cmap, isDots);                
                return image;                
        elif wptype == 'P3M1':
                alpha = np.pi/3;
                s = n/math.sqrt(3*np.tan(alpha));
                height = round(s);
                start_tile = texture[:height,:]; 
                if (isDots):
                    p3m1 = new_p3m1(texture, isDots);  
                else:
                    p3m1 = new_p3m1(start_tile, isDots);  
                image = catTiles(p3m1, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p3m1, isFR, isLattice, N, ratio, cmap, isDots); 
                return image;                
        elif wptype == 'P31M':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s);
                start_tile = texture[:height,:];
                if (isDots):
                    p31m = new_p31m(texture, isDots); 
                else:
                    p31m = new_p31m(start_tile, isDots);
               
                # ugly trick
                p31m_1 = np.fliplr(np.transpose(p31m));
                image = catTiles(p31m_1, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p31m_1, isFR, isLattice, N, ratio, cmap, isDots); 
                return image;
        elif wptype == 'P6':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s);
                start_tile = texture[:height,:]; 
                if (isDots):
                    p6 = new_p6(texture, isDots); 
                else:
                    p6 = new_p6(start_tile, isDots);
                image = catTiles(p6, N, wptype);
                if (isDiagnostic):
                    diagnostic(image, wptype, p6, isFR, isLattice, N, ratio, cmap, isDots); 
                return image;
        elif wptype == 'P6M':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s/2);
                start_tile = texture[:height,:];
                if (isDots):
                    p6m = new_p6m(texture, isDots); 
                else:
                    p6m = new_p6m(start_tile, isDots);
                image = catTiles(p6m, N, wptype); 
                if (isDiagnostic):
                    diagnostic(image, wptype, p6m, isFR, isLattice, N, ratio, cmap, isDots); 
                return image;
        else:
                warnings.warn('Unexpected Wallpaper Group type. Returning random noise.', UserWarning);
                image = np.matlib.repmat(texture, [np.ceil(N/n),  np.ceil(N/n)]);
                return image;
    except Exception as err:
        print('new_SymmetricNoise:Error making ' + wptype);
        print(err.args);
        
def catTiles(tile, N, wptype):
    # disp tile square
    sq = np.shape(tile)[0] * np.shape(tile)[1];
    print(wptype + ' area of tile = ', sq);                
    
    # write tile
    #tileIm = Image.fromarray(tile);
    #tileIm.save('~/Documents/PYTHON/tiles/' + wptype + '_tile.jpeg', 'JPEG');
    
    # resize tile to ensure it will fit wallpaper size properly
    if (tile.shape[0] > N):
        tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), : ];
        N = tile.shape[0];
    if (tile.shape[1] > N):
        tile = tile[:, round((tile.shape[1] - N) / 2): round((N + (tile.shape[1] - N) / 2))];
        N = tile.shape[1];
    if (tile.shape[0] % 2 != 0):
        tile = tile[:tile.shape[0] - 1,:];
    if (tile.shape[1] % 2 != 0):
        tile = tile[:,:tile.shape[1] - 1];
    dN = tuple(1 +(math.floor(N / ti)) for ti in np.shape(tile));

    row = dN[0];
    col = dN[1];
    
    # to avoid divide by zero errors
    if(dN[0] == 1):
        row = row + 1;
    if(dN[1] == 1):
        col = col + 1;
    
    # repeat tile to create initial wallpaper less the excess necessary to complete the wallpaper to desired size
    img = numpy.matlib.repmat(tile, row - 1, col - 1);
    
    row = math.floor(img.shape[0] + tile.shape[0] * ((1 + (N / tile.shape[0])) - dN[0]));
    col = math.floor(img.shape[1] + tile.shape[1] * ((1 + (N / tile.shape[1])) - dN[1]));
    if (math.floor(img.shape[0] + tile.shape[0] * ((1 + (N / tile.shape[0])) - dN[0])) % 2 != 0):
        row = row + 1;

    if (math.floor(img.shape[1] + tile.shape[1] * ((1 + (N / tile.shape[1])) - dN[1])) % 2 != 0):
        col =  col + 1;

    img_final = np.zeros((row, col));
    

    # centers the evenly created tile and then even distributes the rest of the tile around the border s.t. the total size of the wallpaper = the desired input size of the wallpaper
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2),math.ceil((img_final.shape[1] - img.shape[1]) / 2): img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[:,:];
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2) : img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2), : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[ :, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2) : img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2), img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2): ] = img[ :, : math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), math.ceil((img_final.shape[1] - img.shape[1]) / 2) : img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) ] = img[ img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, : ];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2) : , math.ceil((img_final.shape[1] - img.shape[1]) / 2) : img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) ] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), : ];
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2): , : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2): , img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) : ] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), :math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    img_final[:math.ceil((img_final.shape[0] - img.shape[0]) / 2) ,  img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2): ] = img[ img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2) : , :math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    

    return img_final

def diagCatTiles(tile, N, diagTile):           
    
    # write tile
    #tileIm = Image.fromarray(tile);
    #tileIm.save('~/Documents/PYTHON/tiles/' + wptype + '_tile.jpeg', 'JPEG');
    
    # resize tile to ensure it will fit wallpaper size properly
    if (tile.shape[0] > N):
        tile = tile[round((tile.shape[0] - N) / 2): round((N + (tile.shape[0] - N) / 2)), : ];
        N = tile.shape[0];
    if (tile.shape[1] > N):
        tile = tile[:, round((tile.shape[1] - N) / 2): round((N + (tile.shape[1] - N) / 2))];
        N = tile.shape[1];
    if (tile.shape[0] % 2 != 0):
        tile = tile[:tile.shape[0] - 1,:];
    if (tile.shape[1] % 2 != 0):
        tile = tile[:,:tile.shape[1] - 1];
    dN = tuple(1 +(math.floor(N / ti)) for ti in np.shape(tile));

    row = dN[0];
    col = dN[1];
    
    # to avoid divide by zero errors
    if(dN[0] == 1):
        row = row + 1;
    if(dN[1] == 1):
        col = col + 1;
    
    
    # repeat tile to create initial wallpaper less the excess necessary to complete the wallpaper to desired size
    img = numpy.matlib.repmat(tile, row - 1, col - 1);

    row = math.floor(img.shape[0] + tile.shape[0] * ((1 + (N / tile.shape[0])) - dN[0]));
    col = math.floor(img.shape[1] + tile.shape[1] * ((1 + (N / tile.shape[1])) - dN[1]));
    if (math.floor(img.shape[0] + tile.shape[0] * ((1 + (N / tile.shape[0])) - dN[0])) % 2 != 0):
        row = row + 1;

    if (math.floor(img.shape[1] + tile.shape[1] * ((1 + (N / tile.shape[1])) - dN[1])) % 2 != 0):
        col =  col + 1;
    
    img_final = np.zeros((row, col));
    

    # centers the evenly created tile and then even distributes the rest of the tile around the border s.t. the total size of the wallpaper = the desired input size of the wallpaper
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2),math.ceil((img_final.shape[1] - img.shape[1]) / 2): img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[:,:];
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2) : img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2), : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[ :, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2) : img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2), img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2): ] = img[ :, : math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), math.ceil((img_final.shape[1] - img.shape[1]) / 2) : img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) ] = img[ img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, : ];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2) : , math.ceil((img_final.shape[1] - img.shape[1]) / 2) : img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) ] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), : ];
    img_final[: math.ceil((img_final.shape[0] - img.shape[0]) / 2), : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2):, img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2): , : math.ceil((img_final.shape[1] - img.shape[1]) / 2)] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), img.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2):];
    img_final[img_final.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2): , img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2) : ] = img[ : math.ceil((img_final.shape[0] - img.shape[0]) / 2), :math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    img_final[:math.ceil((img_final.shape[0] - img.shape[0]) / 2) ,  img_final.shape[1] - math.ceil((img_final.shape[1] - img.shape[1]) / 2): ] = img[ img.shape[0] - math.ceil((img_final.shape[0] - img.shape[0]) / 2) : , :math.ceil((img_final.shape[1] - img.shape[1]) / 2)];
    
    img_final[math.ceil((img_final.shape[0] - img.shape[0]) / 2): math.ceil((img_final.shape[0] - img.shape[0]) / 2) + diagTile.shape[0], math.ceil((img_final.shape[1] - img.shape[1]) / 2) : math.ceil((img_final.shape[1] - img.shape[1]) / 2) + diagTile.shape[1]] = diagTile[:, :, 1];

    return img_final

def diagnostic(img, wptype, tile, isFR, isLattice, N, ratio, cmap, isDots):
    # function to take care of all diagnostic tasks related to the wallpaper generation
    # img is the full wallpaper
    # wptype is the wallpaper type
    # tile is a single tile of the wallpaper
    # isFR is if the wallpaper is sized as a ratio of the fundamental region
    # isLattice is if the wallpaper is sized as a ratio of the lattice
    # N is the overall size of the wallpaper
    # ratio is the ratio of the FR/lattice sizing
    
    #tile = filterImg(tile,N);
    #img = filterImg(img,N);
    if (isDots == False):
        tile = np.array(tile * 255, dtype=np.uint8);
        tile[:,:] = cv.equalizeHist(tile[:,:]);

    #img = np.array(img * 255, dtype=np.uint8); 
    #img[:,:] = cv.equalizeHist(img[:,:]);
    saveStr = os.getcwd() + '\\WPSet\\';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr; 
    if (wptype == 'P1'):
        #rgb(47,79,79)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1])):.1f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - tile.shape[0] * tile.shape[1]) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_" + wptype  + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        
        draw.rectangle((0, 0, tile.shape[0], tile.shape[1]), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), fill=(47,79,79, 125), outline=(255,255,0,255), width=2);
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'P2'):
        #rgb(128,0,0)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - (tile.shape[0] * tile.shape[1]) / 2) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0], tile.shape[1]), fill=(128,0,0, 125), outline=(255,255,0,255), width=2);
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.polygon(((2, -1), (-1, 3), (2, 8), (5, 3), (2, -1)), fill=(128,0,0, 125));
        alpha_mask__rec_draw.line(((1, -2), (-1, 4), (3, 9), (6, 4), (1, -2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, -1), (tile.shape[0] + 1, 3), (tile.shape[0] - 2, 8), (tile.shape[0] - 5, 3), ((tile.shape[0] - 2, -1))), fill=(128,0,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, -2), (tile.shape[0] + 1, 4), (tile.shape[0] - 3, 9), (tile.shape[0] - 6, 4), (tile.shape[0] - 1, -2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, tile.shape[1] + 1), (tile.shape[0] + 1, tile.shape[1] - 3), (tile.shape[0] - 2, tile.shape[1] - 8), (tile.shape[0] - 5, tile.shape[1] - 3), ((tile.shape[0] - 2, tile.shape[1] + 1))), fill=(128,0,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, tile.shape[1] + 2), (tile.shape[0] + 1, tile.shape[1] - 4), (tile.shape[0] - 3, tile.shape[1] - 9), (tile.shape[0] - 6, tile.shape[1] - 4), (tile.shape[0] - 1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((2, tile.shape[1] + 1), (-1, tile.shape[1] - 3), (2, tile.shape[1] - 8), (5, tile.shape[1] - 3), (2, tile.shape[1] + 1)), fill=(128,0,0, 125));
        alpha_mask__rec_draw.line(((1, tile.shape[1] + 2), (-1, tile.shape[1] - 4), (3, tile.shape[1] - 9), (6, tile.shape[1] - 4), (1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 - 2), (tile.shape[0] / 2 + 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 2)), fill=(128,0,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 2 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 - 3), (tile.shape[0] / 2 + 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 3)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1] / 2), 4), 4, 345, fill=(128,0,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 4, tile.shape[1] / 2), 4), 4, 345, fill=(128,0,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 4), 4), 4, 15, fill=(128,0,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 4), 4), 4, 15, fill=(128,0,0, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'PM'):
        #rgb(0,128,0)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0], tile.shape[1]), fill=(0,128,0, 125), outline=(255,255,0,255), width=2);
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'PG'):
        #rgb(127,0,127)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 2):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 2)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((tile.shape[0] / 2, 0, tile.shape[0], tile.shape[1]), fill=(127,0,127, 125), outline=(255,255,0,255), width=2);
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'CM'):
        #rgb(143,188,143)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cma = plt.get_cmap("gray");
        tileCm = cma(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2,tile.shape[1]), (tile.shape[0], tile.shape[1] / 2),(tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve");
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (tile.shape[0] / 2,tile.shape[1]), (tile.shape[0], tile.shape[1] / 2)), fill=(143,188,143, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2,tile.shape[1]), (tile.shape[0], tile.shape[1] / 2),(tile.shape[0] / 2, 0)), fill=(255, 255, 0, 255), width=2, joint="curve");
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'PMM'):
        #rgb(255,69,0)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(255,69,0, 125));
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.polygon(((2, -1), (-1, 3), (2, 8), (5, 3), (2, -1)), fill=(255,69,0, 125));
        alpha_mask__rec_draw.line(((1, -2), (-1, 4), (3, 9), (6, 4), (1, -2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, -1), (tile.shape[0] + 1, 3), (tile.shape[0] - 2, 8), (tile.shape[0] - 5, 3), ((tile.shape[0] - 2, -1))), fill=(255,69,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, -2), (tile.shape[0] + 1, 4), (tile.shape[0] - 3, 9), (tile.shape[0] - 6, 4), (tile.shape[0] - 1, -2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 2, tile.shape[1] + 1), (tile.shape[0] + 1, tile.shape[1] - 3), (tile.shape[0] - 2, tile.shape[1] - 8), (tile.shape[0] - 5, tile.shape[1] - 3), ((tile.shape[0] - 2, tile.shape[1] + 1))), fill=(255,69,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] - 1, tile.shape[1] + 2), (tile.shape[0] + 1, tile.shape[1] - 4), (tile.shape[0] - 3, tile.shape[1] - 9), (tile.shape[0] - 6, tile.shape[1] - 4), (tile.shape[0] - 1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((2, tile.shape[1] + 1), (-1, tile.shape[1] - 3), (2, tile.shape[1] - 8), (5, tile.shape[1] - 3), (2, tile.shape[1] + 1)), fill=(255,69,0, 125));
        alpha_mask__rec_draw.line(((1, tile.shape[1] + 2), (-1, tile.shape[1] - 4), (3, tile.shape[1] - 9), (6, tile.shape[1] - 4), (1, tile.shape[1] + 2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 - 2), (tile.shape[0] / 2 + 5, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 2)), fill=(255,69,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 2 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 - 3), (tile.shape[0] / 2 + 6, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2 + 3)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1] / 2), 4), 4, 345, fill=(255,69,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 4, tile.shape[1] / 2), 4), 4, 345, fill=(255,69,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 4), 4), 4, 15, fill=(255,69,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 4), 4), 4, 15, fill=(255,69,0, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'PMG'):
        #rgb(255,165,0)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(255,165,0, 125));
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.polygon(((1, tile.shape[1] / 4), (3, tile.shape[1] / 4 - 5), (6, tile.shape[1] / 4), (3, tile.shape[1] / 4 + 5), (1, tile.shape[1] / 4)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 4), (4, tile.shape[1] / 4 - 6), (7, tile.shape[1] / 4), (4, tile.shape[1] / 4 + 6), (0, tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((1, tile.shape[1] - tile.shape[1] / 4), (3, tile.shape[1] - tile.shape[1] / 4 - 5), (6, tile.shape[1] - tile.shape[1] / 4), (3, tile.shape[1] - tile.shape[1] / 4 + 5), (1, tile.shape[1] - tile.shape[1] / 4)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((0, tile.shape[1] - tile.shape[1] / 4), (4, tile.shape[1] - tile.shape[1] / 4 - 6), (7, tile.shape[1] - tile.shape[1] / 4), (4, tile.shape[1] - tile.shape[1] / 4 + 6), (0, tile.shape[1] - tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 4 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] / 4 - 4), (tile.shape[0] / 2 + 5, tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] / 4 + 2)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] / 4 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] / 4 - 5), (tile.shape[0] / 2 + 6, tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] / 4 + 3)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 2), (tile.shape[0] / 2 - 5, tile.shape[1] - tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 - 4), (tile.shape[0] / 2 + 5, tile.shape[1] - tile.shape[1] / 4 - 1), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 2)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 3), (tile.shape[0] / 2 - 6, tile.shape[1] - tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 - 5), (tile.shape[0] / 2 + 6, tile.shape[1] - tile.shape[1] / 4 - 2), (tile.shape[0] / 2, tile.shape[1] - tile.shape[1] / 4 + 3)), fill=(255, 255, 0, 255), width=1);  
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 1, tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] / 4 - 5), (tile.shape[0] - 6, tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] / 4 + 5), (tile.shape[0] - 1, tile.shape[1] / 4)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] / 4 - 6), (tile.shape[0] - 7, tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] / 4 + 6), (tile.shape[0], tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] - 1, tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] - tile.shape[1] / 4 - 5), (tile.shape[0] - 6, tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 3, tile.shape[1] - tile.shape[1] / 4 + 5), (tile.shape[0] - 1, tile.shape[1] - tile.shape[1] / 4)), fill=(255,165,0, 125));
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] - tile.shape[1] / 4 - 6), (tile.shape[0] - 7, tile.shape[1] - tile.shape[1] / 4), (tile.shape[0] - 4, tile.shape[1] - tile.shape[1] / 4 + 6), (tile.shape[0], tile.shape[1] - tile.shape[1] / 4)), fill=(255, 255, 0, 255), width=1);
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'PGG'):
        #rgb(189,183,107)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2)), fill=(189,183,107, 125));
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((4, 4), 6), 4, 45, fill=(189,183,107, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1]-5), 6), 4, 45, fill=(189,183,107, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, tile.shape[1]-5), 6), 4, 45, fill=(189,183,107, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, 4), 6), 4, 45, fill=(189,183,107, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 45, fill=(189,183,107, 125), outline=(255,255,0));
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (4, tile.shape[1] / 2 - 3), (10, tile.shape[1] / 2), (4, tile.shape[1] / 2 + 3), (0, tile.shape[1] / 2)), fill=(189,183,107, 125));
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (5, tile.shape[1] / 2 - 4), (11, tile.shape[1] / 2), (5, tile.shape[1] / 2 + 4), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, 0), (tile.shape[0] / 2 - 5, 3), (tile.shape[0] / 2, 6), (tile.shape[0] / 2 + 5, 3), (tile.shape[0] / 2, 0)), fill=(189,183,107, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2 - 6, 4), (tile.shape[0] / 2, 7), (tile.shape[0] / 2 + 6, 4), (tile.shape[0] / 2, 0)), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1]), (tile.shape[0] / 2 - 5, tile.shape[1] - 3), (tile.shape[0] / 2, tile.shape[1] - 6), (tile.shape[0] / 2 + 5, tile.shape[1] - 3), (tile.shape[0] / 2, tile.shape[1])), fill=(189,183,107, 125));
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, tile.shape[1]), (tile.shape[0] / 2 - 6, tile.shape[1] - 4), (tile.shape[0] / 2, tile.shape[1] - 7), (tile.shape[0] / 2 + 6, tile.shape[1] - 4), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=1);
        alpha_mask__rec_draw.polygon(((tile.shape[0], tile.shape[1] / 2), (tile.shape[0] - 4, tile.shape[1] / 2 - 3), (tile.shape[0] - 10, tile.shape[1] / 2), (tile.shape[0] - 4, tile.shape[1] / 2 + 3), (tile.shape[0], tile.shape[1] / 2)), fill=(189,183,107, 125));
        alpha_mask__rec_draw.line(((tile.shape[0], tile.shape[1] / 2), (tile.shape[0] - 5, tile.shape[1] / 2 - 4), (tile.shape[0] - 11, tile.shape[1] / 2), (tile.shape[0] - 5, tile.shape[1] / 2 + 4), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=1);
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'CMM'):
        #rgb(0,0,205)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2,tile.shape[1]), (tile.shape[0], tile.shape[1] / 2),(tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve");
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1])), fill=(0,0,205, 125));
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 30, fill=(0,0,205, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((6, tile.shape[1] / 2), 6), 4, 345, fill=(0,0,205, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 6, tile.shape[1] / 2), 6), 4, 345, fill=(0,0,205, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 6), 6), 4, 345, fill=(0,0,205, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 6), 6), 4, 345, fill=(0,0,205, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'P4'):
        #rgb(124,252,0)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 4):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 4)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.rectangle((0, tile.shape[1] / 2, tile.shape[0] / 2, tile.shape[1]), fill=(124,252,0, 125));
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((4, 4), 6), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1]-5), 6), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, tile.shape[1]-5), 6), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, 4), 6), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1] / 2), 4), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 4, tile.shape[1] / 2), 4), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 4), 4), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 4), 4), 4, 45, fill=(124,252,0, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'P4M'):
        #rgb(0,250,154)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((tile.shape[0] / 2, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]),(0, tile.shape[1])), fill=(0,250,154, 125));
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[0] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((0, 0), (tile.shape[0], tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[0], 0), (0, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((4, 4), 6), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1]-5), 6), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, tile.shape[1]-5), 6), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, 4), 6), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1] / 2), 4), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 4, tile.shape[1] / 2), 4), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 4), 4), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 4), 4), 4, 45, fill=(0,250,154, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'P4G'):
        #rgb(65,105,225)
        if(isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 8):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 8)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[0] / 2, 0), (0, tile.shape[1] / 2), (tile.shape[0] / 2,tile.shape[1]), (tile.shape[0], tile.shape[1] / 2),(tile.shape[0] / 2, 0)), fill=(255, 255, 0), width=2, joint="curve");
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1])), fill=(65,105,225, 125));
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0], tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] / 2, 0), (tile.shape[0] / 2, tile.shape[1])), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.line(((0, tile.shape[1] / 2), (tile.shape[0] / 2, tile.shape[1]), (tile.shape[0], tile.shape[1] / 2), (tile.shape[0] / 2, 0), (0, tile.shape[1] / 2)), fill=(255, 255, 0, 255), width=2);
        alpha_mask__rec_draw.rectangle((0, 0, tile.shape[0] - 1, tile.shape[1] - 1), outline=(255,255,0, 255), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((4, 4), 6), 4, 0, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((4, tile.shape[1]-5), 6), 4, 0, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, tile.shape[1]-5), 6), 4, 0, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]-5, 4), 6), 4, 0, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1] / 2), 6), 4, 0, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((6, tile.shape[1] / 2), 6), 4, 45, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0]- 6, tile.shape[1] / 2), 6), 4, 45, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[0] / 2, tile.shape[1]  - 6), 6), 4, 45, fill=(65,105,225, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, 6), 6), 4, 45, fill=(65,105,225, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm);
    elif (wptype == 'P3'):
        #rgb(233,150,122)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 18):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 18)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=3);
        alpha_mask__rec_draw.polygon(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(233,150,122, 125));
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(233,150,122, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(233,150,122, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 0, fill=(233,150,122, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 60, fill=(233,150,122, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] - 5, 3), 5), 3, 210, fill=(233,150,122, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(233,150,122, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm.rotate(270, expand=1));
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm.rotate(270, expand=1));
    elif (wptype == 'P3M1'):
        #rgb(0,191,255)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(0,191,255, 125));
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] / 1.5, tile.shape[0] / 3), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(0,191,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(0,191,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 0, fill=(0,191,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 60, fill=(0,191,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] - 5, 3), 5), 3, 210, fill=(0,191,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(0,191,255, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm.rotate(270, expand=1));
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm.rotate(270, expand=1));
    elif (wptype == 'P31M'):
        #rgb(255,0,255)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(255,0,255, 125));
        alpha_mask__rec_draw.line(((tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] / 1.5, tile.shape[0] / 3)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, 0), (tile.shape[1] / 2, tile.shape[0] / 6), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] - 1, 0)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] / 3), (tile.shape[1] / 1.5, tile.shape[0] / 3), (tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), ((tile.shape[1] - 1), tile.shape[0] / 3)), fill=(255, 255, 0), width=2);        
        alpha_mask__rec_draw.line((((tile.shape[1] / 2, tile.shape[0] / 6), ((tile.shape[1] - 1), tile.shape[0] / 3))), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((((tile.shape[1] - 1, 0), ((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75))), fill=(255, 255, 0), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 6), 5), 3, 210, fill=(255,0,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 3), tile.shape[0] / 3), 5), 3, 210, fill=(255,0,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 1.5, tile.shape[0] / 3), 5), 3, 30, fill=(255,0,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((((tile.shape[1] - 1) / 1.25, tile.shape[0] / 5.75), 5), 3, 30, fill=(255,0,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] - 5, 3), 5), 3, 210, fill=(255,0,255, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon(((tile.shape[1] / 2, tile.shape[0] / 2), 5), 3, 210, fill=(255,0,255, 125), outline=(255,255,0));
        
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype))
        #display(diaLatIm.rotate(270, expand=1));
        display(Markdown('Fundamental Region for ' + wptype))
        #display(diaFRIm.rotate(270, expand=1));
    elif (wptype == 'P6'):
        #rgb(221,160,221)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 36):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 36)) / (N**2 * ratio)) * 100):.2f}%'); 
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1]  - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2);
        diaLatIm.save(diagPath1, "png");
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(221,160,221, 125));
        alpha_mask__rec_draw.line(((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1]  - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((((tile.shape[1] - ((tile.shape[1] - 1) / 3)), tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (tile.shape[0] / 2)), (tile.shape[1] - ((tile.shape[1] - 1) / 5.75), (tile.shape[0] / 1.25)), (tile.shape[1] - ((tile.shape[1] - 1) / 3), tile.shape[0] - 1)), fill=(255, 255, 0), width=2);        
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2))), fill=(255, 255, 0), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5), 5), 3, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0]  - (tile.shape[0] / 2), 3), 4, 45, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] / 2, (tile.shape[0] / 2), 5), 6, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2), 5), 6, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - 5, tile.shape[0] - 5, 5), 6, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 5, 5), 6, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25), 5), 3, 0, fill=(221,160,221, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 3, 3), 4, 45, fill=(221,160,221, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype));
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype));
        #display(diaFRIm);
    elif (wptype == 'P6M'):
        #rgb(255,20,147)
        if (isFR):
            print('Area of Fundamental Region of ' + wptype + f' =  {((tile.shape[0] * tile.shape[1]) / 72):.2f}');
            print('Area of Fundamental Region of ' + wptype + ' should be = ', (N**2 * ratio));
            print(f'Percent Error is approximately = {((np.abs(N**2 * ratio - ((tile.shape[0] * tile.shape[1]) / 72)) / (N**2 * ratio)) * 100):.2f}%');
        diagPath1 = sPath + "_DIAGNOSTIC_LATTICE_"  + wptype + '.' + "png";
        cm = plt.get_cmap("gray");
        tileCm = cm(tile);
        if(isDots):
            diaLatIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
            draw = ImageDraw.Draw(diaLatIm, 'RGBA');
        else:
            diaLatIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
            draw = ImageDraw.Draw(diaLatIm);
        draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1]  - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2);
        diaLatIm.save(diagPath1, "png");
        
        diagPath2 = sPath + "_DIAGNOSTIC_FR_"  + wptype + '.' + "png";
        if(isDots):
            diaFRIm = Image.fromarray((tile[:, :]).astype(np.uint32), 'RGBA');
        else:
            diaFRIm = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));
        alpha_mask_rec = Image.new('RGBA', diaFRIm.size, (0,0,0,0));
        alpha_mask__rec_draw = ImageDraw.Draw(alpha_mask_rec);
        alpha_mask__rec_draw.polygon(((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0]  - (tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255,20,147, 125));
        alpha_mask__rec_draw.line((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0]  - (tile.shape[0] / 2), ((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5)))), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2)), (tile.shape[1] / 2, (tile.shape[0] / 2)), (tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5))), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), (tile.shape[1]  - (tile.shape[1] / 6), tile.shape[0] / 2), (tile.shape[1] / 2, tile.shape[0] / 2), (tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), (tile.shape[1] - 1, tile.shape[0] - 1)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((((tile.shape[1] - ((tile.shape[1] - 1) / 3)), tile.shape[0] - 1), (tile.shape[1] - (tile.shape[1] / 3), (tile.shape[0] / 1.5)), (tile.shape[1] - (tile.shape[1] / 6), tile.shape[0] - (tile.shape[0] / 2)), (tile.shape[1] - ((tile.shape[1] - 1) / 5.75), (tile.shape[0] / 1.25)), (tile.shape[1] - ((tile.shape[1] - 1) / 3), tile.shape[0] - 1)), fill=(255, 255, 0), width=2);        
        alpha_mask__rec_draw.line(((tile.shape[1] - 1, tile.shape[0] - 1), tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5), tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line(((tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 1), tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25)), fill=(255, 255, 0), width=2);
        alpha_mask__rec_draw.line((((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 1), tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2))), fill=(255, 255, 0), width=2);
        
        #symmetry axes symbols
        alpha_mask__rec_draw.regular_polygon((tile.shape[1]  - (tile.shape[1] / 3), (tile.shape[0] / 1.5), 5), 3, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0]  - (tile.shape[0] / 2), 3), 4, 45, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] / 2, (tile.shape[0] / 2), 5), 6, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 6), tile.shape[0]  - (tile.shape[0] / 2), 5), 6, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - 5, tile.shape[0] - 5, 5), 6, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] / 3), tile.shape[0] - 5, 5), 6, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, (tile.shape[0] / 1.25), 5), 3, 0, fill=(255,20,147, 125), outline=(255,255,0));
        alpha_mask__rec_draw.regular_polygon((tile.shape[1] - (tile.shape[1] - 1) / 5.75, tile.shape[0] - 3, 3), 4, 45, fill=(255,20,147, 125), outline=(255,255,0));
        
        diaFRIm = Image.alpha_composite(diaFRIm, alpha_mask_rec);
        
        diaFRIm.save(diagPath2, "png");
        #display(Markdown('Lattice for ' + wptype));
        #display(diaLatIm);
        display(Markdown('Fundamental Region for ' + wptype));
        #display(diaFRIm);
    #diaFRImTile = Image.fromarray((tileCm[:, :, :] * 255).astype(np.uint8));


    diagWallpaper = diagCatTiles(tile, N, np.array(diaFRIm).astype(np.uint32));
    #print(diagWallpaper);
    #display(diagWallpaper)
    if isDots:
        display(Image.fromarray((diagWallpaper[:, :]).astype(np.uint32), 'RGBA'));
    else:
        display(Image.fromarray((diagWallpaper[:, :]).astype(np.uint8)));
    if (isDots == False):
        # diagnostic plots
        logging.getLogger('matplotlib.font_manager').disabled = True;
        patternPath = sPath + wptype  + '_diagnostic_1' + '.' + "png";
        hidx_0 = int(img.shape[0] * (1/3));
        hidx_1 = int(img.shape[0] / 2);
        hidx_2 = int(img.shape[0] * (2/3));
        I = np.dstack([img,img,img]);
        I[hidx_0-2:hidx_0+2,:] = np.array([1,0,0]);
        I[hidx_1-2:hidx_1+2,:] = np.array([0,1,0]);
        I[hidx_2-2:hidx_2+2,:] = np.array([0,0,1]);
        cm = plt.get_cmap("gray");
        cm(I);
        #ax = plt.subplot()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,30));
        
        ax1.imshow(I);
        ax1.set_title(wptype + ' diagnostic image 1');
        ax1.set(adjustable='box', aspect='auto')
        #plt.show();
        #plt.savefig(patternPath);
        
        #plt.clf();
        #ax = plt.subplot()
        patternPath = sPath + wptype  + '_diagnostic_2' + '.' + "png";
        ax2.plot(img[hidx_0,:],c=[1,0,0])
        ax2.plot(img[hidx_1,:],c=[0,1,0])
        ax2.plot(img[hidx_2,:],c=[0,0,1])
        ax2.set_title('Sample values along the horizontal lines {} {} and {}'.format(hidx_0, hidx_1, hidx_2));
        #ax2.set(adjustable='box', aspect='equal')
        #plt.show();
        #plt.savefig(patternPath);
        #plt.clf();
        #ax = plt.subplot()
        patternPath = sPath + wptype  + '_diagnostic_3' + '.' + "png";
        bins = np.linspace(0, 1, 100)
        ax3.hist(img[hidx_0,:],bins,color=[1,0,0]) ;
        ax3.hist(img[hidx_1,:],bins,color=[0,1,0]) ;
        ax3.hist(img[hidx_2,:],bins,color=[0,0,1]) ;
        ax3.set_title('Frequency of sample values across the horizontal lines');
        #ax3.set(adjustable='box', aspect='equal')
        
        
        plt.show();
        #plt.savefig(patternPath);

#old cat function
# def catTiles(tile, N, wptype):
#     #disp tile square
#     sq = np.shape(tile)[0] * np.shape(tile)[1];
#     print(wptype + ' area of tile = ', sq);                
    
#     #write tile
#     #tileIm = Image.fromarray(tile);
#     #tileIm.save('~/Documents/PYTHON/tiles/' + wptype + '_tile.jpeg', 'JPEG');
#     dN = tuple((1 + math.floor(N / ti)) for ti in np.shape(tile))
#     #dN = 1 + math.floor(N / np.shape(tile));
#     img = numpy.matlib.repmat(tile, dN[0], dN[1]);
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