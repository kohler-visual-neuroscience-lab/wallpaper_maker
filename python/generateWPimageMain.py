"""

The code will generate an arbitrary number of exemplars belonging to each of the 17 groups, as well as matched control exemplars using phase-scrambling and Portilla-Simoncelli scrambling.

To run the code use the following function with the available optional parameters: 

generateWPTImagesMain(groups to create, number of images per group, wallpaper size (visual angle), distance beteween eye and wallpaper, .... 
tile area, image save format, save raw, print analysis, Portilla-Simoncelli scrambled, phase scrambled, new magnitude, color or greyscale map, debug parameters on/off)

Check function for expected data types for each argument.

"""
import os
from datetime import datetime
import numpy as np
import math
from PIL import Image
import generateWPimage as gwi
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2 as cv
import sys
from PyQt5.QtWidgets import QApplication
np.set_printoptions(threshold=sys.maxsize)
import texture_synthesis_g as pss
import logging
import argparse

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

def generateWPTImagesMain(groups: list=['P1','P2','P4','P3','P6'], nGroup: int=10, visualAngle: float=30.0, wpSize: int=500, latticeSize: bool=False,
                          fundRegSize: bool=False, ratio: float=1.0, spatFreqFilt: bool=False, spatFreqFiltFWHM: int=5, spatFreqFiltLowpass: bool=True, saveFmt: str="png", saveRaw: bool=False, printAnalysis: bool=False, pssscrambled: bool=False, psscrambled: bool=False, new_mag: bool=False, 
                          cmap: str="gray", debug: bool=False):
    #mapGroup = containers.Map(keySet, valueSet);
    #hexLattice = ['P3', 'P6'];
    #sqrLattice = ['P4'];
    #recLattice = [];
    #rhoLattice = [];
    #obqLattice = ['P1', 'P2'];
    # hexLattice = ['P3', 'P3M1', 'P31M', 'P6', 'P6M'];
    # sqrLattice = ['P4', 'P4M', 'P4G'];
    # recLattice = ['PM', 'PMM', 'PMG', 'PGG', 'PG'];
    # rhoLattice = ['CM', 'CMM'];
    # obqLattice = ['P1', 'P2'];
    # define groups to be generated
    #Groups = ['P1','P2','P4','P3','P6'];
    #Groups = ['P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'];
    # image parameters
    # image size determined by visual angle
    #app = QApplication(sys.argv)
    #screen = app.screens()[0]
    #dpi = screen.physicalDotsPerInch()
    #wpSize = (round((math.tan(math.radians(visualAngle / 2)) * (2 * distance)) * dpi / 2.54));
    #app.quit();   

    # save parameters
    saveStr = os.getcwd() + '\\WPSet\\';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr;    
    
    # define group to index mapping
    keySet = groups;
    
    # useful parameters for debugging
    if (debug == True):
        #nGroup = 1;
        #ratio = 1;
        keySet = ['P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'];
        #keySet = ['P3'];
        #wpSize = 300;
    
    valueSet = np.arange(101, 101 + len(keySet), 1);
    mapgroup = {};
    for i in range(valueSet.shape[0]):
        mapgroup[keySet[i]] = valueSet[i];
    Groups = keySet;
    sRawPath = '';
    try:
        os.mkdirs(sPath);
        if(saveRaw):
            sRawPath = sPath + 'raw\\';
            os.mkdirs(sRawPath);
        if(printAnalysis):
            sAnalysisPath = sPath + 'analysis\\';
            os.mkdirs(sAnalysisPath);
    except:
        print('PYTHON:generateWPSet:mkdir ', sPath);
    isDots = True;
    # Generating WPs and scrambling 
    for i in range(len(Groups)):    
        print('generating ', Groups[i]);
        group = Groups[i];
        if (latticeSize == True):
            n = sizeLattice (ratio, wpSize, group);
        elif (fundRegSize == True):
            n = sizeFundamentalRegion(ratio, wpSize, group);
        else:
            n = sizeTile (ratio, wpSize, group);
        for k in range(nGroup):
            raw = gwi.generateWPimage(group, wpSize, int(n), ratio, visualAngle, False, spatFreqFilt, spatFreqFiltFWHM, spatFreqFiltLowpass, isDots);
            cm = plt.get_cmap(cmap);
            raw_image =  (raw);
            rawFreq = np.fft.fft2(raw, (raw.shape[0], raw.shape[1]));
            avgMag = np.array([]); 
            if(new_mag == True):
                avgMag = meanMag(rawFreq);
        # generating wallpapers, saving freq. representations
        
            
            # image processing steps
            
            # making regular images
            #avgRaw = (spectra(raw, new_mag=avgMag)); # replace each image's magnitude with the average
            #filtered = (filterImg(avgRaw, wpSize)); # low-pass filtering + histeq
            #masked = cm(maskImg(filtered, wpSize)); # masking the image (final step)
            #Image.fromarray((masked[:, :, :3] * 255).astype(np.uint8)).show();
            
            # making scrambled images
            #scrambled_raw = spectra(raw, pssscrambled, psscrambled, cmap=cmap); # only give spectra only arg, to make randoms
            #scrambled_filtered = (filterImg(scrambled_raw, wpSize));
            #scrambled_masked = cm(maskImg(scrambled_filtered, wpSize)); 
            #Image.fromarray(np.hstack(((masked[:, :, :3] * 255).astype(np.uint8), (scrambled_masked[:, :, :3] * 255).astype(np.uint8)))).show();
            groupNumber = mapgroup[group];
            
            # saving averaged and scrambled images
            if(printAnalysis):
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(sPath + "analysis\\steps_" + group + "_" + str(k), "JPEG");
                #imwrite(all_in_one{img},  strcat(sPath, 'analysis/steps_',group, '_', num2str(img), '.jpeg'), 'jpeg');
            if(saveRaw):
                rawPath = sRawPath + group + '_' + str(k) + '.' + saveFmt;
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(rawPath, saveFmt);
            
            if (spatFreqFilt):
                patternPath = sPath + str(1000*groupNumber + k) + '_' + group + '_' + cmap  + '_FWHM_' + str(spatFreqFiltFWHM) + '.' + saveFmt;
            else:
                patternPath = sPath + str(1000*groupNumber + k) + '_' + group + '_' + cmap  + '.' + saveFmt;
            raw_image_c = (200*(raw_image - np.min(raw_image))/np.ptp(raw_image)).astype(np.uint32)        
            Image.fromarray((raw_image_c[:, :]  * 255).astype(np.uint32)).save(patternPath, "png");
            #Image.fromarray((raw_image[:, :] * 255).astype(np.uint8)).save(patternPath, saveFmt);
            if(pssscrambled == True or psscrambled == True):
                if (spatFreqFilt):
                    scramblePath = sPath + str(1000*(groupNumber + 17) + k) + '_' + group + '_Scrambled' + '_' + cmap + '_FWHM_' + str(spatFreqFiltFWHM) + '.' + saveFmt;
                else:
                    scramblePath = sPath + str(1000*(groupNumber + 17) + k) + '_' + group + '_Scrambled' + '_' + cmap + '.' + saveFmt; 
                #Image.fromarray((scrambled_masked[:, :, :3] * 255).astype(np.uint8)).save(scramblePath, saveFmt);
           
            
        #all_in_one = cellfun(@(x,y,z) cat(2,x(1:wpSize,1:wpSize),y(1:wpSize,1:wpSize),z(1:wpSize,1:wpSize)),raw,avgRaw,filtered,'uni',false);
        
        # variables for saving in a mat file
        #symAveraged[:,i]= np.concatenate((avgRaw, scrambled_raw));
        #symFiltered[:,i]= np.concatenate((filtered, scrambled_filtered));
        #symMasked[:,i]= np.concatenate((masked, scrambled_masked));
    #save([sPath,timeStr,'.mat'],'symAveraged','symFiltered','symMasked','Groups');

def matlab_style_gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss - 1.) / 2. for ss in shape]
    y,x = np.ogrid[-m:m + 1,-n:n + 1]
    h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# Filter/mask every image
def filterImg(inImg, N):        
    # Make filter intensity adaptive (600 is empirical number)
    sigma = N / 600;
    lowpass = matlab_style_gauss2D((9, 9), sigma);
    
    # filter
    image = scipy.ndimage.correlate(inImg, lowpass, mode='constant').transpose();
    
    # histeq
    image = np.array(image * 255, dtype=np.uint8); #changed to inImg from image to stop low pass
    image[:,:] = cv.equalizeHist(image[:,:]);

    # normalize
    image = image / np.ptp(image); #scale to unit range
    image = image - np.mean(image[:]); #bring mean luminance to zero		
    image = image/ np.max(np.abs(image[:])); #Scale so max signed value is 1
    image = 125 * image + 127; # Scale into 2-252 range
    image = image / 255;

    outImg = image;
    return outImg;

# apply mask
def maskImg(inImg, N):
    # define mask(circle)
    r = round(0.5 * N);
    mask = np.zeros((inImg.shape[0],inImg.shape[1]), np.uint8);
    cv.circle(mask, (round(inImg.shape[1] / 2), round(inImg.shape[0] / 2)), r, 1, -1);
    mask = cv.bitwise_and(mask, mask, mask=mask);
    outImg = inImg[:np.shape(mask)[0], :np.shape(mask)[1]];
    outImg[mask==0] = 0.5;
    return outImg;

# replace spectra
def spectra(in_image, pssscrambled=False, psscrambled=False, new_mag=np.array([]), cmap="gray"):
    in_spectrum = np.fft.fft2(in_image, (in_image.shape[0], in_image.shape[1]));
    phase = np.angle(in_spectrum);
    mag = np.abs(in_spectrum);
    # phase scrambling
    if (psscrambled == True):
        randPhase = np.fft.fft2(np.random.rand(in_image.shape[0], in_image.shape[1]), (in_image.shape[0], in_image.shape[1]));
        phase = np.angle(randPhase);
        rng = np.random.default_rng()
        [rng.shuffle(x) for x in phase];
    # Portilla-Simoncelli scrambling
    elif(pssscrambled == True):
        outImage = psScramble(in_image, cmap);
        return outImage;
    # use new magnitude instead
    if(new_mag.size != 0):
        mag = new_mag;
    cmplxIm = mag * np.exp(1j * phase);
    #get the real parts and then take the absolute value of the real parts as this is the closest solution to be found to emulate matlab's ifft2 "symmetric" parameter
    outImage = np.abs(np.real(np.fft.ifft2(cmplxIm)));
    return outImage;

#perform Portilla-Simoncelli scrambling
def psScramble(in_image, cmap):
    imagetmp = Image.fromarray(in_image);
    #resize image to nearest power of 2 to make use of the steerable pyramid for PS
    newSize = previous_power_2(in_image.shape[0]);
    imagetmp = imagetmp.resize((newSize, newSize), Image.BICUBIC);
    in_image = np.array(imagetmp);
    outImage = pss.synthesis(in_image, in_image.shape[0], in_image.shape[1], 5, 4, 7, 25)
    return outImage

def previous_power_2(x):
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);

def str2bool(v):
    if isinstance(v, bool):
       return v;
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True;
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False;
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.');

def str2list(v):
    if isinstance(v, list):
        return v;
    else:
        return list(v.split(","));

def sizeFundamentalRegion (ratio, n, cellStruct):
    # 0...1:1 aspect ratio
    if (cellStruct == "P1"):
        return round(np.sqrt((n**2 * ratio)));
    elif (cellStruct == "P2" or cellStruct == "PG" or cellStruct == "PM"):
        return round(np.sqrt((n**2 * ratio)) * np.sqrt(2));
    elif (cellStruct == "PMM" or cellStruct == "CMM" or cellStruct == "PMG" or cellStruct == "PGG" or cellStruct == "P4" or cellStruct == "CM"):
        return round(np.sqrt((n**2 * ratio) * 4));
    elif (cellStruct == "P4M" or cellStruct == "P4G"):
        return round(np.sqrt((n**2 * ratio) * 8));
    elif (cellStruct == "P3"):
        # equilateral rhombus
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * (n**2 * ratio * 3)));
    elif (cellStruct == "P3M1"):
        # equilateral triangle
        return round(np.sqrt(((n**2 * ratio) * 3 / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)));
    elif (cellStruct == "P31M" or cellStruct == "P6"):
        # isosceles triangle
        # solved symbolically using mathematical software (maple)
        return round (6 * np.sqrt((n**2 * ratio)));
    elif (cellStruct == "P6M"):
        # right angle triangle
        # solved symbolically using mathematical software (maple)
        return round (6 * np.sqrt(2) * np.sqrt((n**2 * ratio)));
    else:
        return round(n * ratio * 2);

def sizeLattice (ratio, n, cellStruct):
    # 0...1:1 aspect ratio
    if (cellStruct == "P1" or cellStruct == "PMM" or cellStruct == "PMG" or cellStruct == "PGG" or cellStruct == "P4" or cellStruct == "P4M" or cellStruct == "P4G" or cellStruct == "P2" or cellStruct == "PM" or cellStruct == "PG"):
        # square and rectangular
        return round(np.sqrt((n**2 * ratio)));
    elif (cellStruct == "CM" or cellStruct == "CMM"):
        # rhombic
        return round(np.sqrt((n**2 * ratio) * 2));
    elif (cellStruct == "P3"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * ((n**2 * ratio))));
    elif (cellStruct == "P3M1"):
        # hexagonal
        return round(np.sqrt((((n**2 * ratio)  / 6) / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)));
    elif (cellStruct == "P31M" or cellStruct == "P6" or cellStruct == "P6M"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round (np.sqrt(2) * np.sqrt((n**2 * ratio)));
    else:
        return round(n * ratio * 2);
    
def sizeTile (ratio, n, cellStruct):
    # 0...1:1 aspect ratio
    if (cellStruct == "P1" or cellStruct == "PMM" or cellStruct == "PMG" or cellStruct == "PGG" or cellStruct == "P4" or cellStruct == "P4M" or cellStruct == "P4G" or cellStruct == "P2" or cellStruct == "PM" or cellStruct == "PG" or cellStruct == "CM" or cellStruct == "CMM"):
        # square and rectangular and rhombic
        return round(np.sqrt((n**2 * ratio)));
    elif (cellStruct == "P3"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round(1.316074013 + 2.999999999 * 10**-10 * np.sqrt(1.924500897 * 10**19 + 6.666666668 * 10**19 * ((n**2 * ratio) / 6)));
    elif (cellStruct == "P3M1"):
        # hexagonal
        return round(np.sqrt((((n**2 * ratio)  / 12) / 0.25) * np.tan(np.pi / 3) * np.sqrt(3)));
    elif (cellStruct == "P31M" or cellStruct == "P6" or cellStruct == "P6M"):
        # hexagonal
        # solved symbolically using mathematical software (maple)
        return round (np.sqrt(2) * np.sqrt((n**2 * ratio) / 2));
    else:
        return round(n * ratio * 2);

# returns average mag of the group
def meanMag(freqGroup):
    nImages = 1;
    mag = np.empty((freqGroup.shape[0], freqGroup.shape[1], nImages));
    for n in range(nImages):
        mag[:,:, n] = np.abs(freqGroup);
    out = np.median(mag,2);
    return out;

#generateWPTImagesMain(debug=True);

if __name__ == "__main__":
	LOGGER.info('Generating Wallpapers')

	parser = argparse.ArgumentParser(
	    description='Wallpaper Generator')
	parser.add_argument('--groups', '-g', default=['P1','P2','P4','P3','P6'], type=str2list, #need to write function to convert str to list
                    help='Groups to create')
	parser.add_argument('--nGroup', '-n', default=10, type=int,
                    help='Number of images per group')
	parser.add_argument('--visualAngle', '-v', default=30.0, type=float,
                    help='Wallpaper size (visual angle)')
	parser.add_argument('--wallpaperSize', '-ws', default=500, type=int,
                    help='Side length of the wallpaper in pixels')
	parser.add_argument('--latticeSize', '-l', default=False, type=str2bool,
                    help='Size wallpaper as a ratio between the lattice and wallpaper size')
	parser.add_argument('--fundRegSize', '-fr', default=False, type=str2bool,
                    help='Size wallpaper as a ratio between the fundamental region and wallpaper size')
	parser.add_argument('--ratio', '-ra', default=1.0, type=float,
                    help='Size wallpaper as a ratio')
	parser.add_argument('--spatFreqFilt', '-sff', default=False, type=str2bool,
                    help='Replace the fundamental region with random noise whoses frequency is relative to the visual angle')
	parser.add_argument('--spatFreqFiltFWHM', '-fwhm', default=5, type=int,
                    help='Set fwhm for spatial frequency filtering')
	parser.add_argument('--spatFreqFiltLowpass', '-sfflp', default=True, type=str2bool,
                    help='Set spatial frequency filtering to lowpass filtering otherwise it is highpass filtering')
	parser.add_argument('--saveFmt', '-f', default="png", type=str,
                    help='Image save format')
	parser.add_argument('--saveRaw', '-r', default=False, type=str2bool,
                    help='save raw')
	parser.add_argument('--printAnalysis', '-a', default=False, type=str2bool,
                    help='Print analysis')
	parser.add_argument('--pssscrambled', '-s', default=False, type=str2bool,
                    help='Portilla-Simoncelli scrambled')
	parser.add_argument('--psscrambled', '-p', default=False, type=str2bool,
                    help='Phase scrambled')
	parser.add_argument('--new_mag', '-m', default=False, type=str2bool,
                    help='New magnitude')
	parser.add_argument('--cmap', '-c', default="gray", type=str,
                    help='Color or greyscale map (hsv or gray)')
	parser.add_argument('--debug', '-b', default=False, type=str2bool,
                    help='Debugging default parameters on')

	args = parser.parse_args()
    
    #need to investigate error in eval function
	generateWPTImagesMain(args.groups, args.nGroup, args.visualAngle, args.wallpaperSize, args.latticeSize, args.fundRegSize, float(eval("args.ratio")), args.spatFreqFilt, args.spatFreqFiltFWHM, args.spatFreqFiltLowpass, args.saveFmt, args.saveRaw, 
                       args.printAnalysis, args.pssscrambled, args.psscrambled, args.new_mag, args.cmap, args.debug);

