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

def generateWPTImagesMain(groups: list=['P1','P2','P4','P3','P6'], nGroup: int=100, visualAngle: float=30.0, distance: float=30.0, tileArea: int=150*150, latticeSize: bool=False,
                          fundRegSize: bool=False, ratio: float=1.0, saveFmt: str="png", saveRaw: bool=False, printAnalysis: bool=False, pssscrambled: bool=False, psscrambled: bool=False, new_mag: bool=False, 
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
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    wpSize = (round((math.tan(math.radians(visualAngle / 2)) * (2 * distance)) * dpi / 2.54));
    app.quit();   
    
    # save parameters
    saveStr = os.getcwd() + '\\WPSet\\';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr;    
    
    # define group to index mapping
    keySet = groups;
    
    # useful parameters for debugging
    if (debug == True):
        nGroup = 1;
        #ratio = 1;
        #keySet = ['P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'];
        keySet = ['PM'];
        wpSize = 300;
    
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
    
    # Generating WPs and scrambling 
    for i in range(len(Groups)):    
        print('generating ', Groups[i]);
        group = Groups[i];
        if (latticeSize == True):
            size = sizeLattice (ratio, wpSize, group);
            wpSize = size[1];
            n = size[0];
        elif (fundRegSize == True):
            size = sizeFundamentalRegion(ratio, wpSize, group);
            wpSize = size[1];
            n = size[0];
        else:
            size = round(math.sqrt(tileArea));
        
        #n = 80
        raw = gwi.generateWPimage(group, wpSize, n);
        cm = plt.get_cmap(cmap);
        raw_image =  cm(raw);
        rawFreq = np.fft.fft2(raw, (raw.shape[0], raw.shape[1]));
        avgMag = np.array([]); 
        if(new_mag == True):
            avgMag = meanMag(rawFreq);
        # generating wallpapers, saving freq. representations
        for k in range(nGroup):
            
            # image processing steps
            
            # get average magnitude
            
            
            avgRaw = spectra(raw, new_mag=avgMag); # replace each image's magnitude with the average
            filtered = cm(filterImg(avgRaw, wpSize)); # low-pass filtering + histeq
            
            #masked = maskImg(filtered, wpSize); # masking the image (final step)
            #Image.fromarray((masked[:, :, :3] * 255).astype(np.uint8)).show();
            
            # making scrambled images
            #scrambled_raw = spectra(raw, pssscrambled, psscrambled, cmap=cmap); # only give spectra only arg, to make randoms
            #scrambled_filtered = cm(filterImg(scrambled_raw, wpSize));
            #scrambled_masked = maskImg(scrambled_filtered, wpSize);
            #print(scrambled_masked);
            #scrambled_masked = maskImg(scrambled_filtered, wpSize);
            #print(scrambled_masked);
            #scrambled_masked[scrambled_masked == 0.5] = gray_cm(scrambled_masked); 
            #Image.fromarray(np.hstack(((masked[:, :, :3] * 255).astype(np.uint8), (scrambled_masked[:, :, :3] * 255).astype(np.uint8)))).show();
            groupNumber = mapgroup[group];
            # saving averaged and scrambled images
            
            if(printAnalysis):
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(sPath + "analysis\\steps_" + group + "_" + str(k), "JPEG");
                #imwrite(all_in_one{img},  strcat(sPath, 'analysis/steps_',group, '_', num2str(img), '.jpeg'), 'jpeg');
            if(saveRaw):
                rawPath = sRawPath + group + '_' + str(k) + '.' + saveFmt;
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(rawPath, saveFmt);
            
            patternPath = sPath + str(1000*groupNumber + k) + '_' + group + '_' + cmap + '.' + saveFmt;
            
            Image.fromarray((filtered[:, :, :3] * 255).astype(np.uint8)).save(patternPath, saveFmt);
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
    image = np.array(image * 255, dtype=np.uint8);
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
        print(list(v.split(",")));
        return list(v.split(","));

def sizeFundamentalRegion (ratio, n, cellStruct):
    if (cellStruct == "rhomb"):
        #0...1:1 aspect ratio
        return n;
    elif (cellStruct == "recttb"):
        return n;
    elif (cellStruct == "PM"):
        return round(np.sqrt((n**2 * ratio) * 4));
    elif (cellStruct == "PMM"):
        return round(np.sqrt((n**2 * ratio) * 4));
    elif (cellStruct == "CMM"):
        return round(np.sqrt((n**2 * ratio) * 4));
    elif (cellStruct == "squarec"):
        return n;
    elif (cellStruct == "squarerc"):
        return n;
    elif (cellStruct == "hex"):
        return n;
    elif (cellStruct == "hextm"):
        return n;
    elif (cellStruct == "hextb"):
        return n;
    elif (cellStruct == "hextbc"):
        return n;
    else:
        #P1 tile square cellStructure
        return n * ratio * 2;

def sizeLattice (ratio, n, cellStruct):
    #aRatio = (ratio * n) / n;
    if (cellStruct == "CMM"):
        #rhombus
        tileArea = round(np.sqrt((n**2 * ratio) * 2));
        if (tileArea % 2 == 0):
            wpSize = n;
        else:
            wpSize = np.sqrt(((((tileArea + 1)**2) / ratio) / 2));
        return (tileArea, wpSize);
    #For square lattice = 'P1', 'CM', 'PMG', 'PGG','PMG', 'PMM', 'P4', 'P4M', 'P4G'
    elif (cellStruct == "P1" or cellStruct == "CM" or cellStruct == "PMM" or cellStruct == "PMG" or cellStruct == "PGG" or cellStruct == "P4" or cellStruct == "P4M" or cellStruct == "P4G"):
        #square
        tileArea = round(np.sqrt((n**2 * ratio)));
        if (tileArea % 2 == 0):
            wpSize = n;
        else:
            wpSize = np.sqrt((((tileArea + 1)**2) / ratio));
        return (tileArea, wpSize);
    elif (cellStruct == "P2" or cellStruct == "PM" or cellStruct == "PG"):
        #rectangle
        tileArea = round(np.sqrt((n**2 * ratio) * 2));
        if (tileArea % 2 == 0):
            wpSize = n;
        else:
            wpSize = np.sqrt(((((tileArea + 1)**2) / ratio) / 2));
        return (tileArea, wpSize);
    elif (cellStruct == "P3"):
        #print(math.floor((np.sqrt(((3 * (ratio * n**2)) / 2)) / 1.5) * np.sqrt(3 * np.tan(np.pi / 3))));
        tileArea = round((np.sqrt(((3 * (ratio * n**2)) / 2)) / 1.5) * np.sqrt(3 * np.tan(np.pi / 3)));
        if (tileArea % 2 == 0):
            wpSize = n;
        else:
            wpSize = np.sqrt((((((tileArea + 1) / np.sqrt(3 * np.tan(np.pi / 3)))**2) * 1.5) / 3) / ratio);
        return (tileArea, wpSize);
    else:
        return math.floor(n * ratio * 2);

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
	parser.add_argument('--nGroup', '-n', default=100, type=int,
                    help='Number of images per group')
	parser.add_argument('--visualAngle', '-v', default=30.0, type=str,
                    help='Wallpaper size (visual angle)')
	parser.add_argument('--distance', '-d', default=30.0, type=str,
                    help='Distance beteween eye and wallpaper')
	parser.add_argument('--tileArea', '-t', default=150*150, type=str,
                    help='Tile area')
	parser.add_argument('--latticeSize', '-l', default=False, type=str2bool,
                    help='Size wallpaper as a ratio between the lattice and wallpaper size')
	parser.add_argument('--fundRegSize', '-fr', default=False, type=str2bool,
                    help='Size wallpaper as a ratio between the fundamental region and wallpaper size')
	parser.add_argument('--ratio', '-ra', default=1.0, type=str,
                    help='Size wallpaper as a ratio')
	parser.add_argument('--saveFmt', '-f', default="png", type=str,
                    help='Image save format')
	parser.add_argument('--saveRaw', '-r', default=False, type=str2bool,
                    help='save raw')
	parser.add_argument('--printAnalysis', '-a', default=False, type=str2bool,
                    help='print analysis')
	parser.add_argument('--pssscrambled', '-s', default=False, type=str2bool,
                    help='Portilla-Simoncelli scrambled')
	parser.add_argument('--psscrambled', '-p', default=False, type=str2bool,
                    help='phase scrambled')
	parser.add_argument('--new_mag', '-m', default=False, type=str2bool,
                    help='new magnitude')
	parser.add_argument('--cmap', '-c', default="gray", type=str,
                    help='color or greyscale map (hsv or gray)')
	parser.add_argument('--debug', '-b', default=False, type=str2bool,
                    help='debugging default parameters on')

	args = parser.parse_args()

	generateWPTImagesMain(args.groups, args.nGroup, float(eval("args.visualAngle")), float(eval("args.distance")), round(eval("args.tileArea")), args.latticeSize, args.fundRegSize, float(eval("args.ratio")), args.saveFmt, args.saveRaw, 
                       args.printAnalysis, args.pssscrambled, args.psscrambled, args.new_mag, args.cmap, args.debug);
