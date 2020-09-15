"""

The code will generate an arbitrary number of exemplars belonging to each of the 17 groups, as well as matched control exemplars using phase-scrambling and Portilla-Simoncelli scrambling.

To run the code use the following function with the available optional parameters: 

generateWPTImagesMain(groups to create, number of images per group, wallpaper size (visual angle), distance beteween eye and wallpaper, .... 
tile area, image save format, save raw, print analysis, Portilla-Simoncelli scrambled, phase scrambled, Portilla-Simoncelli scrambled, new magnitude, color or greyscale map)

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

def generateWPTImagesMain(groups: list=['P1','P2','P4','P3','P6'], nGroup: int=100, visualAngle: float=30.0, distance: float=30.0, tileArea: int=100*100, saveFmt: str="png", saveRaw: bool=False, printAnalysis: bool=False, pssscrambled: bool=False, psscrambled: bool=False, new_mag: bool=False, cmap: str="gray"):
    # define group to index mapping
    keySet = ['P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'];
    #keySet = ['P1', 'P2', 'P3', 'P4', 'P6'];
    #keySet = groups;
    valueSet = np.arange(101, 100 + len(keySet) + 1, 1);
    mapgroup = {};
    for i in range(valueSet.shape[0]):
        mapgroup[keySet[i]] = valueSet[i];
    #mapGroup = containers.Map(keySet, valueSet);
    
    hexLattice = ['P3', 'P6'];
    sqrLattice = ['P4'];
    recLattice = [];
    rhoLattice = [];
    obqLattice = ['P1', 'P2'];
    # hexLattice = {'P3', 'P3M1', 'P31M', 'P6', 'P6M'};
    # sqrLattice = {'P4', 'P4M', 'P4G'};
    # recLattice = {'PM', 'PMM', 'PMG', 'PGG', 'PG'};
    # rhoLattice = {'CM', 'CMM'};
    # obqLattice = {'P1', 'P2'};
    # define groups to be generated
    #Groups = ['P1','P2','P4','P3','P6'];
    #Groups = ['P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM', 'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'];
    Groups = keySet;
    # number of images per group
    nGroup = 1;
    
    # image parameters
    # image size
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    wpSize = (round((math.tan(math.radians(visualAngle / 2)) * (2 * distance)) * dpi / 2.54));
    app.quit();
    #wpSize = 512;
    #wpSize = math.tan(wpSize / 2) * (2 * 10**3) * 37.7952755906;
    # area of tile that will be preserved across groups
    #tileArea = 150 * 150;    
    
    # Average magnitude within the each group
    # save parameters
    saveStr = os.getcwd() + '\\WPSet\\';
    today = datetime.today();
    timeStr = today.strftime("%Y%m%d_%H%M%S");
    sPath = saveStr + timeStr;
    #saveFmt = "png"; #Save fmt/numeration     
    
    # print raw images and filtering steps
    #saveRaw = False;
    #printAnalysis = False;
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
        n = round(math.sqrt(tileArea));
        raw = gwi.generateWPimage(group, wpSize, n);
        cm = plt.get_cmap('gray');
        raw_image =  cm(raw);
        rawFreq = np.fft.fft2(raw, (raw.shape[0], raw.shape[1]));
        # generating wallpapers, saving freq. representations
        for k in range(nGroup):
            
            
            # image processing steps
            
            # get average magnitude
            #avgMag = meanMag(rawFreq);
            
            avgRaw = spectra(raw); # replace each image's magnitude with the average
            filtered = filterImg(avgRaw, wpSize); # low-pass filtering + histeq
            
            masked = cm(maskImg(filtered, wpSize).T); # masking the image (final step)
            #Image.fromarray((masked[:, :, :3] * 255).astype(np.uint8)).show();
            
            # making scrambled images
            scrambled_raw = spectra(raw, pssscrambled=True); # only give spectra only arg, to make randoms
            scrambled_filtered = filterImg(scrambled_raw, wpSize);
            scrambled_masked = cm(maskImg(scrambled_filtered, wpSize));

            #Image.fromarray(np.hstack(((masked[:, :, :3] * 255).astype(np.uint8), (scrambled_masked[:, :, :3] * 255).astype(np.uint8)))).show();
            groupNumber = mapgroup[group];
            # saving averaged and scrambled images
            
            if(printAnalysis):
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(sPath + "analysis\\steps_" + group + "_" + str(k), "JPEG");
                #imwrite(all_in_one{img},  strcat(sPath, 'analysis/steps_',group, '_', num2str(img), '.jpeg'), 'jpeg');
            if(saveRaw):
                rawPath = sRawPath + group + '_' + str(k) + '.' + saveFmt;
                Image.fromarray((raw_image[:, :, :3] * 255).astype(np.uint8)).save(rawPath, saveFmt);
            
            patternPath = sPath + str(1000*groupNumber + k) + '_' + group + '.' + saveFmt;
            
            Image.fromarray((masked[:, :, :3] * 255).astype(np.uint8)).save(patternPath, saveFmt);
            scramblePath = sPath + str(1000*(groupNumber + 17) + k) + '_' + group + '_Scrambled' + '.' + saveFmt;
            Image.fromarray((scrambled_masked[:, :, :3] * 255).astype(np.uint8)).save(scramblePath, saveFmt);
           
            
        #all_in_one = cellfun(@(x,y,z) cat(2,x(1:wpSize,1:wpSize),y(1:wpSize,1:wpSize),z(1:wpSize,1:wpSize)),raw,avgRaw,filtered,'uni',false);
        
        # variables for saving in a mat file
        #symAveraged[:,i]= np.concatenate((avgRaw, scrambled_raw));
        #symFiltered[:,i]= np.concatenate((filtered, scrambled_filtered));
        #symMasked[:,i]= np.concatenate((masked, scrambled_masked));
    #save([sPath,timeStr,'.mat'],'symAveraged','symFiltered','symMasked','Groups');
    
#def saveImg(img,savePath,saveFmt):
#    img = uint8(round(img.*255));
#    imwrite(img, savePath, saveFmt);
#end 

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
    #define mask(circle)
    r = round(0.5 * N);
    mask = np.zeros((inImg.shape[0],inImg.shape[1]), np.uint8);
    cv.circle(mask, (round(inImg.shape[1] / 2), round(inImg.shape[0] / 2)), r, 1, -1);
    mask = cv.bitwise_and(mask, mask, mask=mask);
    outImg = inImg[:np.shape(mask)[0], :np.shape(mask)[1]];
    outImg[mask==0] = 0.5;
    return outImg;

# replace spectra
def spectra(in_image, pssscrambled=False, psscrambled=False, new_mag=None, cmap="gray"):
    in_spectrum = np.fft.fft2(in_image, (in_image.shape[0], in_image.shape[1]));
    
    phase = np.angle(in_spectrum);
    mag = np.abs(in_spectrum);
    
    if (psscrambled == True):
        randPhase =  np.fft.fft2(np.random.rand(in_image.shape[0], in_image.shape[1]), (in_image.shape[0], in_image.shape[1]));
        phase = np.angle(randPhase);
        rng = np.random.default_rng()
        [rng.shuffle(x) for x in phase];
    elif(pssscrambled == True):
        outImage = psScramble(in_image, cmap);
        return outImage;
    if(new_mag):
        mag = new_mag;
    cmplxIm = mag * np.exp(1j * phase);
    outImage = np.abs(np.real(np.fft.ifft2(cmplxIm)));
    return outImage;

def psScramble(in_image, cmap):
    imagetmp = Image.fromarray(in_image);
    newSize = previous_power_2(in_image.shape[0]);
    print(newSize);
    imagetmp = imagetmp.resize((newSize, newSize), Image.BICUBIC);
    #imagetmp.resize((image.shape[0], image.shape[0]), Image.BICUBIC);
    in_image = np.array(imagetmp);
    print(in_image.shape);
    outImage = pss.synthesis(in_image, in_image.shape[0], in_image.shape[1], 5, 4, 7, 25)
    return outImage

def previous_power_2(x):
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);


# returns average mag of the group
"""
def meanMag(freqGroup):
    nImages = 1;
    mag = np.empty((freqGroup.shape[0], freqGroup.shape[1], nImages));
    for n in range(nImages):
        mag[:,:, n] = np.abs(freqGroup);
    out = np.median(mag,2);
    return out;
"""
generateWPTImagesMain();
