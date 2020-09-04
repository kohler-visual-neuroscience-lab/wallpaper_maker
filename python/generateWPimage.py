import numpy as np
import math
from PIL import Image
from skimage import draw as skd

def filterTile(inTile, filterIntensity):
    #outTile = generate_ftile(size(inTile, 1), size(inTile, 2));

    mu = 0.5;
    nx = np.size(inTile, 0);
    ny = np.size(inTile, 1);
    # make adaptive filtering
    
    sigma_x = 10*filterIntensity/nx;
    sigma_y = 10*filterIntensity/ny;
    x = np.linspace(0, 1, nx);
    y = np.linspace(0, 1, ny);
    
    gx = np.exp((-(x - mu)**2) / (2*sigma_x**2)) / (sigma_x*math.sqrt(2*np.pi));
    gy = np.exp((-(y - mu)**2) / (2*sigma_y**2)) / (sigma_y*math.sqrt(2*np.pi));
    
    
    gauss2 = np.matmul(gx.reshape(gx.shape[0], 1),gy.reshape(1, gy.shape[0]));
    gauss2 = gauss2 - gauss2.min();
    gauss2 = gauss2 / gauss2.max();
    gauss2 = gauss2 * 5;
    filtered = np.abs(np.fft.ifft2(np.fft.fft2(inTile) * gauss2));
    
    #normalize tile

    outTile = filtered - filtered.min();
    outTile = outTile / outTile.max();
    return outTile;
    #outTile = histeq(outTile);

def new_p3(tile):
    
    # For magfactor, use a multiple of 3 to avoid the rounding error
    # when stacking two_tirds, one_third tiles together
    
    magfactor = 9;
    tileIm = Image.fromarray(tile);
    # (tuple(i * magfactor for i in reversed(tile.shape)) to calculate the (width, height) of the image
    tile1 = np.array(tileIm.resize((tuple(i * magfactor for i in reversed(tile.shape))), Image.BICUBIC));
    height = np.size(tile1, 0);
    
    # fundamental region is equlateral rhombus with side length = s
    
    s1 = round(height / 3);
    s = 2 * s1;
    
    # NOTE on 'ugly' way of calculating the widt = h
    # after magnification width(tile1) > tan(pi/6)*height(tile1) (should be equal)
    # and after 240 deg rotation width(tile240) < 2*width(tile1) (should be
    # bigger or equal, because we cat them together).
    # subtract one, to avoid screwing by imrotate(240)
    
    width = round(height / math.sqrt(3)) - 1;
    # width = min(round(height/sqrt(3)), size(tile1, 2)) - 1;
   
    # define rhombus-shaped mask
    
    xy = np.array ([[0, 0], [width, s1], [width, height], [0, 2 * s1], [0, 0]]);
    
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
    return p3       

def generateWPimage(wptype,N,n,optTexture):
    #  generateWPimage(type,N,n,optTexture)
    # generates single wallaper group image
    # wptype defines the wallpaper group
    # N is the size of the output image. For complex groups
    #   returned image will have size at least NxN.
    # n the size of repeating pattern for all groups.
    
    #default 
    if len(locals()) < 4: 
        grain = 1;
        texture = filterTile(np.random.rand(n,n), grain);
    else:
        minDim = np.min(np.shape(optTexture));
        #stretch user-defined texture, if it is too small for sampling
        if minDim < n:
            ratio = round(n/minDim);
            #optTexture = imresize(optTexture, ratio, 'nearest');
            optTexture = np.array(Image.resize(reversed((optTexture.shape * ratio)), Image.NEAREST));
        texture = optTexture;
    try:
        if wptype == 'P0':
                p0 = np.array(Image.resize(reversed((texture.shape * round(N/n))), Image.NEAREST));
                image = p0;
                return image;                                
        elif wptype == 'P1':
                width = n;
                height = width;
                p1 = texture[:height, :width];
                image = catTiles(p1, N, wptype);
                return image;                
        elif wptype == 'P2':
                height = round(n/2);
                width = 2*height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                tileR180 = start_tileIm.rotate(180)
                tileR180 = np.array(tileR180);
                p2 = np.concatenate((start_tile, tileR180));
                image = catTiles(p2, N, wptype);       
                return image;
        elif wptype == 'PM':
                height = round(n/2);
                width = 2*height;
                start_tile = texture[:height, :width];
                mirror = np.flipud(start_tile);
                pm = np.concatenate((start_tile, mirror));
                image = catTiles(pm, N, wptype);
                return image;                
        elif wptype == 'PG':
                height = round(0.5*n);
                width = 2*height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                tile = start_tileIm.rotate(270);
                tile = np.array(tile);
                glide = np.flipud(tile);
                pg = np.concatenate((tile, glide), axis=1);
                image = catTiles(pg, N, wptype);
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
                return image;                 
        elif wptype == 'PMG':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                start_tileIm = start_tileIm.rotate(180);
                start_tile_rot180 = np.array(start_tileIm);
                concatTmp1 = np.concatenate((start_tile, start_tile_rot180), axis=1);
                concatTmp2 = np.concatenate((np.flipud(start_tile), np.fliplr(start_tile)), axis=1);
                pmg = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(pmg, N, wptype);
                return image;                 
        elif wptype == 'PGG':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                start_tileIm = start_tileIm.rotate(180);
                start_tile_rot180 = np.array(start_tileIm);
                concatTmp1 = np.concatenate((start_tile, np.flipud(start_tile)), axis=1);
                concatTmp2 = np.concatenate((np.fliplr(start_tile), start_tile_rot180), axis=1);
                pgg = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(pgg, N, wptype);
                return image;                 
        elif wptype == 'CMM':
                height = round(n/4);
                width = 2*height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                start_tileIm = start_tileIm.rotate(180);
                start_tile_rot180 = np.array(start_tileIm);
                tile1 = np.concatenate((start_tile, start_tile_rot180));               
                tile2 = np.flipud(tile1);
                concatTmp1 = np.concatenate((tile1, tile2), axis=1);
                concatTmp2 = np.concatenate((tile2, tile1), axis=1);
                cmm = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(cmm, N, wptype);
                return image;                 
        elif wptype == 'P4':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                start_tileIm = Image.fromarray(start_tile);
                start_tileIm_rot90 = start_tileIm.rotate(90);
                start_tileIm_rot180 = start_tileIm.rotate(180);
                start_tileIm_rot270 = start_tileIm.rotate(270);
                start_tile_rot90 = np.array(start_tileIm_rot90);  
                start_tile_rot180 = np.array(start_tileIm_rot180);  
                start_tile_rot270 = np.array(start_tileIm_rot270);
                concatTmp1 = np.concatenate((start_tile, start_tile_rot270), axis=1);
                concatTmp2 = np.concatenate((start_tile_rot90, start_tile_rot180), axis=1);                
                p4 = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4, N, wptype); 
                return image; 
        elif wptype == 'P4M':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                xy = np.array ([[0, 0], [width, height], [0, height], [0, 0]]);    
                mask = skd.polygon2mask((height, width), xy);
                tile1 = mask * start_tile;
                tile2 = np.fliplr(tile1);
                tile2Im = Image.fromarray(tile2);
                tile2Im = start_tileIm.rotate(90);
                tile2 = np.array(tile2Im);
                tile = np.maximum(tile1, tile2);
                tileIm = Image.fromarray(tile);
                tileIm_rot90 = tileIm.rotate(90);
                tileIm_rot180 = tileIm.rotate(180);
                tileIm_rot270 = tileIm.rotate(270);
                tile_rot90 = np.array(tileIm_rot90);
                tile_rot180 = np.array(tileIm_rot180);
                tile_rot270 = np.array(tileIm_rot270);
                concatTmp1 = np.concatenate((tile, tile_rot270), axis=1);
                concatTmp2 = np.concatenate((tile_rot90, tile_rot180), axis=1); 
                p4m = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4m, N, wptype);   
                return image; 
        elif wptype == 'P4G':
                height = round(n/2);
                width = height;
                start_tile = texture[:height, :width];
                xy = np.array ([[0, 0], [width, 0], [width, height], [0, 0]]);    
                mask = skd.polygon2mask((height, width), xy);
                tile1 = mask * start_tile;
                tile2 = np.fliplr(tile1);
                tile2Im = Image.fromarray(tile2);
                tile2Im = start_tileIm.rotate(90);
                tile2 = np.array(tile2Im);
                tile = np.maximum(tile1, tile2);
                tileIm = Image.fromarray(tile);
                tileIm_rot90 = tileIm.rotate(90);
                tileIm_rot180 = tileIm.rotate(180);
                tileIm_rot270 = tileIm.rotate(270);
                tile_rot90 = np.array(tileIm_rot90);
                tile_rot180 = np.array(tileIm_rot180);
                tile_rot270 = np.array(tileIm_rot270);
                concatTmp1 = np.concatenate((tile_rot270, tile_rot180), axis=1);
                concatTmp2 = np.concatenate((tile, tile_rot90), axis=1); 
                p4g = np.concatenate((concatTmp1, concatTmp2));
                image = catTiles(p4g, N, wptype);   
                return image;
        elif wptype == 'P3':
                alpha = np.pi/3;
                s = n / math.sqrt(3 * np.tan(alpha));
                height = round(s * 1.5);
                start_tile = texture[:height,:];
                p3 = new_p3(start_tile);
                image = catTiles(p3, N, wptype);
                return image;                
        elif wptype == 'P3M1':
                alpha = np.pi/3;
                s = n/math.sqrt(3*np.tan(alpha));
                height = round(s);
                start_tile = texture[:height,:];               
                p3m1 = new_p3m1(start_tile);                
                image = catTiles(p3m1, N, wptype);                
        elif wptype == 'P31M':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s);
                start_tile = texture[:height,:];               
                p31m = new_p31m(start_tile);
                #ugly trick
                p31m_1 = np.fliplr(np.transpose(p31m));
                image = catTiles(p31m_1, N, wptype);                
        elif wptype == 'P6':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s);
                start_tile = texture[:height,:];              
                p6 = new_p6(start_tile);
                image = catTiles(p6, N, wptype);                
        elif wptype == 'P6M':
                s = n/math.sqrt(math.sqrt(3));
                height = round(s/2);
                start_tile = texture[:height,:];
                p6m = new_p6m(start_tile);
                image = catTiles(p6m, N, wptype);                
        else:
                warning('Unexpected Wallpaper Group type. Returning random noise.');
                image = repmat(texture, [np.ceil(N/n),  np.ceil(N/n)]);
    except Exception as err:
        print('new_SymmetricNoise:Error making ' + wptype);
        print(err.args);
    return image;
        
def catTiles(tile, N, wptype):
    #disp tile square
    sq = np.shape(tile[0]) * np.shape(tile[1]);
    print(wptype + ' area of tile = ' + sq);                
    
    #write tile
    tileIm = Image.fromarray(tile);
    tileIm.save('~/Documents/MATLAB/tiles/' + wptype + '_tile.jpeg', 'JPEG');
    dN = 1 + math.floor(N / np.shape(tile));
    img = np.matlib.repmat(tile, dN(1), dN(2));
    return img                 
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