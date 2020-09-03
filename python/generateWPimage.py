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
    
    %default 
    if(nargin < 4) 
        grain = 1;
        texture = filterTile(rand(n), grain);
    else
        minDim = min(size(optTexture));
        %stretch user-defined texture, if it is too small for sampling
        if(minDim < n)
            ratio = round(n/minDim);
            optTexture = imresize(optTexture, ratio, 'nearest');
        end;
        texture = optTexture;
    end
    try
        switch type
            case 'P0'
                p0 = imresize(texture, round(N/n), 'nearest');
                image = p0;                                
            case 'P1'
                width = n;
                height = width;
                p1 = texture(1:height, 1:width);
                image = catTiles(p1, N, type);                
            case 'P2'
                height = round(n/2);
                width = 2*height;
                start_tile = texture(1:height, 1:width);
                tileR180 = imrotate(start_tile, 180);
                p2 = [start_tile; tileR180];
                image = catTiles(p2, N, type);                
            case 'PM'
                height = round(n/2);
                width = 2*height;
                start_tile = texture(1:height, 1:width);
                mirror = flipud(start_tile);
                pm = [start_tile; mirror];
                image = catTiles(pm, N, type);                
            case 'PG'
                height = round(0.5*n);
                width = 2*height;
                start_tile = texture(1:height, 1:width);
                tile = rot90(start_tile, 3);
                glide = flipud(tile);
                pg = [tile, glide];
                image = catTiles(pg, N, type);                
            case 'CM'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);                
                mirror = fliplr(start_tile);
                tile1 = [start_tile, mirror];
                tile2 = [mirror, start_tile];
                cm = [tile1; tile2];
                image = catTiles(cm, N, type);                
            case 'PMM'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);                      
                mirror = fliplr(start_tile);
                pmm = [start_tile, mirror; flipud(start_tile), flipud(mirror)];
                image = catTiles(pmm, N, type);                
            case 'PMG'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);
                pmg = [start_tile, rot90(start_tile, 2); flipud(start_tile), fliplr(start_tile)];
                image = catTiles(pmg, N, type);                
            case 'PGG'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);
                pgg = [start_tile, flipud(start_tile); fliplr(start_tile), rot90(start_tile, 2)];
                image = catTiles(pgg, N, type);                
            case 'CMM'
                height = round(n/4);
                width = 2*height;
                start_tile = texture(1:height, 1:width);                
                tile1 = [start_tile; rot90(start_tile,2)];
                tile2 = flipud(tile1);
                cmm = [tile1, tile2; tile2, tile1];
                image = catTiles(cmm, N, type);                
            case 'P4'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);                
                p4 = [start_tile, rot90(start_tile, 3); rot90(start_tile, 1), rot90(start_tile, 2),];
                image = catTiles(p4, N, type);                
            case 'P4M'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);
                x1 = [0 width 0 0];
                y1 = [0 height height 0];
                mask = poly2mask(x1, y1, width, height);
                tile1 = mask.*start_tile;
                tile2 = rot90(fliplr(tile1), 1);
                tile = max(tile1, tile2);
                p4m = [tile, rot90(tile, 3); rot90(tile, 1), rot90(tile, 2)];
                image = catTiles(p4m, N, type);                
            case 'P4G'
                height = round(n/2);
                width = height;
                start_tile = texture(1:height, 1:width);
                y = [0 width width 0];
                x = [0 0 height 0];
                mask = poly2mask(x, y, height, width);
                tile1 = mask.*start_tile;
                tile2 = rot90(fliplr(tile1), 1);
                tile = max(tile1, tile2);
                p4g = [rot90(tile, 3), rot90(tile, 2); tile, rot90(tile, 1)];
                image = catTiles(p4g, N, type);                
            case 'P3'
                alpha = pi/3;
                s = n/sqrt(3*tan(alpha));
                height = round(s*1.5);
                start_tile = texture(1:height, :);
                p3 = new_p3(start_tile);
                image = catTiles(p3, N, type);                
            case 'P3M1'
                alpha = pi/3;
                s = n/sqrt(3*tan(alpha));
                height = round(s);
                start_tile = texture(1:height, :);                
                p3m1 = new_p3m1(start_tile);                
                image = catTiles(p3m1, N, type);                
            case 'P31M'
                s = n/sqrt(sqrt(3));
                height = round(s);
                start_tile = texture(1:height, :);                
                p31m = new_p31m(start_tile);
                %ugly trick
                p31m_1 = fliplr(p31m');
                image = catTiles(p31m_1, N, type);                
            case 'P6'
                s = n/sqrt(sqrt(3));
                height = round(s);
                start_tile = texture(1:height, :);                
                p6 = new_p6(start_tile);
                image = catTiles(p6, N, type);                
            case 'P6M'
                s = n/sqrt(sqrt(3));
                height = round(s/2);
                start_tile = texture(1:height, :);
                p6m = new_p6m(start_tile);
                image = catTiles(p6m, N, type);                
            otherwise
                warning('Unexpected Wallpaper Group type. Returning random noise.');
                image = repmat(texture, [ceil(N/n),  ceil(N/n)]);
    catch err:
        print(strcat('new_SymmetricNoise:Error making ', wptype));
        print(err.message);
        print(err.stack(1));
        print(err.stack(2));
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