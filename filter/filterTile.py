def filterTile(inTile, filterIntensity):
    #filterTile(inTile, filterIntensity)

    mu = float(0.5);
    nx = float(inTile.shape[0]);
    ny = float(inTile.shape[1]);

    ## make adaptive filtering
    sigma_x = float(10*filterIntensity/nx);
    sigma_y = float(10*filterIntensity/ny);
    x = np.linspace(0, 1, nx);
    y = np.linspace(0, 1, ny);
    gx = np.exp(-pow(x - mu,2))/pow(2*sigma_x,2)/(sigma_x*np.sqrt(2*np.pi));
    gy = np.exp(-pow(y - mu,2))/pow(2*sigma_y,2)/(sigma_y*np.sqrt(2*np.pi));
    
    gauss2 = gx.T*gy;
    gauss2 = gauss2 - np.amin(np.amin(gauss2));
    gauss2 = gauss2/np.amax(np.amax(gauss2));
    gauss2 = gauss2*5;
    filtered = abs(np.fft.ifft2(np.fft.fft2(inTile)*gauss2));

    #normalize tile
    outTile = filtered - np.amin(np.amin(filtered));
    outTile = outTile/np.amax(np.amax(outTile));
    return outTile