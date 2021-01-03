import numpy as np
import matplotlib.pyplot as plt

class Cosine_filter:
    def __init__(self, center_cyc_per_degree, resolution, vis_angle_per_img,pad_width=50):
        self.center_cyc_per_degree= center_cyc_per_degree   
        self.pad_width = pad_width

        if isinstance(resolution,(list, tuple, np.ndarray)):
            self.width = resolution[0]
            self.height = resolution[1]
        else:
            self.width = resolution
            self.height = resolution

        self.width_was_even = False
        self.height_was_even = False
        if np.mod(self.width,2) == 0:
            # padding
            self.width = self.width+1
            self.width_was_even = True
        if np.mod(self.height,2) == 0:
            # padding
            self.height = self.height+1
            self.height_was_even = True

        self.vis_angle_per_img = vis_angle_per_img

        cyc_per_pixel_hor = np.linspace(-0.5, 0.5, self.width)
        cyc_per_pixel_ver = np.linspace(-0.5, 0.5, self.height)

        self.cyc_per_degree_hor = cyc_per_pixel_hor*self.width/self.vis_angle_per_img
        self.cyc_per_degree_ver = cyc_per_pixel_ver*self.height/self.vis_angle_per_img

        xx, yy = np.meshgrid(self.cyc_per_degree_ver,self.cyc_per_degree_hor)
        ff    = (xx**2+yy**2)**0.5 ;
        self.ff = ff    
        self.spectral_filter_mask = 0.5*(1+np.cos(np.pi*(np.log2(np.maximum(ff,np.exp(-20)))-np.log2(center_cyc_per_degree))))
        self.spectral_filter_mask[(ff < 0.5*center_cyc_per_degree) | (2*center_cyc_per_degree < ff)] = 0
        self.padded_spectral_filter_mask = np.pad(self.spectral_filter_mask, self.pad_width)

    def filter_image(self, img):
        img_pad_width = ((self.pad_width, self.pad_width+self.width_was_even), (self.pad_width,\
                                                                                self.pad_width+self.height_was_even))
        padded_img = np.pad(img, img_pad_width, mode='symmetric')
        assert self.padded_spectral_filter_mask.shape[0]==padded_img.shape[0], '{} != {}'.format(self.padded_spectral_filter_mask.shape[0],padded_img.shape[0])
        assert self.padded_spectral_filter_mask.shape[1]==padded_img.shape[1], '{} != {}'.format(self.padded_spectral_filter_mask.shape[1],padded_img.shape[1])
        filtered_padded_img =\
        np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(padded_img))*self.padded_spectral_filter_mask))
        filtered_img = filtered_padded_img[self.pad_width:self.pad_width+img.shape[0],self.pad_width:self.pad_width+img.shape[1]]
        assert np.abs(np.real(filtered_img)/np.imag(filtered_img)).min()>10^10, 'something is fishy: imaginery part is unreasonable large'
        return np.real(filtered_img)

    def display(self):
        fig, axs = plt.subplots(1,2,figsize=(30,5))
        axs[0].imshow(self.spectral_filter_mask,extent=[self.cyc_per_degree_hor.min(),self.cyc_per_degree_hor.max(),self.cyc_per_degree_ver.min(),self.cyc_per_degree_ver.max()])
        axs[1].plot(self.cyc_per_degree_hor,np.abs(self.spectral_filter_mask[:,int(self.width/2)]))
        


