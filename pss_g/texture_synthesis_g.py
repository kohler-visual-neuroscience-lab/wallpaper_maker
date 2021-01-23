'''
	Texture Synthesis (Gray Scale Version)
	this is a port of textureSynth/textureSynthesis.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/
	Differences:
	(1) i use real version of steerable pyramid.
	    Sorry. I could not understand the algorithm of complex version in textureSynthesis.m.
	(2) i don't use filter masks of orientations in the process of coarse to fine.
	Usage:
	python texture_synthesis_g.py -i radish-mono.jpg -o tmp -n 5 -k 4 -m 7 --iter 100
	-i : input image
	-o : path for output
	-n : depth of steerable pyramid (default:5)
	-k : num of orientations of steerable pyramid (default:4)
	-n : pixel distance for calicurationg auto-correlations (default:7)
	--iter : number of iterations (default:100)
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.stats import skew, kurtosis
from PIL import Image
import sys, os
import logging
import argparse, copy
import time
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

ALPHA = 0.8
PS = 1e-6

'''
	Texture Synthesis by Portilla-Simoncelli's algorithm
'''
def synthesis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor, iter):
	LOGGER.disabled = True
	# analyse original image
	orig_data = TextureAnalysis(image, resol_x, resol_y, num_depth, num_ori, num_neighbor)
	orig_data.analyse()

	# initialize random image
	im = np.random.normal(0, 1, resol_x * resol_y).reshape(resol_y, resol_x)
	im = im * np.sqrt(orig_data.IM_VAR)
	im = im + orig_data.IM_MAR[0]

	# iteration
	prev_im = np.array([])
	prev_dst = 0.

	for it in range(0, iter):
		LOGGER.debug('iteration {}'.format(str(it)))
	
		pyr_l = []
		lr_l = []

		# ------------------------------------
		# Create pyramids of each PCA channel
		# steerable pyramid
		_sp = SteerablePyramid(im, resol_x, resol_y, num_depth, num_ori, '', '', 0)
		_sp.create_pyramids()

		# subtract means from lowpass residuals
		_sp.LR['s'] = _sp.LR['s'].real - np.mean(_sp.LR['s'].real.flatten())

		pyr_l = copy.deepcopy(_sp)
		lr_l = _sp.LR

		# ------------------------------------
		# Adjust lowpass residual and get initial image for coarse to fine
		# modify central auto correlation
		try:
			lr_l['s'] = mod_acorr(lr_l['s'], orig_data.LR_CA, num_neighbor)
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))
			lr_l['s'] = lr_l['s'] * np.sqrt(orig_data.LR_MAR[1] / np.var(lr_l['s']))

		lr_l['s'] = lr_l['s'].real
		# modify skewness of lowpass residual
		try:
			lr_l['s'] = mod_skew(lr_l['s'], orig_data.LR_MAR[2])
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))
		# modify kurtosis of lowpass residual
		try:
			lr_l['s'] = mod_kurt(lr_l['s'], orig_data.LR_MAR[3])
		except LinAlgError as e:
			LOGGER.info('LinAlgError {}'.format(e))

		lr_l['f'] = np.fft.fftshift(np.fft.fft2(lr_l['s']))

		 # initial coarse to fine
		rec_im = lr_l['s']

		## get original statistics of bandpass signals.
		# create parents
		bnd = copy.deepcopy(pyr_l.BND)
		_b_m, _, _ = trans_b(pyr_l.BND)
		for i in range(len(_b_m)):
			for k in range(len(_b_m[i])):
				_b_m[i][k] -= np.mean(_b_m[i][k])
		## magnitude
		bnd_m = _b_m

		_b_p, _b_rp, _b_ip = get_parent_g(pyr_l.BND, pyr_l.LR)
		## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
		bnd_p = _b_p
		## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		bnd_rp = _b_rp
		## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		bnd_ip = _b_ip

		# ------------------------------------
		# Coarse to fine adjustment
		for dp in range(num_depth-1, -1, -1):

			# combine orientations
			cousins = cori_b(bnd_m, dp)

			# adjust covariances
			_prev = cousins
			if dp < num_depth-1:
				parents = bnd_p[dp]
				cousins = adjust_corr2(_prev, orig_data.CF_COUS[dp], parents, orig_data.CF_CPAR[dp])
				if np.isnan(cousins).any():
					LOGGER.info('NaN in adjust_corr2')
					cousins = adjust_corr1(_prev, orig_data.CF_COUS[dp])
				rparents = cori_rp(bnd_rp, bnd_ip, dp)
			else:
				cousins = adjust_corr1(_prev, orig_data.CF_COUS[dp])

			# separate orientations
			cousins = sori_b(cousins, num_ori)

			# adjust central auto corr. and update bandpass.
			bnd_r = []
			for k in range(num_ori):
				# adjust central auto-correlations
				try:
					_tmp = mod_acorr(cousins[k], orig_data.BND_MCOR[dp][k], num_neighbor)
				except LinAlgError as e:
					LOGGER.info('LinAlgError {}'.format(e))
					_tmp = cousins[k]

				# update BND_N
				bnd_m[dp][k] = _tmp
				_mean = orig_data.BND_MMAR[dp][k][0]
				_tmp = _tmp + _mean
				_idx = np.where(_tmp < 0)
				_tmp[_idx] = 0

				_bnd = pyr_l.BND[dp][k]['s']
				_idx1 = np.where(np.abs(_bnd) < 10**(-12))
				_idx2 = np.where(np.abs(_bnd) >= 10**(-12))
				_bnd[_idx1] = _bnd[_idx1] * _tmp[_idx1]
				_bnd[_idx2] = _bnd[_idx2] * _tmp[_idx2] / np.abs(_bnd[_idx2])

				bnd_r.append(_bnd.real)

			# combine orientations & make rcousins
			rcousins = cori_bc(bnd_r, dp)

			# adjust cross-correlation of real values of B and real/imaginary values of parents
			_prev = rcousins
			try:
				if dp < num_depth-1:
					rcousins = adjust_corr2(_prev, orig_data.CF_RCOU[dp], rparents, orig_data.CF_RPAR[dp])
					if np.isnan(rcousins).any():
						LOGGER.info('NaN in adjust_corr2')
						rcousins = adjust_corr1(_prev, orig_data.CF_RCOU[dp])
						if np.isnan(rcousins).any():
							LOGGER.info('NaN in adjust_corr1')
							rcousins = _prev
				else:
					rcousins = adjust_corr1(_prev, orig_data.CF_RCOU[dp])
					if np.isnan(rcousins).any():
						LOGGER.info('NaN in adjust_corr1')
						rcousins = _prev
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))
				rcousins = adjust_corr1(_prev, orig_data.CF_RCOU[dp])
				if np.isnan(rcousins).any():
					LOGGER.info('NaN in adjust_corr1')
					rcousins = _prev

			# separate orientations
			rcousins = sori_b(rcousins, num_ori)
			for k in range(num_ori):
				## update pyramid
				pyr_l.BND[dp][k]['s'] = rcousins[k]
				pyr_l.BND[dp][k]['f'] = np.fft.fftshift(np.fft.fft2(rcousins[k]))

			# combine bands
			_rc = copy.deepcopy(rcousins)
			# same size
			_z = np.zeros_like(_rc[0])
			_s = SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)

			_recon = np.zeros_like(_z)
			for k in range(num_ori):
				_recon = _recon + pyr_l.BND[dp][k]['f'] * _s.B_FILT[0][k]
			_recon = _recon * _s.L0_FILT
			_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

			# expand image created before and sum up
			_im = rec_im
			_im = expand(_im, 2).real / 4.
			_im = _im.real + _recon

			# adjust auto-correlation
			try:
				_im = mod_acorr(_im, orig_data.CF_CA[dp], num_neighbor)
			except LinAlgError as e:
				LOGGER.info('Pass. LinAlgError {}'.format(e))

			# modify skewness
			try:
				_im = mod_skew(_im, orig_data.CF_MAR[dp][2])
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))

			# modify kurtosis
			try:
				_im = mod_kurt(_im, orig_data.CF_MAR[dp][3])
			except LinAlgError as e:
				LOGGER.info('LinAlgError {}'.format(e))

			rec_im = _im

		# end of coarse to fine

		# ------------------------------------
		# Adjustment variance in H0 and final adjustment of coarse to fine.
		_tmp = pyr_l.H0['s'].real
		_var = np.var(_tmp)
		_tmp = _tmp * np.sqrt(orig_data.H0_PRO / _var)

		# recon H0
		_recon = np.fft.fftshift(np.fft.fft2(_tmp))
		_recon = _recon * _s.H0_FILT
		_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

		## this is final data of coarse to fine.
		rec_im = rec_im + _recon

		# adjust skewness and kurtosis to original.
		_mean = np.mean(rec_im)
		_var = np.var(rec_im)
		rec_im = ( rec_im - _mean ) * np.sqrt( orig_data.IM_MAR[1] / _var)
		rec_im = rec_im + orig_data.IM_MAR[0]

		## skewness
		rec_im = mod_skew(rec_im, orig_data.IM_MAR[2])
		## kurtosis
		rec_im = mod_kurt(rec_im, orig_data.IM_MAR[3])

		_idx  = np.where(rec_im > orig_data.IM_MAR[4])
		rec_im[_idx] = orig_data.IM_MAR[4]
		_idx  = np.where(rec_im < orig_data.IM_MAR[5])
		rec_im[_idx] = orig_data.IM_MAR[5]

		im = rec_im

		# ------------------------------------
		# Save image
		#_o_img = Image.fromarray(np.uint8(im)).convert('L')
		#_o_img.save(out_path + '/out-n{}-k{}-m{}-{}.png'.format(str(num_depth), str(num_ori), str(num_neighbor), str(it)))

		if it > 0:
			dst = np.sqrt(np.sum((prev_im - im)**2))
			rt = np.sqrt(np.sum((prev_im - im)**2)) / np.sqrt(np.sum(prev_im**2))
			LOGGER.debug('change {}, ratio {}'.format(str(dst), str(rt)))

			if it > 1:
				thr = np.abs(np.abs(prev_dst) - np.abs(dst)) / np.abs(prev_dst)
				LOGGER.debug('threshold {}'.format(str(thr)))
				if thr < 1e-6:
					break

			prev_dst = dst

			# acceleration
			im = im + ALPHA * (im - prev_im)

		prev_im = im
	return im;

"""
if __name__ == "__main__":
	LOGGER.info('script start')
	
	start_time = time.time()

	parser = argparse.ArgumentParser(
	    description='Texture Synthesis (Gray Version) by Portilla and Simoncelli')
	parser.add_argument('--orig_img', '-i', default='pebbles.jpg',
                    help='Original image')
	parser.add_argument('--out_dir', '-o', default='tmp',
                    help='Output directory')
	parser.add_argument('--num_depth', '-n', default=5, type=int,
                    help='depth of steerable pyramid')
	parser.add_argument('--num_ori', '-k', default=4, type=int,
                    help='orientation of steerable pyramid')
	parser.add_argument('--num_neighbor', '-m', default=7, type=int,
                    help='local neighborhood')
	parser.add_argument('--iter', default=100, type=int,
                    help='number of iterations')

	args = parser.parse_args()

	## validation of num. of neighbours.
	ms = [3, 5, 7, 9, 11, 13]
	if not args.num_neighbor in ms:
			LOGGER.error('illegal number of orientation: {}'.format(str(args.num_neighbor)))
			raise ValueError('illegal number of orientation: {}'.format(str(args.num_neighbor)))

	im = np.array(Image.open(args.orig_img))
	synthesis(im, im.shape[1], im.shape[0], args.num_depth, args.num_ori, args.num_neighbor, args.iter, args.out_dir)
"""

'''
	Upsampling
	this is a port of textureSynth/expand.m by J. Portilla and E. Simoncelli.
	http://www.cns.nyu.edu/~lcv/texture/
	https://github.com/LabForComputationalVision/textureSynth
'''
def expand(t ,f, p=0):
	my,mx = t.shape
	T = np.zeros((my, mx), dtype=complex)
	my = f*my
	mx = f*mx
	Te = np.zeros((my, mx), dtype=complex)

	T = f**2 * np.fft.fftshift(np.fft.fft2(t))
	y1 = my/2 + 2 - my/(2*f)
	y2 = my/2 + my/(2*f)
	x1 = mx/2 + 2 - mx/(2*f)
	x2 = mx/2 + mx/(2*f)
	y1 = int(y1)
	y2 = int(y2)
	x1 = int(x1)
	x2 = int(x2)
    
	Te[y1-1:y2, x1-1:x2] = T[1:int(my/f), 1:int(mx/f)]
	Te[y1-2, x1-1:x2] = T[0, 1:int(mx/f)]/2
	Te[y2, x1-1:x2] = T[0, int(mx/f):0:-1]/2
	Te[y1-1:y2, x1-2] = T[1: int(my/f), 0]/2
	Te[y1-1:y2, x2] = T[int(my/f):0:-1, 0]/2

	esq = T[0,0] / 4
	Te[y1-2, x1-2] = esq
	Te[y1-2, x2] = esq
	Te[y2, x1-2] = esq
	Te[y2, x2] = esq

	Te = np.fft.fftshift(Te)
	te = np.fft.ifft2(Te)
	te = te.real

	return te


'''
	Downsampling
	this is a port of textureSynth/shrink.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	http://www.cns.nyu.edu/~lcv/texture/
'''
def shrink(t, f):
	my,mx = t.shape
	T=np.fft.fftshift(np.fft.fft2(t))/f**2
	Ts=np.zeros((int(my/f), int(mx/f)), dtype=complex)
	y1=int(my/2 + 2 - my/(2*f))
	y2=int(my/2 + my/(2*f))
	x1=int(mx/2 + 2 - mx/(2*f))
	x2=int(mx/2 + mx/(2*f))

	Ts[1:int(my/f), 1:int(mx/f)] = T[y1-1:y2 ,x1-1:x2]
	Ts[0,1:int(mx/f)]=(T[y1-2, x1-1:x2]+T[y2, x1-1:x2])/2
	Ts[1:int(my/f),0] = (T[y1-1:y2, x1-2] + T[y1-1:y2, x2])/2
	Ts[0,0] = (T[y1-2,x1-1] + T[y1-2,x2] + T[y2, x1-1] + T[y2, x2+1])/4
	Ts=np.fft.fftshift(Ts)
	ts=np.fft.ifft2(Ts)
	ts = ts.real
#	ts = np.abs(ts)

	return ts

'''
	Doubling phases
	this is a port of textureSynth/modskew.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
'''
def double_phase(image):
	
	_rtmp = image.real
	_itmp = image.imag

	_theta = np.arctan2(_itmp, _rtmp)
	_rad = np.sqrt(_rtmp**2 + _itmp**2)

	_tmp = _rad * np.exp(2 * complex(0,1) * _theta)

	return _tmp

#def double_phase_ng(image):
#	ft = np.fft.fft2(image)
#	_ft = np.fft.fftshift(ft)	
#	
#	tmp_theta = np.angle(_ft)
#	tmp_pol = np.absolute(_ft)
#	_tmp = tmp_pol * np.exp(2 * complex(0,1) * tmp_theta)
#
#	_tmp = np.fft.ifft2(np.fft.ifftshift(_tmp))
#
#	return _tmp


'''
	modify skewness
	this is a port of textureSynth/modskew.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	http://www.cns.nyu.edu/~lcv/texture/
'''
def mod_skew(im, sk):
	# mu
	_mean = np.mean(im.flatten())

	im = im - _mean

	_tmp = im**2
	_sd = np.sqrt(np.mean(_tmp.flatten()))

	mu = [im**i for i in range(3,7)]
	mu = [np.mean(mu[i].flatten()) for i in range(len(mu))]

#	print(im)
#	print(mu[0])
#	print(_sd)
	if _sd > 0:
		_sk = mu[0] / (_sd)**3

		_A = mu[3] - 3.*_sd*_sk*mu[2] + 3.*_sd**2.*(_sk**2.-1.)*mu[1] + _sd**6*(2 + 3*_sk**2 - _sk**4)
		_B = 3.*( mu[2] - 2.*_sd*_sk*mu[1] + _sd**5*_sk**3 )
		_C = 3.*( mu[1] - _sd**4*( 1. + _sk**2) )
		_D = _sk * _sd**3

		a = np.zeros_like(range(0,7), dtype='double')
		a[6] = _A**2.
		a[5] = 2.*_A*_B
		a[4] = _B**2 + 2.*_A*_C
		a[3] = 2.*(_A*_D + _B*_C)
		a[2] = _C**2 + 2.*_B*_D
		a[1] = 2.*_C*_D
		a[0] = _D**2

		_a2 = _sd**2
		_b2 = mu[1] - (1. + _sk**2)*_sd**4

		b = np.zeros_like(range(0,7), dtype='double')
		b[6] = _b2**3
		b[4] = 3.*_a2*_b2**2
		b[2] = 3.*_a2**2*_b2
		b[0] = _a2**3

		d = np.zeros_like(range(0,8), dtype='double')
		d[7] = _B * b[6]
		d[6] = 2*_C*b[6] - _A*b[4]
		d[5] = 3*_D*b[6]
		d[4] = _C*b[4] - 2.*_A*b[2]
		d[3] = 2*_D*b[4] - _B*b[2]
		d[2] = -3.*_A*b[0]
		d[1] = _D*b[2] - 2*_B*b[0]
		d[0] = -_C*b[0]

		d = d[::-1]
		mMlambda = np.roots(d)

		tg = mMlambda.imag / mMlambda.real
		_idx = np.where(np.abs(tg) < 1e-6)
		mMlambda = mMlambda[_idx].real
		lNeg = mMlambda[np.where(mMlambda < 0)]
		if lNeg.shape[0] == 0:
			lNeg = -1/2**-50

		lPos = mMlambda[np.where(mMlambda >= 0)]
		if lPos.shape[0] == 0:
			lPos = 1/2**-50

		lmi = np.max(lNeg)
		lma = np.min(lPos)

		lam = np.array([lmi, lma], dtype='double')

		mMnewSt = np.polyval(np.array([_A, _B, _C, _D], dtype='double'), lam) / np.sqrt(np.polyval(b[::-1], lam))

		skmin = np.min(mMnewSt)
		skmax = np.max(mMnewSt)

# Given a desired skewness, solves for lambda
		if sk <= skmin:
			lam = lmi
			LOGGER.debug('Saturating (down) skewness!')
		elif sk >= skmax:
			lam = lma
			LOGGER.debug('Saturating (up) skewness!')
		else:
			c = a - b*sk**2
			c = c[::-1]

			r = np.roots(c)

# Chose the real solution with minimum absolute value with the rigth sign
			lam = np.array( [0.] )
			co = 0
			tg = np.abs(r.imag / r.real)
			_idx = np.where(( np.abs(tg) < 1e-6 ) & ( np.sign(r.real) == np.sign(sk - _sk)))
			if r[_idx].shape[0] > 0:
				lam = r[_idx].real

			if np.all(lam == 0.):
				LOGGER.info('Warning: Skew adjustment skipped!')

			p = [_A, _B, _C, _D]

			if lam.shape[0] > 1:
				foo = np.sign(np.polyval(p, lam))
				if np.any(foo == 0):
					lam = lam[np.where(foo == 0)]
				else:
					lam = lam[np.where(foo == np.sign(sk))]		# rejects the symmetric solution

				if lam.shape[0] > 0:
					lam = lam[np.where(np.abs(lam) == np.min(abs(lam)))]	# the smallest that fix the skew
					lam = lam[0]
				else:
					lam = 0.

# Modify the channel
		chm = im + lam*(im**2 - _sd**2 - _sd*_sk*im)		# adjust the skewness
		chm = chm * _sd / np.sqrt(np.var((chm).flatten()))		# adjust the variance
		chm = chm + _mean				# adjust the mean
	
			# test
#			np.savetxt('chm.csv', im, delimiter=',')
#			_dst = np.sqrt(np.sum((im - chm)**2))
#			LOGGER.debug('change {}'.format(str(_dst)))
	else:
		chm = im

	return chm
	


'''
	modify kurtosis
	this is a port of textureSynth/modkurt.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	http://www.cns.nyu.edu/~lcv/texture/
'''
def mod_kurt(im, kt):
	# mu
	_mean = np.mean(im.flatten())
#	_sd = np.sqrt(np.var(im.flatten()))

	im = im - _mean

	_tmp = im**2
	_sd = np.sqrt(np.mean(_tmp.flatten()))

	mu = [im**i for i in range(3,13)]
	mu = [np.mean(mu[i].flatten()) for i in range(len(mu))]
	
	if _sd > 0:
		_kt = mu[1] / (_sd)**4

		_a = mu[1] / _sd**2

		_A = mu[9] - 4.*_a*mu[7] - 4*mu[0]*mu[6] + 6.*_a**2*mu[5] + 12*_a*mu[0]*mu[4] + 6*mu[0]**2*mu[3] \
			- 4*_a**3*mu[3] - 12*_a**2*mu[0]*mu[2] + _a**4*mu[1] - 12*_a*mu[0]**2*mu[1] \
			+ 4*_a**3*mu[0]**2 + 6*_a**2*mu[0]**2*_sd**2 - 3.*mu[0]**4
		_B = 4.* ( mu[7] - 3*_a*mu[5] - 3*mu[0]*mu[4] + 3.*_a**2*mu[3] + 6.*_a*mu[0]*mu[2] + 3.*mu[0]**2*mu[1] \
			- _a**3*mu[1] - 3.*_a**2*mu[0]**2 - 3*mu[1]*mu[0]**2 )
		_C = 6.* ( mu[5] - 2.*_a*mu[3] - 2.*mu[0]*mu[2] + _a**2*mu[1] + 2.*_a*mu[0]**2 + mu[0]**2*_sd**2 )
		_D = 4.* ( mu[3] - _a**2*_sd**2 - mu[0]**2 )
		_E = mu[1]

		# Define the coefficients of the denominator (F*lam^2+G)^2
		_F = _D / 4.
		_G = _sd**2

		d = np.zeros_like(range(0,5), dtype='double')
		d[0] = _B * _F
		d[1] = 2.*_C*_F - 4.*_A*_G
		d[2] = 4.*_F*_D - 3.*_B*_G - _D*_F
		d[3] = 4.*_F*_E - 2.*_C*_G
		d[4] = -1. * _D * _G

		mMlambda = np.roots(d)

		tg = mMlambda.imag / mMlambda.real
		_idx = np.where(np.abs(tg) < 1e-6)
		mMlambda = mMlambda[_idx].real
		lNeg = mMlambda[np.where(mMlambda < 0)]
		if lNeg.shape[0] == 0:
			lNeg = -1/2**-50

		lPos = mMlambda[np.where(mMlambda >= 0)]
		if lPos.shape[0] == 0:
			lPos = 1/2**-50

		lmi = np.max(lNeg)
		lma = np.min(lPos)

		lam = np.array([lmi, lma], dtype='double')

		mMnewKt = np.polyval(np.array([_A, _B, _C, _D, _E], dtype='double'), lam) / np.polyval(np.array([_F, 0, _G], dtype='double'), lam)**2

		kmin = np.min(mMnewKt)
		kmax = np.max(mMnewKt)

	# Given a desired skewness, solves for lambda
		if kt <= kmin:
			lamb = lmi
			LOGGER.debug('Saturating (down) skewness!')
		elif kt >= kmax:
			lamb = lma
			LOGGER.debug('Saturating (up) skewness!')
		else:
			c = np.zeros_like(range(0,5), dtype='double')
			_tmp = kt*(_G**2)
			c[0] = _E - _tmp
			c[1] = _D
			c[2] = _C - 2.*kt*_F*_G
			c[3] = _B
			c[4] = _A - kt*_F**2

			c = c[::-1]

			r = np.roots(c)

# Chose the real solution with minimum absolute value with the rigth sign
			lam = np.array( [0.] )
			co = 0
			tg = r.imag / r.real
			_idx = np.where( np.abs(tg) == 0. )
			lam = r[_idx].real

			if lam.shape[0] > 0:
				lamb = lam[np.where(np.abs(lam) == np.min(np.abs(lam)))].real
				lamb = lamb[0]
			else:
				lamb = 0.

		# Modify the channel
		chm = im + lamb*(im**3 - _a*im - mu[0])		# adjust the skewness
		chm = chm * _sd / np.sqrt(np.var((chm).flatten()))		# adjust the variance
		chm = chm + _mean				# adjust the mean

#		_dst = np.sqrt(np.sum((im - chm)**2))
#		LOGGER.debug('change {}'.format(str(_dst)))

	else:
		chm = im

	return chm


'''
	modify auto correlation
	this is a port of textureSynth/modacor22.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	
	http://www.cns.nyu.edu/~lcv/texture/
'''
def mod_acorr(im, cy, mm):
	_la = np.floor((mm-1)/2)
	_nc = cy.shape[1]

	centy = int(im.shape[0]/2+1)
	centx = int(im.shape[1]/2+1)

	# calicurate auto correlation of original image.
	ft = np.fft.fft2(im)
	ft2 = np.abs(ft)**2
	cx = np.fft.ifftshift(np.fft.ifft2(ft2).real)
	if not np.all(np.isreal(cx)):
		cx = cx / 2.
	
	cy = cy*np.prod(im.shape)	# Unnormalize the previously normalized correlation

	# Take just the part that has influence on the samples of cy (cy=conv(cx,im))
	ny = int(cx.shape[0]/2.0)+1
	nx = int(cx.shape[1]/2.0)+1
	_sch = min((ny, nx))
	le = int(min((_sch/2-1, _la)))

	cx = cx[ny-2*le-1: ny+2*le, nx-2*le-1: nx+2*le]

	# Build the matrix that performs the convolution Cy1=Tcx*Ch1
	_ncx = 4*le + 1
	_win = int(((_nc)**2 + 1)/2)
	_tcx = np.zeros((_win, _win))

	for i in range(le+1, 2*le+1):
		for j in range(le+1, 3*le+2):
			ccx = cx[i-le-1:i+le, j-le-1:j+le].copy()
			ccxi = ccx[::-1, ::-1]
			ccx += ccxi
			ccx[le, le] = ccx[le, le]/2.
			ccx = ccx.flatten()
			nm = (i-le-1)*(2*le+1) + (j-le) 
			_tcx[nm-1,] = ccx[0:_win]

	i = 2*le + 1
	for j in range(le+1, 2*le+2):
		ccx = cx[i-le-1:i+le, j-le-1:j+le].copy()
		ccxi = ccx[::-1, ::-1]
		ccx = ccx + ccxi
		ccx[le, le] = ccx[le, le]/2.
		ccx = ccx.flatten()
		nm = (i-le-1)*(2*le+1) + (j-le)
		_tcx[nm-1,] = ccx[0:_win]

	# Rearrange Cy indices and solve the equation
	cy1 = cy.flatten()
	cy1 = cy1[0:_win]

	# np.solve might be better than np.inv
	ch1 = np.linalg.solve(_tcx, cy1)

	# Rearrange Ch1

	ch1 = np.hstack((ch1, ch1[-2::-1]))
	ch = ch1.reshape((_nc, _nc))

	aux = np.zeros(im.shape)
	aux[centy-le-1:centy+le, centx-le-1:centx+le] = ch
	ch = np.fft.fftshift(aux)
	chf = np.fft.fft2(ch).real

	yf = ft*np.sqrt(np.abs(chf))
	y = np.fft.ifft2(yf).real

	return y


'''
	adjust correlation
	this is a port of textureSynth/adjustCorr1s.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	
	http://www.cns.nyu.edu/~lcv/texture/
'''
def adjust_corr1(xx, c0):

	# get variance
	_C = np.dot(xx.T, xx) / xx.shape[0]
	_D, _E = np.linalg.eig(_C)

	_D[np.where(np.abs(_D) < PS)] = 0
	if np.sum(np.where(_D < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D)
		LOGGER.info(_C)
	_idx = np.argsort(_D)[::-1]
	_D = np.diag(np.sqrt(_D[_idx]))
	_iD = np.zeros_like(_D)
	_iD[np.where(_D != 0.)] = 1. / _D[np.where(_D != 0.)]
	_E = _E[:, _idx]

	_D0, _E0 = np.linalg.eig(c0)

	_D0[np.where(np.abs(_D0) < PS)] = 0
	if np.sum(np.where(_D0 < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D0)
		LOGGER.info(c0)
		LOGGER.info(c0-c0.T)

	_idx = np.argsort(_D0)[::-1]
	_D0 = np.diag(np.sqrt(_D0[_idx]))
	_E0 = _E0[:, _idx]

	_orth = np.dot(_E.T, _E0)

	# _E * inv(D) * _orth * _D0 * _E0'
	_M = np.dot(_E, np.dot(_iD, np.dot(_orth, np.dot(_D0, _E0.T))))

	_new = np.dot(xx, _M)

	return _new


'''
	adjust correlation
	this is a port of textureSynth/adjustCorr2s.m by J. Portilla and E. Simoncelli.
	https://github.com/LabForComputationalVision/textureSynth
	http://www.cns.nyu.edu/~lcv/texture/
'''
def adjust_corr2(xx, cx, yy, cxy):
	# subtract mean
	_mean = np.mean(xx, axis=0)
	xx = xx - _mean
	_mean = np.mean(yy, axis=0)
	yy = yy - _mean
	# get variance , covariance
	_Bx = np.dot(xx.T, xx) / xx.shape[0]
	_Bxy = np.dot(xx.T, yy) / xx.shape[0]
	_By = np.dot(yy.T, yy) / yy.shape[0]
	_iBy = np.linalg.inv(_By)

	_Cur = _Bx - np.dot(_Bxy, np.dot(_iBy, _Bxy.T))
	_Des = cx - np.dot(cxy, np.dot(_iBy, cxy.T))

	_D, _E = np.linalg.eig(_Cur)
	_D[np.where(np.abs(_D) < PS)] = 0
	if np.sum(np.where(_D < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D)
	_idx = np.argsort(_D)[::-1]
	_D = np.diag(np.sqrt(_D[_idx]))
	_iD = np.zeros_like(_D)
	_iD[np.where(_D != 0.)] = 1. / _D[np.where(_D != 0.)]
	_E = _E[:, _idx]

	_D0, _E0 = np.linalg.eig(_Des)
	_D0[np.where(np.abs(_D0) < PS)] = 0
	if np.sum(np.where(_D0 < 0)):
		LOGGER.info('negative eigenvalue')
		LOGGER.info(_D0)
	_idx = np.argsort(_D0)[::-1]
	_D0 = np.diag(np.sqrt(_D0[_idx]))
	_E0 = _E0[:, _idx]

	_orth = np.dot(_E.T, _E0)

	# _E * inv(D) * _orth * _D0 * _E0'
	_Mx = np.dot(_E, np.dot(_iD, np.dot(_orth, np.dot(_D0, _E0.T))))

	_My = np.dot(_iBy, (cxy.T - np.dot(_Bxy.T, _Mx)))

	_new = np.dot(xx, _Mx) + np.dot(yy, _My)
	
	return _new


'''
	calicurate auto correlation
'''
def get_acorr(im, mm):
	_fr = np.fft.fft2(im)
#	_fr = np.fft.fftshift(np.fft.fft2(im))
	_la = np.floor((mm-1)/2)
	
	_t = np.absolute(_fr)
	_tmp = _t ** 2 / np.prod(_t.shape)
#	_tmp = ( _t - np.mean(_t.flatten()) )**2 / np.prod(_t.shape)
	# important!! auto-correlation
	_tmp = np.fft.ifft2(_tmp)
	_tmp = _tmp.real
	_tmp = np.fft.ifftshift(_tmp)
#	_tmp = np.absolute(_tmp)

	ny = int(_t.shape[0]/2.0)
	nx = int(_t.shape[1]/2.0)
	
	_sch = min((ny, nx))
	le = int(min((_sch/2-1, _la)))
	ac = _tmp[ny-le: ny+le+1, nx-le: nx+le+1]

	return ac


'''
	covariance matrix of color image(3 channels)
'''
def cov_im(im):
	_tmp = np.array(im)

	_list = np.zeros((_tmp.shape[0]*_tmp.shape[1], _tmp.shape[2]))

	_dp = []
	for i in range(_tmp.shape[2]):
		_list[:, i] = _tmp[:, :, i].flatten()

	_mean = np.mean(_list, axis=0)
	_list -= _mean

	_t = np.dot(_list.T, _list) / _list.shape[0]

	return _t


'''
	means of color image(3 channels)
'''
def mean_im(im):
	_tmp = np.array(im)

	_list = np.zeros((_tmp.shape[0]*_tmp.shape[1], _tmp.shape[2]))

	_dp = []
	for i in range(_tmp.shape[2]):
		_list[:, i] = _tmp[:, :, i].flatten()

	_mean = np.mean(_list, axis=0)

	return _mean


'''
	normalized PCA
'''
def get_pca_test(image):
	# reshape to ['width of _img' * 'height', 'channel'] matrix.
	_img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

	pca = PCA()
	pca.fit(_img)

	_pcdata = pca.transform(_img)

	# normalize _pcdate
	_sd = np.sqrt(np.var(_pcdata, axis=0))
	_pcdata = _pcdata / _sd

	_pcdata = _pcdata.reshape(image.shape[0], image.shape[1], image.shape[2])

	return _pcdata

'''
	normalized PCA
'''
def get_pca(image):

	# reshape to ['width of _img' * 'height', 'channel'] matrix.
	_img = image.reshape(image.shape[0]*image.shape[1], image.shape[2])

	_mean = np.mean(_img, axis=0)
	_tmp = _img - _mean
	_covar = np.dot(_tmp.T, _tmp)/_img.shape[0]

	_eval, _evec = np.linalg.eig(_covar)
	_idx = np.argsort(_eval)[::-1]
	_ediag = np.diag(_eval[_idx])
	_evec = _evec[:, _idx]
	## this treatment is to get same results as Matlab
	for k in range(_evec.shape[1]):
		if np.sum(_evec[:,k] < 0) > np.sum(_evec[:,k] >= 0):
			_evec[:,k] = -1. * _evec[:,k]

	# get principal components
	_pcscore = np.dot(_tmp, _evec)

	# Moore-Penrose Pseudo Inverse.
	## Generalized inverse matrix is not necessary for this case. (trivial)
	## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
	## **(0.5) would be right. this obstructs color reproduction.
	#_iediag = np.linalg.pinv(_ediag**(0.25))
	_iediag = np.linalg.pinv(_ediag**(0.5))
		
	# normalize principal components
	_npcdata = np.dot(_pcscore, _iediag)
	_npcdata = _npcdata.reshape(image.shape[0], image.shape[1], image.shape[2])

	return _npcdata


'''
	(1) marginal statistics
		mean, variance, skewness, kurtosis, range of original image
		variance of highpass residual		
'''
def mrg_stats(image):
		
	_mean = np.mean(image.real.flatten())
	_var = np.var(image.real.flatten())
	_skew = skew(image.real.flatten())
	_kurt = kurtosis(image.real.flatten()) + 3.0 # make same as MATLAB
	_max = np.max(image.real)
	_min = np.min(image.real)

	return [ _mean, _var, _skew, _kurt, _max, _min ]


'''
	auto-correlation of lowpass residual (Color Version)
'''
def cov_lr(lores):
	
	_dim = 4 * lores[0]['s'].shape[0] * lores[0]['s'].shape[1]

	# expand residuals and combine slided vectors
	_vec = get_2slide(lores)

#	_mean = np.mean(_vec, axis=0)
#	_vec -= _mean

	_res = np.dot(_vec.T, _vec) / _dim
#
	return _res

'''
	conbine slided residuals (Color Version)
'''
def get_2slide(lores):
	_dim = lores[0]['s'].shape[0] * lores[0]['s'].shape[1]
	_dim = 4 * lores[0]['s'].shape[0] * lores[0]['s'].shape[1]
	_vec = np.zeros((_dim, 15))

	for i in range(len(lores)):
		_lo = expand(lores[i]['s'], 2, 1) / 4
		_lo = _lo.real
		_vec[:, 0 + 5*i] = _lo.reshape(-1,)
#		_vec[:, 0 + 5*i] = _lo.flatten()
		_vec[:, 1 + 5*i] = np.roll(_lo, 2, axis=0).flatten()
		_vec[:, 2 + 5*i] = np.roll(_lo, -2, axis=0).flatten()
		_vec[:, 3 + 5*i] = np.roll(_lo, 2, axis=1).flatten()
		_vec[:, 4 + 5*i] = np.roll(_lo, -2, axis=1).flatten()

	return _vec



'''
	auto-correlation of lowpass residual (Gray Version)
'''
def cov_lr_g(lores):
	
	_dim = 4 * lores['s'].shape[0] * lores['s'].shape[1]

	# expand residuals and combine slided vectors
	_vec = get_2slide_g(lores)


	_res = np.dot(_vec.T, _vec) / _dim

	return _res


'''
	conbine slided residuals (Gary Version)
'''
def get_2slide_g(lores):
	_dim = lores['s'].shape[0] * lores['s'].shape[1]
	_dim = 4 * lores['s'].shape[0] * lores['s'].shape[1]
	_vec = np.zeros((_dim, 15))

	_lo = expand(lores['s'], 2, 1) / 4
	_lo = _lo.real
	_vec[:, 0] = _lo.reshape(-1,)
	_vec[:, 1] = np.roll(_lo, 2, axis=0).flatten()
	_vec[:, 2] = np.roll(_lo, -2, axis=0).flatten()
	_vec[:, 3] = np.roll(_lo, 2, axis=1).flatten()
	_vec[:, 4] = np.roll(_lo, -2, axis=1).flatten()

	return _vec



'''
	get magnitude and real values of bandpass
'''
def trans_b(b):
	b_m = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(np.abs(b[i][j]['s']))
		b_m.append(_tmp)

	b_r = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(b[i][j]['s'].real)
		b_r.append(_tmp)

	b_i = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(b[i][j]['s'].imag)
		b_i.append(_tmp)

	return b_m, b_r, b_i


'''
	get parents of bandpass (Color)
'''
def get_parent(b, lores):
	b_p = []
	b_rp = []
	b_ip = []

	for i in range(len(b)):
		if i < len(b) - 1:
			_dimy = b[i][0]['s'].shape[0] * b[i][0]['s'].shape[1]
			_p = np.zeros((_dimy, len(b[i])))
			_rp = np.zeros_like(_p)
			_ip = np.zeros_like(_p)
			for j in range(len(b[i])):
				# expand parent bandpass
				_tmp = expand(b[i+1][j]['s'], 2) / 4.
				# double phase
				_tmp = double_phase(_tmp).flatten()
				_p[:, j] = np.abs(_tmp) # magitude
				_rp[:, j] = _tmp.real # real value
				_ip[:, j] = _tmp.imag # imaginary value

			_p -= np.mean(_p, axis=0)

			b_p.append(_p)
			b_rp.append(_rp)
			b_ip.append(_ip)

		else:
			# when no parents
			_tmp = expand(lores['s'], 2).real / 4.
			_dimy = _tmp.shape[0] * _tmp.shape[1]
			_rp = np.zeros((_dimy, 5))
			_rp[:, 0] = _tmp.flatten()
			_rp[:, 1] = np.roll(_tmp, 2, axis=1).flatten()
			_rp[:, 2] = np.roll(_tmp, -2, axis=1).flatten()
			_rp[:, 3] = np.roll(_tmp, 2, axis=0).flatten()
			_rp[:, 4] = np.roll(_tmp, -2, axis=0).flatten()
			b_rp.append(_rp)

	return b_p, b_rp, b_ip


'''
	get parents of bandpass (Gray)
'''
def get_parent_g(b, lores):
	b_p = []
	b_rp = []
	b_ip = []

	for i in range(len(b)-1):
		_dimy = b[i][0]['s'].shape[0] * b[i][0]['s'].shape[1]
		_p = np.zeros((_dimy, len(b[i])))
		_rp = np.zeros_like(_p)
		_ip = np.zeros_like(_p)
		for j in range(len(b[i])):
			# expand parent bandpass
			_tmp = expand(b[i+1][j]['s'], 2) / 4.
			# double phase
			_tmp = double_phase(_tmp).flatten()
			_p[:, j] = np.abs(_tmp) # magitude
			_rp[:, j] = _tmp.real # real value
			_ip[:, j] = _tmp.imag # imaginary value

		_p -= np.mean(_p, axis=0)

		b_p.append(_p)
		b_rp.append(_rp)
		b_ip.append(_ip)

	return b_p, b_rp, b_ip


'''
	central auto-correlation of magnitude of bandpass
'''
def autocorr_b(b, MM):
	b_c = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(get_acorr(b[i][j], MM))
		b_c.append(_tmp)

	return b_c



'''
	marginal statistics of magnitude of bandpass
'''
def mrg_b(b):
	b_c = []
	for i in range(len(b)):
		_tmp = []
		for j in range(len(b[i])):
			_tmp.append(mrg_stats(b[i][j]))
		b_c.append(_tmp)

	return b_c


'''
	combine colors	(color version)
'''
def cclr_b(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()
		_list[:, _ori+j] = _tmp1[j].flatten()
		_list[:, 2*_ori+j] = _tmp2[j].flatten()

	return _list

def cclr_bc(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0]
	_tmp1 = bnd[1]
	_tmp2 = bnd[2]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()
		_list[:, _ori+j] = _tmp1[j].flatten()
		_list[:, 2*_ori+j] = _tmp2[j].flatten()

	return _list


def cclr_p(bnd, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]

	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	_list = np.zeros((_dy, 3*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[:, j]
		_list[:, _ori+j] = _tmp1[:, j]
		_list[:, 2*_ori+j] = _tmp2[:, j]

	return _list


def cclr_rp(bnd, bnd_i, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[0][dp]
	_tmp1 = bnd[1][dp]
	_tmp2 = bnd[2][dp]
	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	
	if len(bnd_i) < dp:
		_list = np.zeros((_dy, 3*_ori))
		for j in range(_ori):
			_list[:, j] = _tmp0[:, j]
			_list[:, _ori+j] = _tmp1[:, j]
			_list[:, 2*_ori+j] = _tmp2[:, j]
	else:
		_tmp3 = bnd_i[0][dp]
		_tmp4 = bnd_i[1][dp]
		_tmp5 = bnd_i[2][dp]
		_list = np.zeros((_dy, 6*_ori))
		for j in range(_ori):
			_list[:, j] = _tmp0[:, j]
			_list[:, _ori+j] = _tmp1[:, j]
			_list[:, 2*_ori+j] = _tmp2[:, j]
			_list[:, 3*_ori+j] = _tmp3[:, j]
			_list[:, 4*_ori+j] = _tmp4[:, j]
			_list[:, 5*_ori+j] = _tmp5[:, j]

	return _list


'''
	separate colors	
'''
def sclr_b(cous, ori):

	_dim = ( int(np.sqrt(cous.shape[0])), int(cous.shape[1]/3) )
	_ori = _dim[1]

	_list = []
	for i in range(3):
		_tmp = np.zeros((_dim[0], _dim[0]))
		_vec = []

		for j in range(_ori):
			_tmp = cous[:, _ori*i +j].reshape((_dim[0], _dim[0]))
			_vec.append(_tmp)
		
		_list.append(_vec)

	return _list



'''
	combine orientations (Gray Only)
'''
def cori_b(bnd, dp):
	if len(bnd) < dp:
		return np.array([])

	_tmp0 = bnd[dp]

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, _ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()

	return _list


def cori_bc(bnd, dp):
	if len(bnd) < dp:
		return np.array([])

	_tmp0 = bnd

	_ori = len(_tmp0)
	_dy = _tmp0[0].shape[0]
	_dx = _tmp0[0].shape[1]
	_list = np.zeros((_dy*_dx, _ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[j].flatten()

	return _list


def cori_rp(bnd, bnd_i, dp):
	if len(bnd[0]) < dp:
		return np.array([])

	_tmp0 = bnd[dp]
	_tmp1 = bnd_i[dp]
	_ori = 	_tmp0.shape[1]
	_dy = _tmp0.shape[0]
	
	_list = np.zeros((_dy, 2*_ori))
	for j in range(_ori):
		_list[:, j] = _tmp0[:, j]
		_list[:, _ori+j] = _tmp1[:, j]

	return _list


'''
	separate orientations (Gray ony)	
'''
def sori_b(cous, ori):

	_dim = ( int(np.sqrt(cous.shape[0])), int(cous.shape[1]) )
	_ori = _dim[1]

	_list = []

	for j in range(_ori):
		_tmp = cous[:, j].reshape((_dim[0], _dim[0]))
		_list.append(_tmp)

	return _list

'''
	Create mirrored image
'''
def pad_reflect(image):
	image1 = np.pad(image, [[int(image.shape[0]/2), int(image.shape[0]/2)], [int(image.shape[1]/2), int(image.shape[1]/2)]] , 'reflect')
	return image1

'''
	Steerable Pyramid
'''

class SteerablePyramid():
	def __init__(self, image, xres, yres, n, k, image_name, out_path, verbose):
		self.XRES = xres # horizontal resolution
		self.YRES = yres # vertical resolution
		self.IMAGE_ARRAY = np.asarray(image, dtype='complex')
		self.IMAGE_NAME = image_name
#		self.OUT_PATH = out_path # path to the directory for saving images.
		self.OUT_PATH = out_path + '/{}' # path to the directory for saving images.
		## validation of num. of orientaion
		self.Ks = [4, 6, 8, 10, 12, 15, 18, 20, 30, 60]
		if not k in self.Ks:
			LOGGER.error('illegal number of orientation: {}'.format(str(k)))
			raise ValueError('illegal number of orientation: {}'.format(str(k)))
		self.K = k # num. of orientation
		## validation of depth
		_tmp = np.log2(np.min(np.array([xres, yres])))
		if n  > _tmp - 1:
			LOGGER.error('illegal depth: {}'.format(str(n)))
			raise ValueError('illegal depth: {}'.format(str(n)))
		self.N = n # depth
		self.verbose = verbose # verbose
		self.ALPHAK = 2.**(self.K-1) * math.factorial(self.K-1)/np.sqrt(self.K * float(math.factorial(2.*(self.K-1))))
		self.RES = []
		for i in range(0, self.N):
			_tmp = 2.** i
			self.RES.append( (int(self.XRES/_tmp), int(self.YRES/_tmp)) )
		self.GRID = [] # grid
		self.WX = []
		self.WY = []
		for i in range(0, self.N):
			_x = np.linspace(-np.pi, np.pi, num = self.RES[i][0], endpoint = False)
			_y = np.linspace(-np.pi, np.pi, num = self.RES[i][1], endpoint = False)
			self.WX.append(_x)
			self.WY.append(_y)
			self.GRID.append(np.zeros((_x.shape[0], _y.shape[0])))

		self.RS = [] # polar coordinates
		self.AT = [] # angular cordinates

		# Filters
		self.H0_FILT = np.array([])
		self.L0_FILT = np.array([])
		self.L_FILT = []
		self.H_FILT = []
		self.B_FILT = []

		# Pyramids
		self.H0 = {'f':None, 's':None}
		self.L0 = {'f':None, 's':None}
		self.LR = {'f':None, 's':None}
		self.BND = []
		self.LOW = [] # L1, ...LN

		## CREATE FILTERS
		# caliculate polar coordinates.
		self.RS, self.AT = self.caliculate_polar()
		## for debugging, let coordinates same as Matlab.
#		for n in range(self.N):
#			self.RS[n] = self.RS[n].T
#			self.AT[n] = self.AT[n].T

		# caliculate H0 values on the grid.
		fil = self.calicurate_h0_filter()
		self.H0_FILT = fil

		# caliculate L0 values on the grid.
		fil = self.calicurate_l0_filter()
		self.L0_FILT = fil

		# caliculate L(Low pass filter) values on the grid. 
		fil = self.calicurate_l_filter()
		self.L_FILT = fil

		# caliculate H(fot bandpass filter) values on the grid.
		fil = self.calicurate_h_filter()
		self.H_FILT = fil

		# caliculate B values on the grid.
		fils = self.calicurate_b_filters()
		self.B_FILT = fils

	# caliculate polar coordinates on the grid.
	def caliculate_polar(self):
		pol = []
		ang = []
		for i in range(0, self.N):
			# caliculate polar coordinates(radius) on the grid. they are in [0, inf).
			rs = self.GRID[i].copy()
			yy, xx= np.meshgrid(self.WX[i], self.WY[i])
			rs = np.sqrt((xx)**2 + (yy)**2).T

			# caliculate angular coordinates(theta) on the grid. they are in (-pi, pi].
			at= self.GRID[i].copy()
			_idx = np.where((yy == 0) & (xx < 0))
			at[_idx] = np.pi
			_idx = np.where((yy != 0) | (xx >= 0))
			at[_idx] = np.arctan2(yy[_idx], xx[_idx])

			pol.append(rs)
			ang.append(at)

		return pol, ang

	# caliculate H0 values on the grid.
	def calicurate_h0_filter(self):
		fil = self.GRID[0].copy()
		fil[np.where(self.RS[0] >= np.pi)] = 1
		fil[np.where(self.RS[0] < np.pi/2.)] = 0
		_ind = np.where((self.RS[0] > np.pi/2.) & (self.RS[0] < np.pi))
		fil[_ind] = np.cos(np.pi/2. * np.log2( self.RS[0][_ind]/np.pi) )

		if self.verbose == 1:
			# save image
			plt.clf()
			plt.contourf(self.WX[0], self.WY[0], fil)
			plt.axes().set_aspect('equal', 'datalim')
			plt.colorbar()
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('H0 Filter : Fourier Domain')
			plt.savefig(self.OUT_PATH.format('fil_highpass0.png'))

		return fil

	# caliculate L0 values on the grid.
	def calicurate_l0_filter(self):
		fil = self.GRID[0].copy()
		fil[np.where(self.RS[0] >= np.pi)] = 0
		fil[np.where(self.RS[0] <= np.pi/2.)] = 1
		_ind = np.where((self.RS[0] > np.pi/2.) & (self.RS[0] < np.pi))
		fil[_ind] = np.cos(np.pi/2. * np.log2(2. * self.RS[0][_ind]/np.pi))

		if self.verbose == 1:
			# save image
			plt.clf()
			plt.contourf(self.WX[0], self.WY[0], fil)
			plt.axes().set_aspect('equal', 'datalim')
			plt.colorbar()
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('L0 Filter : Fourier Domain')
			plt.savefig(self.OUT_PATH.format('fil_lowpass0.png'))

		return fil

	# caliculate L filter values on the grid.
	def calicurate_l_filter(self):
		_f = []
		for i in range(0, self.N):
			fil = self.GRID[i].copy()
			fil[np.where(self.RS[i] >= np.pi/2.)] = 0
			fil[np.where(self.RS[i] <= np.pi/4.)] = 1
			_ind = np.where((self.RS[i] > np.pi/4.) & (self.RS[i] < np.pi/2.))
			fil[_ind] = np.cos(np.pi/2. * np.log2(4. * self.RS[i][_ind]/np.pi))

			_f.append(fil)

			if i == 0 and self.verbose == 1:
				plt.clf()
				plt.contourf(self.WX[i], self.WY[i], fil)
				plt.axes().set_aspect('equal', 'datalim')
				plt.colorbar()
				plt.xlabel('x')
				plt.ylabel('y')
				plt.title('Lowpass filter of Layer{} : Fourier Domain'.format(str(i)))
				plt.savefig(self.OUT_PATH.format('fil_lowpass-layer{}.png'.format(str(i))))

		return _f

	# caliculate H0 filter values on the grid.
	def calicurate_h_filter(self):
		_f = []
		for i in range(0, self.N):
			fil = self.GRID[i].copy()
			fil[np.where(self.RS[i] >= np.pi/2.)] = 1
			fil[np.where(self.RS[i] <= np.pi/4.)] = 0
			_ind = np.where((self.RS[i] > np.pi/4.) & (self.RS[i] < np.pi/2.))
			fil[_ind] = np.cos(np.pi/2. * np.log2(2.*self.RS[i][_ind]/np.pi))

			_f.append(fil)		

			if i == 0 and self.verbose == 1:
				plt.clf()
				plt.contourf(self.WX[i], self.WY[i], fil)
				plt.axes().set_aspect('equal', 'datalim')
				plt.colorbar()
				plt.xlabel('x')
				plt.ylabel('y')
				plt.title('Highpass filter of Layer{} : Fourier Domain'.format(str(i)))
				plt.savefig(self.OUT_PATH.format('fil_highpass-layer{}.png'.format(str(i))))

		return _f

	def calicurate_b_filters(self):
		f_ = []
		for i in range(0, self.N):
			fils_ = []

			for k in range(self.K):
				# caliculate Bk values on the grid.
				fil_= np.zeros_like(self.GRID[i], dtype=complex)
				th1= self.AT[i].copy()
				th2= self.AT[i].copy()

				th1[np.where(self.AT[i] - k*np.pi/self.K < -np.pi)] += 2.*np.pi
				th1[np.where(self.AT[i] - k*np.pi/self.K > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th1 - k*np.pi/self.K) <= np.pi/2.)
				fil_[ind_] = self.ALPHAK * (np.cos(th1[ind_] - k*np.pi/self.K))**(self.K-1)
#				fil_[ind_] = complex(0,1)**k * self.ALPHAK * (np.cos(th1[ind_] - k*np.pi/self.K))**(self.K-1)
				th2[np.where(self.AT[i] + (self.K-k)*np.pi/self.K < -np.pi)] += 2.*np.pi
				th2[np.where(self.AT[i] + (self.K-k)*np.pi/self.K > np.pi)] -= 2.*np.pi
				ind_ = np.where(np.absolute(th2 + (self.K-k) * np.pi/self.K) <= np.pi/2.)
				fil_[ind_] = self.ALPHAK * (np.cos(th2[ind_]+ (self.K-k) * np.pi/self.K))**(self.K-1)
#				fil_[ind_] = complex(0,1)**k * self.ALPHAK * (np.cos(th2[ind_]+ (self.K-k) * np.pi/self.K))**(self.K-1)

				fil_= self.H_FILT[i] * fil_
				fils_.append(fil_.copy())

				if i == 0 and self.verbose == 1:
					plt.clf()
					plt.contourf(self.WX[i], self.WY[i], np.abs(fil_))
					plt.axes().set_aspect('equal', 'datalim')
					plt.colorbar()
					plt.xlabel('x')
					plt.ylabel('y')
					plt.title('Bandpass filter of layer{} : Fourier Domain'.format(str(i)))
					plt.savefig(self.OUT_PATH.format('fil_bandpass{}-layer{}.png'.format(str(k), str(i))))

					plt.clf()
					plt.contourf(self.WX[i], self.WY[i], np.abs(fil_ * self.L0_FILT))
					plt.axes().set_aspect('equal', 'datalim')
					plt.colorbar()
					plt.xlabel('x')
					plt.ylabel('y')
					plt.title('Bandpass * Lowpass filter of layer{}'.format(str(i)))
					plt.savefig(self.OUT_PATH.format('fil_lo-bandpass{}-layer{}.png'.format(str(k), str(i))))
	
			f_.append(fils_)

		return f_

	# create steerable pyramid
	def create_pyramids(self):

		# DFT
		ft = np.fft.fft2(self.IMAGE_ARRAY)
		_ft = np.fft.fftshift(ft)

		# apply highpass filter(H0) and save highpass resudual
		h0 = _ft * self.H0_FILT
		f_ishift = np.fft.ifftshift(h0)
		img_back = np.fft.ifft2(f_ishift)
		# frequency
		self.H0['f'] = h0.copy()
		# space
		self.H0['s'] = img_back.copy()

		if self.verbose == 1:
			_tmp = np.absolute(img_back)
			Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-h0.png'.format(self.IMAGE_NAME)))

		# apply lowpass filter(L0).
		l0 = _ft * self.L0_FILT
		f_ishift = np.fft.ifftshift(l0)
		img_back = np.fft.ifft2(f_ishift)
		self.L0['f'] = l0.copy()
		self.L0['s'] = img_back.copy()

		if self.verbose == 1:
			_tmp = np.absolute(img_back)
			Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-l0.png'.format(self.IMAGE_NAME)))

		# apply bandpass filter(B) and downsample iteratively. save pyramid
		_last = l0
		for i in range(self.N):
			_t = []
			for j in range(len(self.B_FILT[i])):
				_tmp = {'f':None, 's':None}
				lb = _last * self.B_FILT[i][j]
				f_ishift = np.fft.ifftshift(lb)
				img_back = np.fft.ifft2(f_ishift)
				# frequency
				_tmp['f'] = lb
				# space
				_tmp['s'] = img_back
				_t.append(_tmp)

				#if self.verbose == 1:
				#	_tmp = np.absolute(img_back.real)
				#	Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-layer{}-lb{}.png'.format(self.IMAGE_NAME, str(i), str(j))))
		
			self.BND.append(_t.copy())

			# apply lowpass filter(L) to image(Fourier Domain) downsampled.
			l1 = _last * self.L_FILT[i]

			## Downsampling
			# filter for cutting off high frequerncy(>np.pi/2).
			# (Attn) steerable pyramid is basically anti-aliases. see http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
			# this filter is not needed actually ,but prove anti-aliases characteristic of the steerable filters.
			
			down_fil = np.zeros(_last.shape)
			quant4x = int(down_fil.shape[1]/4)
			quant4y = int(down_fil.shape[0]/4)
			down_fil[quant4y:3*quant4y, quant4x:3*quant4x] = 1

			# apply downsample filter.
			dl1 = l1 * down_fil

			# extract the central part of DFT
			down_image = np.zeros((2*quant4y, 2*quant4x), dtype=complex)
			down_image = dl1[quant4y:3*quant4y, quant4x:3*quant4x]
#
			f_ishift = np.fft.ifftshift(down_image)
			img_back = np.fft.ifft2(f_ishift)
			self.LOW.append({'f':down_image, 's':img_back})
			
			#if self.verbose == 1:
			#	_tmp = np.absolute(img_back)
			#	Image.fromarray(np.uint8(_tmp), mode='L').save(self.OUT_PATH.format('{}-residual-layer{}.png'.format(self.IMAGE_NAME, str(i))))

			_last = down_image

		# lowpass residual
		self.LR['f'] = _last.copy()
		self.LR['s'] = img_back.copy()

		return None


	# image reconstruction from steerable pyramid in Fourier domain.
	def collapse_pyramids(self):
		_resid = self.LR['f']

		for i in range(self.N-1,-1,-1):
			## upsample residual
			_tmp_tup = tuple(int(2*x) for x in _resid.shape)
			_tmp = np.zeros(_tmp_tup, dtype=np.complex)
			quant4x = int(_resid.shape[1]/2)
			quant4y = int(_resid.shape[0]/2)
			_tmp[quant4y:3*quant4y, quant4x:3*quant4x] = _resid
			_resid = _tmp

			_resid = _resid * self.L_FILT[i]
			for j in range(len(self.B_FILT[i])):
				_resid += self.BND[i][j]['f'] * self.B_FILT[i][j]
	
		# finally reconstruction is done.
		recon = _resid * self.L0_FILT + self.H0['f'] * self.H0_FILT

		return recon

	# clear the steerable pyramid
	def clear_pyramids(self):
		self.H0['f'] = np.zeros_like(self.H0['f'])
		self.H0['s'] = np.zeros_like(self.H0['s'])
		self.L0['f'] = np.zeros_like(self.L0['f'])
		self.L0['s'] = np.zeros_like(self.L0['s'])
		self.LR['f'] = np.zeros_like(self.LR['f'])
		self.LR['s'] = np.zeros_like(self.LR['s'])

		for i in range(len(self.BND)):
			for j in range(len(self.BND[i])):
				self.BND[i][j]['s'] = np.zeros_like(self.BND[i][j]['s'])
				self.BND[i][j]['f'] = np.zeros_like(self.BND[i][j]['f'])
		
		for i in range(len(self.LOW)):
			self.LOW[i]['s'] = np.zeros_like(self.LOW[i]['s'])
			self.LOW[i]['f'] = np.zeros_like(self.LOW[i]['f'])

		return

class TextureAnalysis():
	def __init__(self, image, xres, yres, n, k, m):
		self.IMAGE_ARRAY = image # array
		self.XRES = xres # horizontal resolution
		self.YRES = yres # vertical resolution
		self.K = k # num. of orientation
		self.N = n # depth
		self.M = m # window size (must be odd)

		### mean and covariances of original image
		self.MEAN_RGB = np.array([])
		self.COV_RGB = np.array([])

		### marginal statistics of original image
		self.IM_MAR = np.array([])
		self.IM_CA = np.array([])

		## Steerable Pyramid
		self.LR = {}
		self.LR_MAR = np.array([])
		self.LR_MMEAN = 0.
		self.LR_CA = np.array([])

		self.BND = []
		self.BND_M = []
		self.BND_MCOR = []
		self.BND_MMAR = []
		self.BND_R = []
		self.BND_P = []
		self.BND_RP = []
		self.BND_IP = []
		self.H0 = {}
		self.H0_PRO = 0.

		self.COV_LR = np.array([])

		self.CF_MAR = []
		self.CF_CA = []
		self.CF_COUS = []
		self.CF_RCOU = []
		self.CF_CPAR = []
		self.CF_RPAR = []

	'''
		Analyse
	'''
	def analyse(self):

		# marginal statistics of original image.
		self.IM_MAR = mrg_stats(self.IMAGE_ARRAY)

		# covariance matrix of orignal image
		self.IM_VAR = np.var(self.IMAGE_ARRAY)

		# (a1) central auto correlation of image
		self.IM_CA = get_acorr(self.IMAGE_ARRAY, self.M)

		# add noise (according to textureAnalysis.m)
		# this is not neccesary.
#		noise = np.random.normal(0, 1, self.IMAGE_ARRAY.shape[0] * self.IMAGE_ARRAY.shape[1]).reshape(self.IMAGE_ARRAY.shape[0], self.IMAGE_ARRAY.shape[1])
#		noise = (noise * self.IM_MAR[4] - self.IM_MAR[5]) / 1000.
#		self.IMAGE_ARRAY = self.IMAGE_ARRAY + noise

		#-----------------------------------------
		# create steerable pyramid
		_sp = SteerablePyramid(self.IMAGE_ARRAY, self.XRES, self.YRES, self.N, self.K, '', '', 0)
		_sp.create_pyramids()

		#-----------------------------------------
		# lowpass residual
		lr = copy.deepcopy(_sp.LR)
		## marginal statistics of LR
		self.LR_MMEAN = np.mean(np.abs(lr['s']))
		## subtract mean : according to textureColorAnalysis.m
		_mean = np.mean(lr['s'].real)
		lr['s'] = lr['s'].real - _mean
		lr['f'] = np.fft.fftshift(np.fft.fft2(lr['s']))
		self.LR = lr
		## marginal statistics of lowpass residual
		## get L0 of LR of small size.(this tric is for synthesis process)
		_s = SteerablePyramid(lr['s'], lr['s'].shape[1], lr['s'].shape[0], 1, self.K, '', '', 0)
		_s.create_pyramids()

		# initial value of coarse to fine
		im = _s.L0['s'].real
		## marginal statistics of LR
		self.LR_MAR = mrg_stats(im)
		## central auto correlation of lowpass residuals
		self.LR_CA = get_acorr(im, self.M)

		#-----------------------------------------
		# bandpass
		bnd = copy.deepcopy(_sp.BND)
		self.BND = bnd

		_b_m, _b_r, _b_i = trans_b(bnd)
		## marginal statistics of magnitude
		self.BND_MMAR = mrg_b(_b_m)
		## magnitude
		for i in range(len(_b_m)):
			for k in range(len(_b_m[i])):
				_b_m[i][k] -= np.mean(_b_m[i][k])
		self.BND_M = _b_m
		## central auto-correlation of magnitude (this is 'ace' in textureColorAnalysis.m)
		self.BND_MCOR = autocorr_b(_b_m, self.M)
		## real values
		self.BND_R = _b_r

		_b_p, _b_rp, _b_ip = get_parent_g(copy.deepcopy(_sp.BND), lr)
		## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
		self.BND_P = _b_p
		## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		self.BND_RP = _b_rp
		## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
		self.BND_IP = _b_ip

		#-----------------------------------------
		# highpass residual
		_b = copy.deepcopy(_sp.H0)
		self.H0 = _b
		## marginal statistics of highpass residual
		self.H0_PRO = np.var(_b['s'].real)

		#-----------------------------------------
		# statistics for coarse to fine
		_ms = []
		_ac = []
		_cou = []
		for dp in range(self.N-1, -1, -1):
			# create steerable pyramid (create filters only)
			_z = np.zeros_like(bnd[dp][0]['s'])
			_s = SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, self.K, '', '', 0)
			# reconstruct dummy pyramid
			_recon = np.zeros_like(_z)
			for k in range(self.K):
				_recon += _s.B_FILT[0][k] * bnd[dp][k]['f']
			_recon = _recon * _s.L0_FILT
			_recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

			# expand
			im = expand(im, 2).real / 4.
			im = im.real + _recon

			# marginal statistics
			_ms.append(mrg_stats(im))			
			# central auto correlations
			_ac.append(get_acorr(im, self.M))	

		self.CF_MAR = _ms[::-1]
		self.CF_CA = _ac[::-1] # this is 'acr' in textureColorAnalysis.m

		#-----------------------------------------
		# auto correlartion matrix of lowpass residual (2 slided)
		self.COV_LR = cov_lr_g(self.LR)

		# coarse to fine loop (Get statistics of Bandpass)
		for dp in range(self.N-1, -1, -1):
				
			# combine colors
			cousins = cori_b(self.BND_M, dp)
			## save covariance matrices
			_tmp = np.dot(cousins.T, cousins) / cousins.shape[0]
			self.CF_COUS.append(copy.deepcopy(_tmp))

			bnd_r = []
			for k in range(self.K):
				bnd_r.append(self.BND[dp][k]['s'].real)
			
			rcousins = cori_bc(bnd_r, dp)
			# save covariance matrices
			_tmp = np.dot(rcousins.T, rcousins) / rcousins.shape[0]
			self.CF_RCOU.append(copy.deepcopy(_tmp))
			
			if dp < self.N-1:
				parents = self.BND_P[dp]
				# save covariance matrices
				_tmp = np.dot(cousins.T, parents) / cousins.shape[0]
				self.CF_CPAR.append(copy.deepcopy(_tmp))

				rparents = cori_rp(self.BND_RP, self.BND_IP, dp)
				# save covariance matrices
				_tmp = np.dot(rcousins.T, rparents) / rcousins.shape[0]
				self.CF_RPAR.append(copy.deepcopy(_tmp))


		self.CF_COUS = self.CF_COUS[::-1]
		self.CF_RCOU = self.CF_RCOU[::-1]
		self.CF_RPAR = self.CF_RPAR[::-1]
		self.CF_CPAR = self.CF_CPAR[::-1]

		return None