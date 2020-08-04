__author__ = 'ptafo'
from __future__ import division
from ltfatpy import plotdgtreal
import copy
from Gabor_methods import *
from Wavelet_methods import *




class doppler():


    def __init__(self, A, B, expo=2, Fs=100):
        self.n = np.linspace(0, 1, Fs)
        self.A = A
        self.B = B
        self.Fs = Fs
        self.expo = expo
        self.sigma = None
        self.spec = None
        self.sig = np.sin(2 * np.pi * B * np.exp(-A * np.abs(self.n - 1 / 2) ** expo) * self.n)



    def __repr__(self):
        return 'Object: A={0}, B={1}, expo={2}, Fs={3}'.format(self.A, self.B, self.expo, self.Fs)



    def getspec(self, alpha=1, beta=1, window=('bsplines',4), LW=None):
	# generate spectrogram of the signal
	# NO PLOT TO GENERATE
	# alpha  : time step
	# beta   : frequency step
	# window : either (bsplines, level) or (gauss, variance)
	# LW     : length of the window in terms of samples

        L = len(self.sig)
        a, M, Lg, N, Ngood = gabimagepars(L, int(L / alpha), int(L / beta))
        if LW is None: LW = Lg
        if window[0] == 'bsplines':
            gs = gabwin(mySpline(window[1], LW, 0)[-1], a, M, LW)[0]
        else:
            gs = gabwin({'name': ('gauss'), 'tfr': window[1]}, a, M, LW)[0]

        ga = gabdual(gs, a, M)
        self.frame = a, M, gs, ga
        self.spec = dgtreal(self.sig, ga, a, M)[0]



    def plot(self, plotspec=0, dyn=50, **kwargs):
	# plot spectrogram
	# prior generate spectrogram

        plt.figure()
        if not plotspec:
            plt.plot(self.n, self.sig)
        else:
            a, M, gs, ga = self.frame
            plotdgtreal(self.spec, a, M,dynrange=dyn, fs=self.Fs, display=1, **kwargs)



    def noisy(self, SNR=10):
	# generate a noisy version of the signal by adding white to the corresponding SNR
	# SNR: signal-to-noise ratio

        new = copy.deepcopy(self)
        new.sigma = np.sqrt(np.mean(new.sig ** 2) / SNR)
        new.sig = new.sig + np.random.normal(0, new.sigma, size=len(new.sig))
        return new



    def denoise_D_gabor(self, alpha, beta, window, LW=None, th_method='soft', ret='visu', plot=0, facT=1):
	# Gabor coefficients shrinkage
	# Donoho's universal threshold is used
	# th_method: threshold method: Hard or soft thresholding


        new = copy.deepcopy(self)
        new.sig = denoise_gabor_real(new.sig, alpha, beta, window, LW, th_method, plot, ret, new.sigma, facT)
        return new



    def denoise_D_wavelet(self, wvlt, lvl, th_method='soft', ret='visu', plot=0, facT=1):
	# wavelet coefficients shrinkage
	# Donoho's universal threshold is used
	# th_method: threshold method: Hard or soft thresholding

        new = copy.deepcopy(self)
        new.sig = denoise_wavelet(new.sig, wvlt, lvl, th_method, plot, ret, new.sigma, facT)
        return new

