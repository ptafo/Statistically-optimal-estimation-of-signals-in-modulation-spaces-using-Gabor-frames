__author__ = 'ptafo'
from __future__ import division
import numpy as np
import os.path
from ltfatpy import plotdgtreal
import copy
import scipy.io.wavfile as wave
from Gabor_methods import *
from Wavelet_methods import *


class realSound():

    # class for audio manipulation

    def __init__(self, path):
        self.path = path
        self.Fs, self.sig = wave.read(path)
        self.sig = self.sig.astype(float)
        if len(self.sig.shape)==2:
            self.sig = self.sig[:,0]
        self.length = len(self.sig)/self.Fs

    def __repr__(self):
        return 'Location:{0}, length={1}, Fs={2}'.format(self.path, self.length, self.Fs)



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
            plt.plot(self.sig)
        else:
            a, M, gs, ga = self.frame
            plotdgtreal(self.spec, a, M,dynrange=dyn, fs=self.Fs, display=1, **kwargs)

    def noisy(self, SNR=10):
        # generate a noisy version of the signal by adding white to the corresponding SNR
        # SNR: signal-to-noise ratio in decibels

        new = copy.deepcopy(self)
        new.sigma = np.sqrt(np.mean(new.sig ** 2) / (10**(SNR/10)))
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


    def diff(self,B):
        # difference between to signal l2 norm or decibels

        return 10*np.log10(np.mean((self.sig-B.sig)**2)),np.mean((self.sig-B.sig)**2)

    def save(self,name,path=None):
        # save wav file

        if path==None:
            wave.write(os.path.dirname(self.path)+"/"+ name+".wav", self.Fs, self.sig.astype('int16'))
        else:
            wave.write(path+ "/" + name + ".wav", self.Fs, self.sig.astype('int16'))




