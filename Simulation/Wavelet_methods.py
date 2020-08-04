__author__ = 'ptafo'
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.optimize import basinhopping




def error_wavelet(signal,wavelet_noisy,wvlt,th_method,Lambda,max =float('Inf')):
    # evaluate error from threshold of the details coefficients

    L             = len(signal)
    wavelet_thres = [pywt.threshold(i, Lambda, th_method) for i in wavelet_noisy[1:]]
    wavelet_thres.insert(0,wavelet_noisy[0])
    f_hat         = pywt.waverec(wavelet_thres, wvlt)[:L]
    if Lambda<0 or Lambda>max:
        err = float('Inf')
    else:
        err = (np.linalg.norm(f_hat - signal) ** 2)/L

    return err,f_hat



def sim_wavelet(signal, wvlt, lvl, SNR=1, plot=0, ret='visu', facT = 1, Seed=1):
    # simulation wavelet denoising of a signal
    # wvlt: wavelet family
    # lvl: level either an integer or 'max'
    # SNR: Signal-to-noise ratio
    # ret: Threshold type either 'best' for the best threshold
    #	                         'visu' for the universal threshold
    #			      or 'facT' for a the given threshold		

    # set seed
    np.random.seed(Seed)

    # white noise
    L     = len(signal)
    sigma = np.sqrt(np.mean(signal ** 2) / SNR)
    noise = np.random.normal(0, sigma, size=L)

    # level
    if lvl == "max":
        LVL = None
    else:
        LVL = int(lvl)


    # denoising
    wavelet_noisy = pywt.wavedec((signal + noise), wvlt, level=LVL)

    if ret =='best':
        wavelet_noisy_max2 = np.max([np.append(np.abs(x.real), np.abs(x.imag)).max() for x in np.abs(wavelet_noisy)])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": ((0, wavelet_noisy_max2),)}
        BS = basinhopping(lambda y: error_wavelet(signal, wavelet_noisy, wvlt, 'soft', y)[0], 0, minimizer_kwargs=minimizer_kwargs, niter=500,stepsize=wavelet_noisy_max2/5)
        BH = basinhopping(lambda y: error_wavelet(signal, wavelet_noisy, wvlt, 'hard', y)[0], 0, minimizer_kwargs=minimizer_kwargs, niter=500,stepsize=wavelet_noisy_max2/5)
        # BS = optimize.minimize_scalar(lambda y: error_wavelet(signal, wavelet_noisy, wvlt, 'soft', y)[0], bounds=(0, wavelet_noisy_max2), method='bounded', tol=1.48e-08)
        # BH = optimize.minimize_scalar(lambda y: error_wavelet(signal, wavelet_noisy, wvlt, 'hard', y)[0], bounds=(0, wavelet_noisy_max2), method='bounded', tol=1.48e-08)
        res = BS.x, BH.x, BS.fun, BH.fun
    elif ret =='visu':
        US  = sigma * np.sqrt(2*np.log(L))
        ES  = error_wavelet(signal, wavelet_noisy, wvlt, 'soft', US)
        EH  = error_wavelet(signal, wavelet_noisy, wvlt, 'hard', US)
        res =  US, ES[0], EH[0]
    else:
        TH = facT
        ES = error_wavelet(signal, wavelet_noisy, wvlt, 'soft', TH)
        EH = error_wavelet(signal, wavelet_noisy, wvlt, 'hard', TH)
        res = TH, ES[0], EH[0]

    # plots
    if plot:
        plt.clf()
        plt.plot(signal, alpha=0.3)
        plt.plot(ES[-1], linestyle='--', alpha=2)

    return (Seed, sigma,  wvlt, lvl) + res



def denoise_wavelet(signal, wvlt, lvl,th_method='soft', plot=0, ret='visu', sigma=1,facT = 1):
    # signal denoising through wavelet shrinkage
    # wvlt: wavelet family
    # lvl: level either an integer or 'max'
    # ret: Threshold type either 'visu' for the universal threshold
    #			      or 'facT' for a the given threshold
    # sigma: estimated noise standard deviation


    L     = len(signal)

    # level
    if lvl == "max":
        LVL = None
    else:
        LVL = int(lvl)


    # denoising
    wavelet_noisy = pywt.wavedec(signal, wvlt, level=LVL)

    if ret =='visu':
        US  = sigma * np.sqrt(2*np.log(L))
        ES  = error_wavelet(signal, wavelet_noisy, wvlt, th_method, US)
    else:
        TH = facT
        ES = error_wavelet(signal, wavelet_noisy, wvlt, th_method, TH)

    # plots
    if plot:
        plt.clf()
        plt.plot( signal, alpha=0.3)
        plt.plot( ES[-1], linestyle='--', alpha=2)


    return ES[-1]
