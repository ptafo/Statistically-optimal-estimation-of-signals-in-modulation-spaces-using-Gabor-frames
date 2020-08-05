__author__ = 'ptafo'
from __future__ import division
from Modulationspace_Gabor import mu_out_N
import numpy as np
import matplotlib.pyplot as plt
import pywt





def reconstruct_wavelet(list,mu,f,wvlt):

    L = len(f)
    listFlat = np.array([item for sublist in list for item in sublist])
    N    = (abs(listFlat) >= mu).sum()
    C = [np.where(abs(sublist)<mu,0,sublist) for sublist in list]
    f_mu = pywt.waverec(C, wvlt)
    err  = (np.linalg.norm(f_mu - f)**2)/L

    return err, mu, N, f_mu


def compression_wavelet(signal,wvlt,lvl,Ns,plot=0,*args,**kwargs):

    # signal and noise
    L = len(signal)

    #Gabor Lattice
    if lvl == "max":
        LVL = None
    else:
        LVL = int(lvl)

    coef  = pywt.wavedec(signal, wvlt, level=LVL)
    coefFlat = np.array([item for sublist in coef for item in sublist])

    res = []
    for i in Ns:
        mu = mu_out_N(coefFlat, i)
        recon = reconstruct_wavelet(coef,mu,signal,wvlt)
        res = res + [(i,)+recon]

    plt.plot([v[0] for v in res],[v[1] for v in res],*args,**kwargs)



    if plot:
        plt.clf()
        plt.plot(signal)
        plt.plot(recon[-1])

    return res#recon[0],Ns#,Ns*100/len(coefFlat)#int(Ns*len(coefFlat)/100)



