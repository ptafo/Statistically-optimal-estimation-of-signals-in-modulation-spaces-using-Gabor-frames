__author__ = 'ptafo'
from __future__ import division
import multiprocessing as mp
import pandas as pd
from realSound import *


def denoise_realsound(test,LW,SNR=10,Seed=1,th_method='hard'):

    np.random.seed(Seed)
    test_noisy = test.noisy(SNR)
    # test_denoised_hard = test_noisy.denoise_D_gabor(100,100,('gauss',1),None,'hard')
    test_denoised = test_noisy.denoise_D_gabor(100, 100, ('bsplines', 4), LW, th_method)
    test_denoised_w = test_noisy.denoise_D_wavelet('bior4.4', 'max', th_method)

    return Seed, SNR, test_denoised.diff(test)[1], test_denoised_w.diff(test)[1]



def MSE_realsound(test,LW,SNRs,Sims=10,ncores=4,filename='audio',th_method='hard'):

    pool = mp.Pool(processes=ncores)
    results = [pool.apply_async(denoise_realsound, args=(test,LW,i,j,th_method)) for i in SNRs for j in range(Sims)]
    results = [p.get() for p in results]
    resD = pd.DataFrame(results)
    resD.to_pickle('MSE_'+filename+"_" + th_method +"_thresholding"+ ".pkl")
    resVar_w_v = resD.groupby([1]).mean()[[2, 3]]
    return 10*np.log10(resVar_w_v)


