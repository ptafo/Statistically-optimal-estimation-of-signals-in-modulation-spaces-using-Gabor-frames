__author__ = 'ptafo'
from __future__ import division
import multiprocessing as mp
import pandas as pd
import argparse
from doppler import *



def parse_args():
    parser = argparse.ArgumentParser(description='Run simulation')
    parser.add_argument('--args', dest='ARGS', help='arguments', nargs='+')
    args = parser.parse_args()
    return args


def denoised_doppler(A, B, wvlt, lvl, SNR=10, expo=2, Fs=1000, ret='visu', Seed=1):
    signal = doppler(A, B, expo, Fs).sig
    res = sim_wavelet(signal, wvlt, lvl, SNR, 0, ret, 1, Seed)

    return (Fs,) + res


def rate_doppler(A, B, expo, wvlt, lvl, SNR=10, Sims=1, ncores=4, ret='visu'):

    A, B, expo, SNR, Sims, ncores = map(int, [A, B, expo, SNR, Sims, ncores])
    wvlt, lvl, ret = map(str, [wvlt, lvl, ret])

    n = np.linspace(1000, 10000, 10, dtype='int')
    pool = mp.Pool(processes=ncores)
    results = [pool.apply_async(denoised_doppler, args=(A, B, wvlt, lvl, SNR, expo, i, ret, j)) for i in n for j in
               range(Sims)]
    results = [p.get() for p in results]
    resD = pd.DataFrame(results)
    name = "Wavelet_"+"rate_A:" + str(A) + "_B:" + str(B) + "_wvlt:" + str(wvlt) + "_lvl:" + str(lvl) + "_expo:" + str(
        expo) + "_" + "ret:" + str(ret)
    resD.to_pickle(name + ".pkl")
    return n, resD


def rate_doppler_best(A, B, expo, wvlt, lvl, SNR=10, Sims=1, ncores=4, ret='best',n=2000):

    A, B, expo, SNR, Sims, ncores = map(int, [A, B, expo, SNR, Sims, ncores])
    wvlt, lvl, ret = map(str, [wvlt, lvl, ret])

    n = np.linspace(1000, 10000, 10, dtype='int')
    pool = mp.Pool(processes=ncores)
    results = [pool.apply_async(denoised_doppler, args=(A, B, wvlt, lvl, SNR, expo, i, ret, j)) for i in n for j in
               range(Sims)]
    results = [p.get() for p in results]
    resD = pd.DataFrame(results)
    name = "Wavelet_"+"rate_A:" + str(A) + "_B:" + str(B) + "_wvlt:" + str(wvlt) + "_lvl:" + str(lvl) + "_expo:" + str(
        expo) + "_" + "ret:" + str(ret)
    resD.to_pickle(name + ".pkl")
    return n, resD


if __name__ == "__main__":
    args = parse_args()
    results = rate_doppler_best(*args.ARGS)
