__author__ = 'ptafo'
from __future__ import division
import multiprocessing as mp
import pandas as pd
import argparse
from functions.doppler import


def parse_args():
    parser = argparse.ArgumentParser(description='Run Simulation')
    parser.add_argument('--args', dest='ARGS', help='arguments', nargs='+')
    args = parser.parse_args()
    return args




def denoised_doppler(A,B,alpha,beta,window,LW=None,SNR = 10,expo=2,Fs = 1000,ret='visu',Seed=1):

    signal = doppler(A,B,expo,Fs).sig
    res = sim_gabor_real(signal,alpha,beta,window,int(LW*Fs),SNR,0,ret,1,Seed)

    return  (Fs,) + res



def rate_doppler(A, B, expo, alpha, beta, w1,w2, LW=None, SNR = 10,Sims=1,ncores=4,ret='visu'):


    A, B, expo, alpha, beta, SNR, Sims, ncores = map(int,[A, B, expo, alpha, beta, SNR, Sims, ncores])
    LW = float(LW)
    window = (str(w1),float(w2))#var_g = float(var_g)
    ret = str(ret)

    n = np.linspace(1000,10000,10,dtype='int')
    LWs = [LW]#[50,100,200,500,1000,1500,2000]
    pool = mp.Pool(processes=ncores)
    results = [pool.apply_async(denoised_doppler, args=(A,B,alpha,beta,window,LW,SNR,expo,i,ret,j)) for i in n for j in range(Sims) for LW in LWs]
    results = [p.get() for p in results]
    resD = pd.DataFrame(results)#, columns=[ 'sigma', 'alpha', 'var_g', 'US', 'ES', 'EH','Fs'])
    #resD['n'] = n
    name = "Gabor_"+"rate_A:"+ str(A)+"_B:"+ str(B)+"_alpha:"+ str(alpha)+"_beta:"+ str(beta)+"_window:"+ str(window)+"_"+ "ret:" + str(ret)
    resD.to_pickle(name + ".pkl")
    return n, resD


if __name__ == "__main__":

    args = parse_args()
    results = rate_doppler(*args.ARGS)


