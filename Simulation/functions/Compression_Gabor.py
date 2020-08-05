from __future__ import division
from Gabor_methods import *


def modNorm(matrix,weight,p=1,C2=1):
    # evaluate the discrete modulation norm

    V = np.power( np.abs(matrix), float(p))
    # meshgrid
    i    = np.arange(0,matrix.shape[0],1)
    j    = np.arange(0,matrix.shape[1],1)
    grid_i,grid_j = np.meshgrid(i, j,indexing='ij', sparse=1)
    grid = weight(grid_i,grid_j)
    grid = np.power(grid,  float(p))

    return np.power(np.multiply(V,grid).sum(),1/float(p))*C2


def mu_out_N(matrix,N):
    # find the the N-th highest mu

    mu   = np.sort(abs(matrix).flatten())[-np.array(N)]

    return mu



def reconstruct_gabor(matrix,mu,f,gs,a,M):
    # evaluate error from reconstructing the signal from the N highest coefficients

    L = len(f)
    N    = (abs(matrix) >=mu).sum()
    C = np.where(abs(matrix)<mu,0,matrix)
    f_mu = (idgtreal(C,gs,a,M,L)[0]).real
    err  = (np.linalg.norm(f_mu - f)**2)/L

    return err, mu, N, f_mu


def compression_gabor(signal,alpha,beta,window, LW,Ns,plot=0,*args,**kwargs):
    # Signal compression of Gabor coefficients

    L = len(signal)

    # Gabor Frame
    a, M, Lg, N, Ngood = gabimagepars(L, int(L / alpha), int(L / beta))
    if LW is None:  LW = Lg
    if window[0] == 'bsplines':
        gs = gabwin(mySpline(window[1], LW, 0)[-1], a, M, LW)[0]
    else:
        gs = gabwin({'name': ('gauss'), 'tfr': window[1]}, a, M, LW)[0]

    ga = gabdual(gs, a, M)

    coef  = dgtreal(signal, ga, a, M)[0]

    res = []
    for i in Ns:
        mu = mu_out_N(coef, i)
        recon = reconstruct_gabor(coef, mu, signal, gs, a, M)
        res = res + [(i,)+recon]

    plt.plot([v[0] for v in res],[v[1] for v in res],*args,**kwargs)


    if plot:
        plt.figure()
        plt.clf()
        plt.plot(signal)
        plt.plot(recon[-1])

    return res#recon[-1],recon[0],Ns#,Ns*100/N#int(Ns*coef.size/100)



