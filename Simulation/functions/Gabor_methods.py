from __future__ import division
from ltfatpy import gabimagepars, dgtreal, gabwin, idgtreal, gabdual
from ltfatpy.sigproc.thresh import thresh
from scipy.optimize import basinhopping, minimize_scalar
from Splines_function import *




def error_gabor(f, coeff, gs, a, M, th_method, Lambda,max =float('Inf')):
    # evaluate error from threshold of the real and imaginery coefficients

    L = len(f)
    c_thrshd = thresh(coeff.real, float(Lambda), th_method)[0] + 1j * thresh(coeff.imag, float(Lambda), th_method)[0]
    f_hat = idgtreal(c_thrshd, gs, a, M, L)[0]
    if Lambda<0 or Lambda>max:
        err = float('Inf')
    else:
        err = (np.linalg.norm(f_hat - f)**2)/L

    return err,f_hat


def sim_gabor_real(signal, alpha, beta, window, LW=None, SNR=1, plot=0, ret='visu', facT = 1, Seed=1):
    # simulation gabor denoising of a signal
    # alpha, beta: time and frequency step
    # window     : either (bsplines, level) or (gauss, variance)
    # LW         : length of the window in terms of samples
    # SNR        : Signal-to-noise ratio
    # ret        : Threshold type either 'best' for the best threshold
    #	                         'visu' for the universal threshold
    #			      or 'facT' for a the given threshold

    # Seed setzen
    np.random.seed(Seed)

    # noise
    L      = len(signal)
    sigma  = np.sqrt(np.mean(signal ** 2) / SNR)
    noise  = np.random.normal(0, sigma, size=L)

    # Gabor Frame
    a, M, Lg, N, Ngood = gabimagepars(L, int(L/alpha), int(L/beta))
    if LW is None:
        LW = Lg
    if window[0] == 'bsplines':
        gs = gabwin(mySpline(window[1],LW,0)[-1], a, M, LW)[0]
    else:
        gs = gabwin({'name': ('gauss'), 'tfr': window[1]}, a, M, LW)[0]

    ga = gabdual(gs, a, M)

    # denoising
    c_noisy = dgtreal(signal + noise, ga, a, M)[0]


    if ret =='best':
        c_noisy_max = c_noisy.real.max()
        print 'c_noisy_max', c_noisy_max

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": ((0, c_noisy_max),)}
        BS = basinhopping(lambda y: error_gabor(signal, c_noisy, gs, a, M, 'soft', y)[0], 0, minimizer_kwargs=minimizer_kwargs, niter=100, stepsize=c_noisy_max/10)
        BH = basinhopping(lambda y: error_gabor(signal, c_noisy, gs, a, M, 'hard', y)[0], 0, minimizer_kwargs=minimizer_kwargs, niter=100, stepsize=c_noisy_max/10)
        # BS = minimize_scalar(lambda y: error_gabor(signal, c_noisy, gs, a, M, 'soft', y, c_noisy_max)[0], bracket = (0, c_noisy_max))
        # BH = minimize_scalar(lambda y: error_gabor(signal, c_noisy, gs, a, M, 'hard', y, c_noisy_max)[0], bracket = (0, c_noisy_max)) 
        res = BS.x, BH.x, BS.fun, BH.fun
    elif ret =='visu':
        print 'ga:', np.linalg.norm(ga),'gs:', np.linalg.norm(gs), M, N
        US  = sigma * np.linalg.norm(ga) * np.sqrt(2*np.log(M*N))
        ES  = error_gabor(signal, c_noisy, gs, a, M, 'soft', US)
        EH  = error_gabor(signal, c_noisy, gs, a, M, 'hard', US)
        res =  US, ES[0], EH[0]
    else:
        TH = facT
        ES = error_gabor(signal, c_noisy, gs, a, M, 'soft', TH)
        EH = error_gabor(signal, c_noisy, gs, a, M, 'hard', TH)
        res = TH, ES[0], EH[0]

    # plots
    if plot:
        plt.clf()
        plt.plot(signal, alpha=0.3)
        plt.plot(ES[-1], linestyle='--', alpha=2)



    return (Seed, sigma, alpha, beta, window, LW) + res




def denoise_gabor_real(signal, alpha, beta, window, LW=None, th_method='soft', plot=0,ret='visu',sigma=1,facT = 1):
    # signal denoising through shrinkage of gabor coefficients
    # alpha, beta: time and frequency step
    # window     : either (bsplines, level) or (gauss, variance)
    # LW         : length of the window in terms of samples
    # ret        : Threshold type either 'visu' for the universal threshold
    #			              or 'facT' for a the given threshold
    # sigma      : estimated noise standard deviation


    L      = len(signal)

    # Gabor Frame
    a, M, Lg, N, Ngood = gabimagepars(L, int(L/alpha), int(L/beta))
    if LW is None:
        LW = Lg
    if window[0] == 'bsplines':
        gs = gabwin(mySpline(window[1], LW, 0)[-1], a, M, LW)[0]
    else:
        gs = gabwin({'name': ('gauss'), 'tfr': window[1]}, a, M, LW)[0]

    ga = gabdual(gs, a, M)

    # denoising
    c_noisy = dgtreal(signal, ga, a, M)[0]

    if ret =='visu':
        US  = sigma * np.linalg.norm(ga) * np.sqrt(2*np.log(M*N))
        ES  = error_gabor(signal, c_noisy, gs, a, M, th_method, US)
    else:
        TH = facT
        ES = error_gabor(signal, c_noisy, gs, a, M, th_method, TH)

    # plots
    if plot:
        plt.clf()
        plt.plot( signal, alpha=0.3)
        plt.plot( ES[-1], linestyle='--', alpha=2)

    return ES[-1]
