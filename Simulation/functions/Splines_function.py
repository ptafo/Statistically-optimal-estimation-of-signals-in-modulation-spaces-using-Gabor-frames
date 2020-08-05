from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
from ltfatpy.sigproc.normalize import normalize




def mySpline(n, L = 100, plot=1):

    #return normalize splines of degree n and length L
    n = int(n)
    b = BSpline.basis_element(range(n+1), 0)
    x = np.linspace(0, n, L)
    y = np.fft.fftshift(normalize(b(x))[0])


    if plot:
        plt.plot(range(L), y, 'g', lw=3)

    return b,y



