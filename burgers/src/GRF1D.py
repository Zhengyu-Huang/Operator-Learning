# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:09:54 2020

@author: nickh

Written by:
Nicholas H. Nelsen
California Institute of Technology
Email: nnelsen@caltech.edu

A module file with class for 1D Gaussian Random Fields on [0,1].

Last updated: Jun. 18, 2020
"""

# Import default modules/packages
import numpy as np

# Custom imports to this file
from scipy.fftpack import idct, dst, ifft
from scipy.interpolate import interp1d



def dstnhn(x):
    '''
    Orthonormalized disrete sine transform type I (does NOT include zeros at boundaries in x), where dst(dst(x))=x
    Input:
        x: (n,) numpy array
    Output:
        output: (n,) numpy array
    '''
    return dst(x, 1, axis=0, norm='ortho')


def idctnhn(x):
    '''
    1D inverse discrete cosine transform Type 2 (Orthonormalized)
    Input:
        x: (N,) numpy array
    Output:
        _ : (N,) numpy array
    '''
    return idct(x, norm='ortho')


class GaussianRandomField1D:
    '''
    Return a sample of a Gaussian random field on [0,1] with: 
        -- mean function m = 0
        -- covariance operator C = (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with periodic, zero Dirichlet, or zero Neumann boundary conditions.
    Requires the functions: ``idctnhn, dstnhn, interp1d''
    '''

    def __init__(self, tau, alpha, bc=0):
        '''
        Initializes the class.
        Arguments:
            tau:        (float), inverse lengthscale for Gaussian measure covariance operator
            
            alpha:      (alpha), regularity of covariance operator
            
            bc:         (int), ``0'' for Neumann BCs or ``1'' for Dirichlet BCs or ``2'' for periodic

        Parameters:
            tau:        (float), inverse lengthscale for Gaussian measure covariance operator
            
            alpha:      (alpha), regularity of covariance operator
            
            bc:         (int), ``0'' for Neumann BCs or ``1'' for Dirichlet BCs or ``2'' for peridic

            bc_name:    (str), ``neumann'' for Neumann BCs or ``dirichlet'' for Dirichlet BCs or ``periodic'' for periodic BCs
        '''
        
        self.tau = tau
        self.alpha = alpha
        if bc == 0: # neumann
            self.bc = 0
            self.bc_name = 'neumann'
        elif bc==1: # dirichlet
            self.bc = 1
            self.bc_name = 'dirichlet'
        else: # periodic
            self.bc = 2
            self.bc_name = 'periodic'
            

    def draw(self, theta, get_coef=False):
        '''
        Draw a sample Gaussian Random Field on [0,1] with desired BCs
        Input:
            theta: (N,) numpy array of N(0,1) iid Gaussian random variables
            get_coef: (boolean) return field if false, return Fourier coeff if true
        Output:
            grf: (N,) numpy array, a GRF on the grid np.arange(0,1+1/(N-1),1/(N-1))
        '''
        
        # Length of random variables matrix in 1D KL expansion
        N = theta.shape[0]
        
        # Choose BCs
        if self.bc == 0: # neumann
            # Define the (square root of) the eigenvalues of the covariance operator
            K = np.arange(N)
            coef = (self.tau**(self.alpha - 1/2))*(np.pi**2*(K**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            # Construct the KL (discrete cosine transform) coefficients
            B = np.sqrt(N)*coef*theta # multiply by sqrt(N) to satisfy the DCT Type II definition
            B[0] = 0 # set k=0 constant mode to zero (to satisfy zero mean field)
            
            # Inverse (fast FFT-based) 2D discrete cosine transform
            grf_temp = idctnhn(B) # sums B*sqrt(2/N)*cos(k*pi*x) over all k1, k2 = 0, ..., N-1
            
            # Interpolate to physical grid containing the boundary of the domain [0,1]
            X1 = np.arange(1/(2*N),(2*N-1)/(2*N)+1/N, 1/N) # IDCT grid
            X2 = np.arange(0,1+1/(N-1),1/(N-1)) # physical domain grid
            func_interp = interp1d(X1, grf_temp, kind='cubic', fill_value='extrapolate')
            grf = func_interp(X2)
            
        elif self.bc == 1: # dirichlet
            # Define the (square root of) the eigenvalues of the covariance operator
            K = np.arange(1,N-1) # does not include first or last wavenumber
            coef = (self.tau**(self.alpha - 1/2))*(np.pi**2*(K**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            
            # Construct the KL (discrete sine transform) coefficients
            B = np.sqrt(N-1)*coef*theta[1:-1] # multiply by sqrt(N-1) to satisfy the DST Type I definition
            
            # Inverse (fast FFT--based) discrete sine transform
            U_noBC = dstnhn(B) # sums B*sqrt(2/(N-1))*sin(k*pi*x) over all k=1,...,N-2
            
            # Impose zero boundary data on the output
            grf = np.zeros(N)     
            grf[1:-1] = np.copy(U_noBC)
            
        else: # periodic, x[0] = x[-1]
            # Remove right boundary point for now
            N = N - 1 # must be even
            if np.mod(N,2) != 0:
                print('ERROR: N - 1 must be even.')
            
            # Define the (square root of) the eigenvalues of the covariance operator
            K = np.array([i for i in range(N//2)] + [0] + [ii for ii in range(-N//2 + 1,0)]) # fft ordered
            coef = (self.tau**(self.alpha - 1/2))*(4*np.pi**2*(K**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            
            # Construct the KL (discrete fourier transform) coefficients
            theta = theta[:-1]
            theta_re = theta[1:N//2] # k=1,..., N/2 - 1
            theta_im = theta[N//2 + 1:] # k=N/2 + 1,..., -1
            xi_pos = (theta_re - 1j*theta_im)/2
            xi_neg = np.flipud((theta_re + 1j*theta_im)/2)
            xi = np.concatenate(([0], xi_pos, [0], xi_neg)) # set k=N//2 to zero to mirror zero mean k=0
            B = np.sqrt(2)*N*coef*xi # multiply by N to satisfy the scipy.fft.ifft definition
            
            # Inverse FFT
            grftemp = np.real(ifft(B))
            grf = np.append(grftemp, grftemp[0])  # grf[0] = grf[-1]
        
        # Output desired object
        if get_coef:
            return B
        else:
            return grf
    
# %% Test

# =============================================================================
# from utilities.plot_suite import Plotter
# plotter = Plotter() # set plotter class
# from scipy.fft import fft, dct
# 
# K = 1 + 1024
# x = np.arange(0, 1 + 1/(K-1), 1/(K-1))
# tau = 7
# alpha = 2.5
# 
# np.random.seed(12345)
# theta = np.random.standard_normal(K)
# grf = GaussianRandomField1D(tau, alpha)
# g1 = grf.draw(theta)
# g1B = grf.draw(theta, True)
# print(np.allclose(g1B,dct(g1, 3, norm='ortho')))
# print(np.max(g1), np.min(g1))
# # gx = np.gradient(g, 1/(K), edge_order=2)
# 
# ts='Gaussian Random Field: Neumann BC'
# plotter.plot_oneD(222, x, g1, titlelab_str=ts, linestyle_str='k')
# # plotter.plot_oneD(2222, x, gx, linestyle_str='k')
# # print(gx[0], gx[-1])
# 
# grf2 = GaussianRandomField1D(tau, alpha, bc=1)
# g2 = grf2.draw(theta)
# g2B = grf2.draw(theta, True)
# print(np.allclose(g2B,dstnhn(g2[1:-1])))
# print(np.max(g2), np.min(g2))
# # print(np.allclose(g2[0],g2[-1]))
# 
# ts='Gaussian Random Field: Dirichlet BC'
# plotter.plot_oneD(223, x, g2, titlelab_str=ts, linestyle_str='k')
# 
# grf3 = GaussianRandomField1D(tau, alpha, bc=2)
# g3 = grf3.draw(theta)
# g3B = grf3.draw(theta, True)
# print(np.allclose(g3B,fft(g3[:-1])))
# print(np.max(g3), np.min(g3))
# # print(g3[0], g3[-1], g3[-2])
# print('The mean of periodic grf is:', g3.mean())
# # print(np.abs(np.imag(g3)).max())
# 
# ts='Gaussian Random Field: Periodic BC'
# plotter.plot_oneD(224, x, g3, titlelab_str=ts, linestyle_str='k')
# =============================================================================
