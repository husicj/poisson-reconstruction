#!/usr/bin/env pytho3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:31:21 2020

@author: nikolaj
"""

import numpy as np
import jax.numpy as jnp
import time

from src import deconv
from src import image_functions

class IterationData:
    def __init__(self, f, b, ff, coefficients):
        self.ff = ff

        self.coefficients = coefficients
        self.tempCoefficients = None

        self.f = f
        self.b = b
        self.f0 = f.sum()

        self.F = None
        self.H = None
        self.S = None
        self.s = None
        self.g = None
        self.G = None
        self.q = None
        self.Q = None

    def update_coefficients(self, c=None):
        if c is None:
            self.coefficients = self.tempCoefficients
        else:
            self.coefficients = c

    def update_tempCoefficients(self, c):
        self.tempCoefficients = c
        
    def update_F(self):
        self.F = self.ff.fft2(self.f)

    def update_f(self, Sk0, num_imgs, normalize=True):
        dF = (jnp.conj(self.S) * self.Q).sum(axis = 0)
        df = self.ff.ift2(dF) / (Sk0 * num_imgs)
        self.f *= df
        #normalize object estimate if background or regularizer are being used
        #also helps with instability
        if normalize:
            self.f = self.f * (self.f0 / self.f.sum())

    def update_for_line_search(self, zernikes, dim, defocusPhases):
        self.H = jnp.array(image_functions.get_H2(zernikes.zern, zernikes.inds, dim, defocusPhases, self.tempCoefficients))
        self.h = self.ff.ift(self.H)  
        self.s = jnp.abs(self.h)**2
        self.S = self.ff.fft2(self.s)
        self.G = self.F * self.S
        self.g = self.ff.ift2(self.G)+self.b

    def update_Q(self, imgs, lam, reg_option):
        self.q = imgs / (self.g - lam * deconv.get_reg(self.f, reg_option))
        self.Q = self.ff.fft2(self.q)

def iter_p(zernikes, imgs, defocusPhases, Sk0, c, ff, show = False, eps = 1e-3):  #perform poisson iterations
    start = time.time()

    nc = len(c)

    #optimization parameters
    MAX_ITER = 1000
    MIN_ITER = 100
    MAX_ITER_LINESEARCH = 10 #max number of iterations for the line search
    INITIAL_STEP_SIZE = 3e4
    SS_REDUCE = .3 #amount to reduce step size by (multiplicatively) in line search
    BACKGROUND_INTENSITY = 0
    LAM = 1e-2 #regularization strength
    REG_OPTION = 0 #regularization option

    #initialize fields
    ss = INITIAL_STEP_SIZE
    norm_g = 1+eps
    f = jnp.ones(imgs[0].shape)
    f *= imgs.mean()
    iterData = IterationData(f, BACKGROUND_INTENSITY, ff, c.copy())
    dc = jnp.zeros((nc))

    cost = np.zeros((MAX_ITER))
    sss = np.zeros((MAX_ITER))
    fs = np.zeros((MAX_ITER, ) + imgs[0].shape)
    fs[0] = f.copy()

    c_all = np.zeros((1, len(c)))
    c_all[0] = c.copy()

    num_imgs = len(imgs)
    dim = imgs[0].shape
    
    L0 = -jnp.inf
    L1 = 0
    
    #main loop
    n_iter = 0
    while True:
        #line search
        iterData.update_F()
        for i in range(MAX_ITER_LINESEARCH):
            iterData.update_tempCoefficients(iterData.coefficients - ss*dc)
            iterData.update_for_line_search(zernikes, dim, defocusPhases)
            #compute cost function
            L1 = (imgs * jnp.log(iterData.g) - iterData.g).mean()
            if L1 > L0:
                break
            else:
                ss *= SS_REDUCE
            
        L0 = L1
        iterData.update_coefficients()
        
        #update object estimation, using values computed on the last (and
        #therefore successful) iteration of line search
        iterData.update_Q(imgs, LAM, REG_OPTION)
        iterData.update_f(Sk0, num_imgs)
                
        #find new search direction dc        
        temp1 = jnp.conj(iterData.h) * ff.ift2(iterData.Q * jnp.conj(iterData.F))
        temp2 = jnp.imag(iterData.H * ff.ift(temp1)).sum(axis = 0)
        
        dc_integral = temp2[zernikes.inds] * zernikes.zern    
        dc = 2*dc_integral.sum(axis = 1)/(dim[0]*dim[1])
                               
        #Stopping conditions
        if n_iter>=MAX_ITER:
            break
        if n_iter>MIN_ITER:
            #terminate if total step size is small
            norm_g = jnp.linalg.norm(ss*dc)
            if norm_g < eps: break
            
        cost[n_iter] = L1
        sss[n_iter] = norm_g
        fs[n_iter] = f
        c_all = np.vstack((c_all, iterData.coefficients))
        
        n_iter +=1
        if show:
            image_functions.progress(f"{n_iter} iterations")

    end = time.time()
    if show: print(f"\nRuntime: {(end-start):.2f} seconds")
    
    cost = cost[:n_iter]
    sss = sss[:n_iter]
    fs = fs[:n_iter]
    
    return iterData.coefficients, c_all, cost, [sss, iterData.f, end-start, fs]

# TODO: use jax.numpy for image_functions.get_H2 and for the Fast_FFTs that are passed to this potentially as well
