#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 05:26:42 2025

@author: nephilim
"""

import numpy as np
from numba import jit

def denoising_2D_TV(data):
    """
    Main user-facing function. Scales the input 'data', calls the ADMM-based 
    TV denoising routine, and rescales the output.
    """
    # Normalize input
    data_max = np.max(data)
    if data_max == 0:
        # Avoid division by zero if the data is all zeros
        return data
    data_norm = data / data_max
    
    # Dimensions of the input
    M, N = data_norm.shape

    # Hyper-parameters
    lamda = 0.02  # TV weight
    rho_ = 1.0    # ADMM penalty
    max_iter = 500
    tol = 1e-5

    # We pad the data by 1 pixel on each side (total M+2, N+2),
    # because your wrap-around differencing logic references neighbors [i+1], [i-1], etc.
    X = np.zeros((M+2, N+2))
    X[1:M+1, 1:N+1] = data_norm.copy()

    # Same for Y0, Zx, Zy, Ux, Uy
    Y0 = np.zeros_like(X)
    Y0[1:M+1, 1:N+1] = data_norm.copy()
    
    X_old = X.copy()           # For measuring convergence
    Zx    = np.zeros_like(X)
    Zy    = np.zeros_like(X)
    Ux    = np.zeros_like(X)
    Uy    = np.zeros_like(X)

    # Call the JIT-compiled core routine
    denoising_2D_TV_main(X, X_old, Zx, Zy, Ux, Uy, Y0, lamda, rho_, max_iter, tol)

    # Return the central region (un-padded) and rescale
    return X[1:M+1, 1:N+1] * data_max

@jit(nopython=True)
def compute_diff_norm(A, B):
    """
    Compute the Euclidian (L2) norm of (A - B) in no-Python mode,
    equivalent to np.linalg.norm(A - B).
    """
    s = 0.0
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            diff = A[i, j] - B[i, j]
            s += diff*diff
    return np.sqrt(s)

@jit(nopython=True)
def denoising_2D_TV_main(X, X_old, Zx, Zy, Ux, Uy, Y0, lamda, rho_, max_iter, tol):
    """
    JIT-compiled core ADMM routine for 2D TV denoising with wrap-around boundaries.
    
    X:      Current solution (padded by 1 pixel border)
    X_old:  Copy of X from previous iteration
    Zx,Zy:  Dual variables for TV in the x and y directions
    Ux,Uy:  Dual variables (Lagrange multipliers) for ADMM
    Y0:     Original (padded) data
    lamda:  TV weight
    rho_:   ADMM penalty parameter
    max_iter: Maximum number of iterations
    tol:    Convergence tolerance for ||X - X_old||
    """

    # For convenience
    M = X.shape[0] - 2  # Because of +2 padding
    N = X.shape[1] - 2

    # Pre-allocate some temporary arrays so we don't keep creating them
    D    = np.zeros_like(X)  # scratch array for difference ops
    DxtZ = np.zeros_like(X)
    DytZ = np.zeros_like(X)
    DxtU = np.zeros_like(X)
    DytU = np.zeros_like(X)
    RHS  = np.zeros_like(X)
    Dx_X = np.zeros_like(X)
    Dy_X = np.zeros_like(X)

    # We do not rely on "while" loop for tolerance because we want to limit
    # overhead of computing the norm on every iteration. We'll do a "for" loop
    # and check norm every 10 steps.
    for k in range(max_iter):
        
        # 1) Copy current X to X_old (for convergence check)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_old[i, j] = X[i, j]
        
        # 2) Compute Dxt_Zx and Dyt_Zy (the backward divergences of Zx, Zy)
        #    with wrap-around differences.
        #    Dxt_Zx = Zx(i,j) - Zx(i,j+1)  (plus wrap at last column)
        
        # Dimensions (M+2) x (N+2)
        MM = M + 2
        NN = N + 2

        # --- Dxt_Zx ---
        D[:] = 0.0
        for i in range(MM):
            for j in range(NN - 1):
                D[i, j] = Zx[i, j] - Zx[i, j+1]
            # wrap-around
            D[i, NN-1] = Zx[i, NN-1] - Zx[i, 0]
        for i in range(MM):
            for j in range(NN):
                DxtZ[i, j] = D[i, j]

        # --- Dyt_Zy ---
        D[:] = 0.0
        for i in range(MM - 1):
            for j in range(NN):
                D[i, j] = Zy[i, j] - Zy[i+1, j]
        # wrap-around
        for j in range(NN):
            D[MM-1, j] = Zy[MM-1, j] - Zy[0, j]
        for i in range(MM):
            for j in range(NN):
                DytZ[i, j] = D[i, j]

        # 3) Compute Dxt_Ux, Dyt_Uy similarly
        # --- Dxt_Ux ---
        D[:] = 0.0
        for i in range(MM):
            for j in range(NN - 1):
                D[i, j] = Ux[i, j] - Ux[i, j+1]
            D[i, NN-1] = Ux[i, NN-1] - Ux[i, 0]
        for i in range(MM):
            for j in range(NN):
                DxtU[i, j] = D[i, j]

        # --- Dyt_Uy ---
        D[:] = 0.0
        for i in range(MM - 1):
            for j in range(NN):
                D[i, j] = Uy[i, j] - Uy[i+1, j]
        for j in range(NN):
            D[MM-1, j] = Uy[MM-1, j] - Uy[0, j]
        for i in range(MM):
            for j in range(NN):
                DytU[i, j] = D[i, j]

        # 4) RHS = Y0 + lamda*rho_*(DxtZ + DytZ) - lamda*(DxtU + DytU)
        for i in range(MM):
            for j in range(NN):
                RHS[i, j] = (
                    Y0[i, j]
                    + lamda * rho_ * (DxtZ[i, j] + DytZ[i, j])
                    - lamda * (DxtU[i, j] + DytU[i, j])
                )

        # 5) Update X by the standard formula
        #    X(i,j) = [ (X_old[i+1,j] + X_old[i-1,j] + X_old[i,j+1] + X_old[i,j-1]) * lamda*rho_ 
        #              + RHS[i,j] ] / [ 1 + 4*lamda*rho_ ]
        for i in range(1, M+1):
            for j in range(1, N+1):
                val_neighbors = (
                    X_old[i+1, j] + X_old[i-1, j]
                    + X_old[i, j+1] + X_old[i, j-1]
                )
                numerator = val_neighbors * lamda * rho_ + RHS[i, j]
                X[i, j] = numerator / (1.0 + 4.0 * lamda * rho_)

        # 6) Compute Dx_X = X(i,j) - X(i,j-1) with wrap-around in j
        Dx_X[:] = 0.0
        for i in range(MM):
            for j in range(1, NN):
                Dx_X[i, j] = X[i, j] - X[i, j-1]
            # wrap-around in x-direction
            Dx_X[i, 0] = X[i, 0] - X[i, NN-1]

        #    Compute Dy_X = X(i,j) - X(i-1,j) with wrap-around in i
        Dy_X[:] = 0.0
        for i in range(1, MM):
            for j in range(NN):
                Dy_X[i, j] = X[i, j] - X[i-1, j]
        for j in range(NN):
            Dy_X[0, j] = X[0, j] - X[MM-1, j]

        # 7) Update Z with soft-thresholding:
        #       Zx = max(|Tx|-1/rho_, 0)*sign(Tx)
        #       Tx = Ux/rho_ + Dx_X
        #    similarly for Zy
        for i in range(MM):
            for j in range(NN):
                Tx = Ux[i, j] / rho_ + Dx_X[i, j]
                Ty = Uy[i, j] / rho_ + Dy_X[i, j]

                absTx = abs(Tx)
                absTy = abs(Ty)

                # Soft-threshold
                Zx[i, j] = 0.0
                if absTx > 1.0 / rho_:
                    Zx[i, j] = (absTx - 1.0 / rho_) * np.sign(Tx)

                Zy[i, j] = 0.0
                if absTy > 1.0 / rho_:
                    Zy[i, j] = (absTy - 1.0 / rho_) * np.sign(Ty)

        # 8) Update U (dual variables):
        #     Ux <- Ux + (Dx_X - Zx)
        #     Uy <- Uy + (Dy_X - Zy)
        for i in range(MM):
            for j in range(NN):
                Ux[i, j] += (Dx_X[i, j] - Zx[i, j])
                Uy[i, j] += (Dy_X[i, j] - Zy[i, j])

        # 9) Check convergence every 10 iterations to reduce overhead
        if (k % 10) == 0:
            diff = compute_diff_norm(X, X_old)
            if diff < tol:
                # Converged early
                break

    return X

class Regularization(object):
    def __init__(self,method,alpha=1e-2,rho=1e-1,tau=1e-3,max_iter=50):
        self.method=method
        self.alpha = alpha   # TV weight
        self.rho = rho   # ADMM penalty parameter
        self.tau = tau   # Step size
        self.max_iter = max_iter
    
    def _forward_diff_x(self,m):
        """Compute forward difference in the x-direction."""
        return np.roll(m, -1, axis=1) - m

    def _forward_diff_z(self,m):
        """Compute forward difference in the z-direction."""
        return np.roll(m, -1, axis=0) - m

    def _laplacian(self, m):
        """
        Compute a 2D Laplacian using a 5-point stencil with zero Neumann boundary conditions.
        """
        # Pad the array with its edge values (replicate border values)
        m_pad = np.pad(m, pad_width=1, mode='edge')
        # Compute the Laplacian on the interior of the padded array
        lap = (m_pad[:-2, 1:-1] + m_pad[2:, 1:-1] +
               m_pad[1:-1, :-2] + m_pad[1:-1, 2:] -
               4 * m)
        return lap
    
    def penalty_value_gradient(self,m):
        if self.method=='mtv':
            m_solution=denoising_2D_TV(m)
            m_rhs=m-m_solution
            penalty_mtv=0.5*np.linalg.norm(m_rhs.flatten(),2)**2
            gradient_mtv=2*m_rhs
            return penalty_mtv,gradient_mtv

if __name__=='__main__':
    # Set ADMM parameters
    # alpha = 1e-2  # Tikhonov regularization weight
    # rho = 1e-1    # ADMM penalty parameter
    # tau = 1e-3    # Step size for m-update
    # max_iter = 50
    alpha = 0.02  # TV weight
    rho = 1.0    # ADMM penalty
    tau = 1e-3 
    max_iter = 500
    tol = 1e-5
    method='tikhonov'
    # method='tv'
    # method='mtv'
    data=np.load('36_imodel.npy').reshape((100,-1))
    # data=np.load('FieldIinit.npy')
    penalty,grad=Regularization(method,alpha,rho,tau,max_iter).penalty_value_gradient(data[10:-10,10:-10])
