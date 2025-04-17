#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:46:58 2018

@author: nephilim
"""
import numpy as np
import shutil
import os
import cmath
from config_parameters import config
from fwi_parameters import fwi_config

class Optimization(object):
    def __init__(self,fh,iepsilon,tol,maxiter):
        super().__init__()
        self.fh=fh
        self.data=iepsilon.flatten()
        self.tol=tol
        self.maxiter=maxiter
        self.max_ls = 5
        self.M = 5
        self.c1 = 1e-4
        self.c2 = 0.9
        self.ls_interp =2
        
    def optimization(self):        
        # --- 2. Create or recreate the output directory ---
        dir_path = f"./{fwi_config.fwi_freq}Hz_imodel_file"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        # --- 3. Initialization ---
        x = self.data.copy()
        n = len(x)
        d = np.zeros(n)
        alpha0 = 1.0

        # Evaluate function & gradient at the starting point
        f, g = self.fh(x.reshape((config.xl + 20, -1)))
        f0 = f  # keep track of initial function value
        fevals = 1

        print(f"{'iter':>5s}, {'eval':>6s}, {'step length':>15s}, {'function value':>15s}, {'||g(x)||_2':>15s}")
        print(f"{0:5d}, {fevals:6d}, {alpha0:15.5e}, {f:15.5e}, {np.linalg.norm(g, 2):15.5e}")

        # Store iteration info: [iter_index, fevals, alpha, f, ||g||]
        info = [[0, fevals, alpha0, f, np.linalg.norm(g, 2)]]

        # --- 4. Main iteration loop ---
        for iter_index in range(self.maxiter):
            # Initialize or shift L-BFGS memory S, Y
            if iter_index == 0:
                S = np.zeros((n, self.M))
                Y = np.zeros((n, self.M))
            else:
                # Shift (discard old) and update with the new alpha*d, g-g_old
                S = np.hstack((S[:, 1:], alpha * d[:, np.newaxis]))
                Y = np.hstack((Y[:, 1:], (g - g_old)[:, np.newaxis]))

            # Quasi-Newton direction via approximate Hessian
            d = self.__B(-g, S, Y, np.array([]))
            p = -np.dot(d, g) / np.dot(g, g)  # used to detect negative curvature
            if p <= 0:
                # Reset memory if negative curvature is detected
                S = np.zeros((n, self.M))
                Y = np.zeros((n, self.M))

            g_old = g.copy()
            gtd = np.dot(g, d)

            if iter_index == 0:
                alpha0 = -f / gtd / 10.0
            else:
                alpha0 = 1.0
                if alpha0 <= 0:
                    alpha0 = 1.0

            f_old = f
            gtd_old = gtd

            # --- 5. Line search (Wolfe) ---
            alpha, f, g, ls_iter = self.__WolfeLineSearch(x, alpha0, d, f, g, gtd)
            fevals += ls_iter

            # Update x
            x = x + alpha * d
            x = self.__counts(x)  # Some post-processing of x (clamping, etc.)

            # Save models and info
            np.save(f"{dir_path}/{iter_index}_imodel.npy", x)
            info.append([iter_index + 1, fevals, alpha, f, np.linalg.norm(g, 2)])
            np.save(f"{dir_path}/{iter_index}_info.npy", info)

            print(f"{iter_index + 1:5d}, {fevals:6d}, "f"{alpha:15.5e}, {f:15.5e}, {np.linalg.norm(g, 2):15.5e}")

            # --- 6. Check stopping conditions ---
            if f / f0 < self.tol:
                print("Function value is below tolerance.")
                break
            if alpha == 0.0:
                print("Line search returned alpha = 0.")
                break
            if fevals >= 1000:  # or use another max_fevals if you prefer
                print("Reached maximum number of function evaluations.")
                break

        # If we exit normally (or from break), return the final solution and info
        return x, info
    
    
    def __B(self, x, S, Y, H0):
        """
        Apply the L-BFGS two-loop recursion to vector x using stored S and Y matrices.
        
        Parameters
        ----------
        x : np.ndarray
            Vector to which the approximate Hessian is applied.
        S : np.ndarray
            Matrix whose columns are the step vectors (x_{k+1} - x_k).
        Y : np.ndarray
            Matrix whose columns are the difference of gradient vectors (g_{k+1} - g_k).
        H0 : np.ndarray
            Initial guess for the diagonal of the Hessian approximation. If empty, it is computed
            from the last columns of S and Y.

        Returns
        -------
        np.ndarray
        The result of multiplying the (approximate) Hessian by x.
        """
        # Identify and keep only non-zero columns in S (and corresponding in Y)
        nonzero_cols = np.nonzero(np.sum(np.abs(S), axis=0))[0]
        S = S[:, nonzero_cols]
        Y = Y[:, nonzero_cols]
        M = S.shape[1]
        n = len(x)
        
        # If H0 is empty and we have at least one column, compute an initial diagonal from the last (S, Y) pair
        if (H0.size == 0) and (M > 0):
            denom = np.dot(S[:, -1], Y[:, -1])
            H0_val = (np.linalg.norm(Y[:, -1], 2) ** 2) / denom
            H0 = H0_val * np.ones(n)
        else:
            # Otherwise, default to an identity-like diagonal if not provided
            H0 = np.ones(n)

        alpha = np.zeros(M)
        rho = np.zeros(M)
        for k in range(M):
            rho[k] = 1.0 / np.dot(Y[:, k], S[:, k])

        # First loop: going backwards through stored vectors
        q = x.copy()
        for k in reversed(range(M)):
            alpha[k] = rho[k] * np.dot(S[:, k], q)
            q -= alpha[k] * Y[:, k]

        # Multiply by initial Hessian diagonal
        z = q / H0

        # Second loop: going forward
        for k in range(M):
            beta = rho[k] * np.dot(Y[:, k], z)
            z += (alpha[k] - beta) * S[:, k]

        return z
    
    def __WolfeLineSearch(self, x, t, d, f, g, gtd):
        """
        Perform a Wolfe line search to find a step size that satisfies 
        the Armijo and curvature conditions.

        Parameters
        ----------
        x : np.ndarray
            Current point (vector).
        t : float
            Initial step size (alpha).
        d : np.ndarray
            Descent direction.
        f : float
            Current function value at x.
        g : np.ndarray
            Current gradient at x.
        gtd : float
            Dot product of g and d (g^T d).

        Returns
        -------
        t : float
            The chosen step size.
        f_new : float
            The function value at the new point x + t * d.
        g_new : np.ndarray
            The gradient at the new point x + t * d.
        ls_iter : int
            Total number of function evaluations used by the line search.
        """
        # Evaluate at initial step
        x_new = x + t * d
        x_new = self.__counts(x_new)  # A possible projection or adjustment
        f_new, g_new = self.fh(x_new.reshape((config.xl + 20, -1)))  # 'para' may be external
        ls_iter = 1  # Number of function evaluations
        gtd_new = np.dot(g_new, d)

        # Store info for possible bracket
        ls_total = 0  # Iteration counter for the line search
        t_prev = 0.0
        f_prev = f
        g_prev = g
        gtd_prev = gtd

        done = False
        bracket = None
        bracket_fval = None
        bracket_gval = None

        # -- Phase 1: Expand or adjust step until bracket found or conditions are satisfied --
        while ls_total < self.max_ls:

            # 1. Check Armijo or if we've grown the function after the first iteration
            if (f_new > f + self.c1 * t * gtd) or ((ls_total > 0) and (f_new >= f_prev)):
                # We have a bracket between t_prev and t
                bracket = np.array([t_prev, t])
                bracket_fval = np.array([f_prev, f_new])
                bracket_gval = np.column_stack((g_prev, g_new))
                break

            # 2. Check strong (or standard) Wolfe curvature condition
            elif abs(gtd_new) <= -self.c2 * gtd:
                bracket = t
                bracket_fval = f_new
                bracket_gval = g_new
                done = True
                break

            # 3. If derivative sign changes, we also have a bracket
            elif gtd_new >= 0.0:
                bracket = np.array([t_prev, t])
                bracket_fval = np.array([f_prev, f_new])
                bracket_gval = np.column_stack((g_prev, g_new))
                break

            # If none of the above, update step size (t) by extrapolation/interpolation
            temp = t_prev
            t_prev = t
            min_step = t + 0.01 * (t - temp)
            max_step = t * 10.0

            if self.ls_interp <= 1:
                # If interpolation mode <= 1, just pick the max step
                t = max_step
            elif self.ls_interp == 2:
                # Cubic extrapolation
                args = np.array([[temp, f_prev, gtd_prev],[t, f_new, gtd_new]])
                t = self.__polyinterp(args, min_step, max_step)
                print(f"Line Search Cubic Extrapolation Iter {ls_total}, alpha={t}")

            # Evaluate at the new step
            f_prev = f_new
            g_prev = g_new
            gtd_prev = gtd_new

            x_new = x + t * d
            x_new = self.__counts(x_new)
            f_new, g_new = self.fh(x_new.reshape((config.xl + 20, -1)))
            ls_iter += 1
            gtd_new = np.dot(g_new, d)
            ls_total += 1

        # If no bracket is found by max_ls, create a fallback bracket
        if ls_total == self.max_ls and bracket is None:
            bracket = np.array([0.0, t])
            bracket_fval = np.array([f, f_new])
            bracket_gval = np.column_stack((g_prev, g_new))

        # -- Phase 2: Use bracket to refine step (if not done) --
        insuf_progress = False
        while (not done) and (ls_total < self.max_ls):
            if isinstance(bracket, np.ndarray):
                # Identify which side of bracket is lower
                lo_pos = np.argmin(bracket_fval)
                hi_pos = 1 - lo_pos  # the other index
                f_lo = bracket_fval[lo_pos]

                # Interpolate within bracket
                if self.ls_interp <= 1:
                    t = np.mean(bracket)  # simple midpoint
                else:
                    # e.g., gradient-based cubic interpolation
                    args = np.array([[bracket[0], bracket_fval[0], np.dot(bracket_gval[:, 0], d)],[bracket[1], bracket_fval[1], np.dot(bracket_gval[:, 1], d)]])
                    t = self.__polyinterp(args)
                    print(f"Line Search Grad-Cubic Interp Iter {ls_total}, alpha={t}")

                # Check if t is too close to bracket boundaries
                bracket_range = bracket.max() - bracket.min()
                dist_to_bracket = min(t - bracket.min(), bracket.max() - t)
                if dist_to_bracket / bracket_range < 0.1:
                    print("Interpolation close to boundary.")
                    if insuf_progress or t >= bracket.max() or t <= bracket.min():
                        # Evaluate 0.1 away from boundary
                        if abs(t - bracket.max()) < abs(t - bracket.min()):
                            t = bracket.max() - 0.1 * bracket_range
                        else:
                            t = bracket.min() + 0.1 * bracket_range
                        print(f"Evaluating at 0.1 away from boundary, alpha={t}")
                        insuf_progress = False
                    else:
                        insuf_progress = True
                else:
                    insuf_progress = False

                # Evaluate at the new t
                x_new = x + t * d
                x_new = self.__counts(x_new)
                f_new, g_new = self.fh(x_new.reshape((config.xl + 20, -1)))
                ls_iter += 1
                gtd_new = np.dot(g_new, d)
                ls_total += 1

                # Check Armijo condition
                armijo_ok = (f_new < f + self.c1 * t * gtd)
                if (not armijo_ok) or (f_new >= f_lo):
                    # Shrink bracket from the high side
                    bracket[hi_pos] = t
                    bracket_fval[hi_pos] = f_new
                    bracket_gval[:, hi_pos] = g_new
                else:
                    # Curvature condition check
                    if abs(gtd_new) <= -self.c2 * gtd:
                        done = True
                    elif gtd_new * (bracket[hi_pos] - bracket[lo_pos]) >= 0.0:
                        # Switch bracket
                        bracket[hi_pos] = bracket[lo_pos]
                        bracket_fval[hi_pos] = bracket_fval[lo_pos]
                        bracket_gval[:, hi_pos] = bracket_gval[:, lo_pos]

                    # Move the low side up
                    bracket[lo_pos] = t
                    bracket_fval[lo_pos] = f_new
                    bracket_gval[:, lo_pos] = g_new

            else:
                # Bracket is just a scalar => done
                done = True

        if ls_total == self.max_ls:
            print("Line Search exceeded maximum iterations.")

        # -- Final step: choose the best bracket endpoint if not already chosen --
        if isinstance(bracket, np.ndarray):
            lo_pos = np.argmin(bracket_fval)
            t = bracket[lo_pos]
            f_new = bracket_fval[lo_pos]
            g_new = bracket_gval[:, lo_pos]
        else:
            # bracket is a scalar (t), so no further adjustment needed
            f_new = bracket_fval
            g_new = bracket_gval

        return t, f_new, g_new, ls_iter
    

    def __polyinterp(self, points, *vargs):
        """
        Perform a polynomial interpolation (with optional bounds) based on two points.
    
        Parameters
        ----------
        points : np.ndarray (shape: (2, 3))
            Each row must contain [x, f(x), f'(x)] for interpolation.
            - points[0, :] = [x0, f0, fprime0]
            - points[1, :] = [x1, f1, fprime1]
        *vargs : optional
            - vargs[0] (xminBound): lower bound for the interpolation
            - vargs[1] (xmaxBound): upper bound for the interpolation
    
        Returns
        -------
        float
            The chosen interpolation point (clamped to [xminBound, xmaxBound] if given).
        """
        # Extract minimum/maximum x among the provided points
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        print(x_min, x_max)  # Debug print; remove if not needed

        # Set interpolation bounds
        if len(vargs) < 1:
            xmin_bound = x_min
        else:
            xmin_bound = vargs[0]
        if len(vargs) < 2:
            xmax_bound = x_max
        else:
            xmax_bound = vargs[1]

        # Identify the row (index) in 'points' that has the smaller x-value
        idx_min = np.argmin(points[:, 0])
        idx_other = 1 - idx_min  # The other index (since we expect 2 points)

        # If the two x-values are the same, return the midpoint of the bounding interval
        if (points[idx_min, 0] - points[idx_other, 0]) == 0:
            return (xmax_bound + xmin_bound) / 2.0

        # Compute interpolation coefficients
        d1 = (points[idx_min, 2] + points[idx_other, 2] - 3.0 * (points[idx_min, 1] - points[idx_other, 1]) / (points[idx_min, 0] - points[idx_other, 0]))
        d2 = cmath.sqrt(d1 ** 2 - points[idx_min, 2] * points[idx_other, 2])

        # If d2 is real, compute the new t by cubic interpolation formula
        if d2.imag == 0.0:
            numerator = points[idx_other, 2] + d2.real - d1
            denominator = points[idx_other, 2] - points[idx_min, 2] + 2.0 * d2.real
            t = points[idx_other, 0] - (points[idx_other, 0] - points[idx_min, 0]) * (numerator / denominator)

            # Clamp t between xmin_bound and xmax_bound
            t_clamped = max(xmin_bound, min(xmax_bound, t))
            return t_clamped
        else:
            # If d2 is complex, fall back to midpoint of the bounding interval
            return (xmax_bound + xmin_bound) / 2.0


    def __counts(self, x):
        """
        Clamp the values in the input array to the range [1, 81].
    
        Parameters
        ----------
        x : np.ndarray
            The array to be clamped.

        Returns
        -------
        np.ndarray
            A new array with values clamped between 1 and 81.
        """
        return np.clip(x, 1, 81)
