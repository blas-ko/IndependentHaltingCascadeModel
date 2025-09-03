import numpy as np

#### DIFFUSION BOUNDARIES ####
def diffusion_boundary_pr(pa, ph, k):
    return 1/(k*pa*ph)

def diffusion_boundary_quantile_pr(pa, ph, k, q=0.5):
    return (1 - q**(1/k))/(pa*ph)

def diffusion_boundary_contour_pr(ph, k, n_pa=200):
    pa_vals = np.linspace(1e-4, 1-1e-4, n_pa)
    pr_vals = np.clip( diffusion_boundary_pr(pa_vals, ph, k), 0,1 )
    return pa_vals, pr_vals

def diffusion_boundary_quantile_contour_pr(ph, k, q=0.5, n_pa=200):
    pa_vals = np.linspace(1e-4, 1-1e-4, n_pa)
    pr_vals = np.clip( diffusion_boundary_quantile_pr(pa_vals, ph, k, q), 0,1 )
    return pa_vals, pr_vals

#### NO HALTING PROBABILITY CURVES ####
from scipy.optimize import brentq

def prob_no_halting(pa, pr, ph, k):
    """Probability that an active node leads to never halting."""
    f = lambda U: pa*(1-ph) + (1-pa)*(1-pr + pr*U)**k - U
    if pa == 0.0:  # at pa=0 both U=0 and U=1 are roots; physical solution is failure U=1
        return 1.0
    return brentq(f, 0.0, 1.0)

# For fixed pa, find pr with P_fail = tau, or return NaN if no solution
def success_pr_at_prob_tau(ph, k, pa, tau):
    """Returns p_r such that initial spreader (that always recommends) leads to never halting with probability tau"""
    def h(pr):
        U = prob_no_halting(pa, pr, ph, k)
        return (1 - pr + pr*U)**k - tau
    try:
        return brentq(h, 0.0, 1.0)
    except:
        return np.nan

def success_contour_at_prob_tau(ph, k, tau=0.5, n_pa=200):
    """Returns (pa,pr) curve such that prob. that initial spreader leads to never halting is tau."""
    pa_vals = np.linspace(0.0, 1.0, n_pa)
    pr_vals = np.array([success_pr_at_prob_tau(ph, k, pa, tau) for pa in pa_vals])
    return pa_vals, pr_vals