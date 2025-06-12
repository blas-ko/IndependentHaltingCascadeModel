import numpy as np
from fractions import Fraction
from scipy.stats import poisson
from scipy.stats import binom
from scipy.special import comb
from functools import cache
# from tqdm import tqdm

# @cache
def prob_successful_hire( N, M, pr, λ, n_ν, approx=True):
    """Computes probability that the oracle gets a successful hire.

    Args:
        N (int): Number of users in full population
        M (int): Number of users reachable by the oracle
        pr (int): Probability of successful recommendation
        n_ν (int): Specificity of the vacancy requirements
        approx (bool, optional): Whether to approximate the probability by the most common values. Defaults to True.

    Returns:
        float: probability that the oracle gets a successful hire for the given inputs.
    """    
    
    # Compute the probability to sample a hirable candidate from the given skills model.
    p_λs, ps_nmax  = pλ_pdf(λ, n_ν, N)
    p_λ = np.dot( p_λs, ps_nmax )
                        
    if approx:        
        L_expected = N*p_λ # Comes from pH being a binomial distribution
        L_std = np.sqrt( N*p_λ*(1 - p_λ) ) # Comes from pH being a binomial distribution
        nstd = 3
        L_lower = max(0, int(round(L_expected - nstd*L_std))) 
        L_upper = min(N, int(round(L_expected + nstd*L_std)))
        L_upper = max(1, L_upper) # If L = 1, the range is a single point, so make it at least 2
        L_bounds = range(L_lower, L_upper+1)
    else:
        L_bounds = range(N+1)

    p_XHK = 0
    for i,L in enumerate(L_bounds):
        
        if approx:
            ## OBS: This is right only for the C(L,k) term, but it doesn't dominate in the product C(L,k) C(N-L, M-k) present in p_K.
            # k_expected = L/2 # Comes from binomial coefficient having mean L/2
            # k_std = np.sqrt(L)/2 # binomial coefficient is equivalent to binomial dist with p = 0.5
            # k_lower = max(0, int(np.floor( k_expected - 2.5*k_std ))) 
            # k_upper = min(L, int(np.ceil( k_expected + 2.5*k_std )))

            # This seems to be the resulting peak in the p_K distribution, centered on \rho * L somehow
            # TODO: Check the theory behind
            r = M/N
            k_expected = r*L
            k_std = np.sqrt(r*L/2)
            nstd = 3
            k_lower = max(0, int(np.floor( k_expected - nstd*k_std ))) 
            k_upper = min(L, int(np.ceil( k_expected + nstd*k_std )))
            k_bounds = range(k_lower, k_upper+1) 
        else:
            k_bounds = range(L + 1)

        p_XK = 0.0
        for k in k_bounds:
            p_XK += pX(pr, k) * pK(k, N, M, L)

        p_XHK += p_XK * pH( L, N, p_λ )

    # p_XHK = sum( [ np.sum([ pX(pr, k) * pK(k, N, M, L) for k in range(L+1)])*pH( L, N, p_λ ) for L in L_bounds ] )        
    return p_XHK

@cache
def pX( pr, k ):
    """Probability of getting a successes from `k` trials with prob `pr` each.

    Args:
        pr (float): Success probability
        k (int): Number of trials

    Returns:
        _type_: _description_
    """    
    return 1 - ( 1 - pr )**k

@cache
def pK(k,N,M,L):
    """Probability that the intersection of M and L random elements of a set of N elements is k.

    Args:
        k (int): Desired size of the intersection of sets of size M and L
        N (int): Size of universal set
        M (int): Size of one subset
        L (int): Size of other subset 

    Returns:
        float: probability
    """
    # TODO: Investigate a faster way to compute this. This is the main bottleneck

    if k == L:
        return choose(M, L) / choose(N,L)
    elif M == 0:
        return 0
    elif M == N:
        if N - M > L - k:
            return 1
        else:
            return 0
    else:
        ways_choosing_k = choose(L,k)
        ways_choosing_remaining = choose( N-L, M-k )
        total_combinations = choose( N,M )
        return ways_choosing_k * ( ways_choosing_remaining / total_combinations )

@cache
def pH( L, N, p_λ, ):
    """Probability of having L hirable elements from a set of N with probability of hirable  p_λ.

    Args:
        L (int): Desired number of hirable elements.
        N (int): Size of universal set.
        p_λ (float): probability of sampling a hirable element.

    Returns:
        float: probability
    """     
    return binom.pmf(L, N, p_λ)

@cache
def pλ_pdf( λ, n_ν, N ):
    """Returns probability dist of filling a skillset of n_ν requirements from a population of N users
    sampling their n_skills ~ Poisson(λ) and nmax ~ P( max(n_skills) = n_m ).

    Returns:
        list[float]: probability dist of p_λ for different n_max
        list[float]: probability dist of n_max
    """
        
    # Compute skill inclusion probability as the weighted sum of drawing each possible number of skills from a poisson
    # TODO: Make this range more principled based on λ, n_ν, and N
    n_max_range = range(5, 25) # for N=5000, most mass is between 10 and 12 and changes very slowly with N.
    skill_inclusion_probs = np.zeros( len(n_max_range) )
    nmax_probs = np.zeros( len(n_max_range) )

    # Compute the maximum number of skills based on the confidence interval of the number of nodes
    for (i, nmax) in enumerate( n_max_range ):

        if nmax >= n_ν:
            for l in range(nmax+1):
                skill_inclusion_probs[i] += poisson.pmf(l, λ) * comb(l, n_ν)
                # skill_inclusion_probs[i] += poisson.pmf(l, λ) * ( comb(nmax - n_ν, nmax - l) / comb(nmax, l) )
                # skill_inclusion_probs[i] += poisson.pmf(l, λ) * ( comb(n_ν, l) * comb(n, l) comb(nmax - n_ν, nmax - l) ) / comb(nmax, n_ν)            
            
            # Normalize by the number of counts
            skill_inclusion_probs[i] /= comb(nmax, n_ν)
            # Add nmax prob
            nmax_probs[i] =  nmax_pdf(nmax, λ, N)

        else:            
            skill_inclusion_probs[i] = 0.0
            nmax_probs[i] = nmax_pdf(nmax, λ, N)

    # Return highest probabilities so that 98% of the probability mass is contained
    prob_indexes = highest_probabilities_indices(nmax_probs)

    return skill_inclusion_probs[prob_indexes], nmax_probs[prob_indexes]

## helpers
def highest_probabilities_indices(prob_dist, mass_threshold=0.99):
    prob_dist = np.array(prob_dist)
    sorted_indices = np.argsort(prob_dist)[::-1]  # Sort indices based on probabilities in descending order
    
    cumulative_sum = 0.0
    selected_indices = []
    for i, prob in enumerate(prob_dist[sorted_indices]):
        cumulative_sum += prob
        selected_indices.append(sorted_indices[i])
        if cumulative_sum >= mass_threshold:
            break
    
    return selected_indices

# Weighting of p_lambda: exact probability distribution of n_max according to extreme value theory
def nmax_cdf(n_max, λ, n_samples):
    return poisson.cdf(n_max, λ)**n_samples

# This works because the floor function quantizes the Poisson CDF
def nmax_pdf( n_max, λ, n_samples ):
    return nmax_cdf(n_max, λ, n_samples) - nmax_cdf(n_max - 1, λ, n_samples)

# reduce to lowest terms along the way 
# From https://stackoverflow.com/questions/39412892/handling-big-numbers-in-python-3-for-calculating-combinations
@cache
def choose(n,k):

    # Modification to treat edge cases:
    if k > n:
        return 0
    elif n < 0:
        return 0
    elif k < 0:
        return 0
    else:
    
        if k > n//2: k = n - k
        p = Fraction(1)
        for i in range(1,k+1):
            p *= Fraction(n - i + 1, i)
        return int(p)
    

###### APPROXIMATIONS 
def prob_successful_hire_approx( N, M, pr, λ, n_ν, approx=True):
    """Computes probability that the oracle gets a successful hire.

    Args:
        N (int): Number of users in full population
        M (int): Number of users reachable by the oracle
        pr (int): Probability of successful recommendation
        n_ν (int): Specificity of the vacancy requirements
        approx (bool, optional): Whether to approximate the probability by the most common values. Defaults to True.

    Returns:
        float: probability that the oracle gets a successful hire for the given inputs.
    """    
    
    # Compute the probability to sample a hirable candidate from the given skills model.
    p_λs, ps_nmax  = pλ_pdf(λ, n_ν, N)
    p_λ = np.dot( p_λs, ps_nmax )
                        
    if approx:        
        L_expected = N*p_λ # Comes from pH being a binomial distribution
        L_std = np.sqrt( N*p_λ*(1 - p_λ) ) # Comes from pH being a binomial distribution
        nstd = 2.5
        L_lower = max(0, int(round(L_expected - nstd*L_std))) 
        L_upper = min(N, int(round(L_expected + nstd*L_std)))
        L_upper = max(1, L_upper) # If L = 1, the range is a single point, so make it at least 2
        L_bounds = range(L_lower, L_upper+1)
    else:
        L_bounds = range(N+1)

    p_XHK = 0
    for i,L in enumerate(L_bounds):
        
        if approx:
            ## OBS: This is right only for the C(L,k) term, but it doesn't dominate in the product C(L,k) C(N-L, M-k) present in p_K.
            # k_expected = L/2 # Comes from binomial coefficient having mean L/2
            # k_std = np.sqrt(L)/2 # binomial coefficient is equivalent to binomial dist with p = 0.5
            # k_lower = max(0, int(np.floor( k_expected - 2.5*k_std ))) 
            # k_upper = min(L, int(np.ceil( k_expected + 2.5*k_std )))

            # This seems to be the resulting peak in the p_K distribution, centered on \rho * L somehow
            # TODO: Check the theory behind
            r = M/N
            k_expected = r*L
            k_std = np.sqrt(r*L/2)
            nstd = 2.5
            k_lower = max(0, int(np.floor( k_expected - nstd*k_std ))) 
            k_upper = min(L, int(np.ceil( k_expected + nstd*k_std )))
            k_bounds = range(k_lower, k_upper+1) 
        else:
            k_bounds = range(L + 1)

        p_XK = 0.0
        for k in k_bounds:
            p_XK += pX(pr, k) * pK_approx(k, N, M, L)

        p_XHK += p_XK * pH( L, N, p_λ )

    # p_XHK = sum( [ np.sum([ pX(pr, k) * pK(k, N, M, L) for k in range(L+1)])*pH( L, N, p_λ ) for L in L_bounds ] )        
    return p_XHK

def pK_approx(k,N,M,L):
    """Probability that the intersection of M and L random elements of a set of N elements is k.

    Args:
        k (int): Desired size of the intersection of sets of size M and L
        N (int): Size of universal set
        M (int): Size of one subset
        L (int): Size of other subset 

    Returns:
        float: probability
    """
    return np.exp( comb_log_approx(L,k) + comb_log_approx(N-L, M-k) - comb_log_approx(N,M) )
    
# helpers
@cache
def factorial_log_approx(n):
    return 0.5*( np.log( 2*np.pi*n ) ) + n*(np.log(n) - 1)

def comb_log_approx(n,k, tol=1e-3):
    if (k > n) | (k < 0) | (n < 0):
        return -np.inf
    elif (k == n) | (n == 0) | (k == 0):
        return 0
    elif (k/n < tol) | ((n-k)/n < tol): # if k is very close to the extreme, compute the exact combinatoric
        # return np.log( comb(n,k) ) # no need to make it exact, as k small makes it manageable
        return np.log( choose(n,k) ) # no need to make it exact, as k small makes it manageable
    else:
        return ( factorial_log_approx(n) - factorial_log_approx(k) - factorial_log_approx(n-k) )