import numpy as np
from fractions import Fraction
from scipy.stats import poisson
from scipy.stats import binom
from scipy.special import comb
from functools import cache

@cache
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
    p_λ = np.dot( p_λs, ps_nmax ) # Type 1
    # Type 2: do nothing and for loop each p_λ
    # for (p_λ, p_nmax) in zip(p_λs, ps_nmax):
    # p_XHK += p_nmax * sum( p_XHKs ) 
                        
    if approx:        
        L_expected = N*p_λ
        L_std = np.sqrt( N*p_λ*(1 - p_λ) )
        L_lower = max(0, int(round(L_expected - 2.5*L_std))) 
        L_upper = min(int(round(L_expected + 2.5*L_std)), N)        
        L_bounds = range(L_lower, L_upper+1)
    else:
        L_bounds = range(N+1)


    p_XHK = sum( [ np.sum([ pX(pr, k) * pK(k, N, M, L) for k in range(L+1)])*pH( L, N, p_λ ) for L in L_bounds ] )        
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
    n_max_range = range(5, 25) # for N=5000, most mass is between 10 and 12 and changes very slowly with N.
    skill_inclusion_probs = np.zeros( len(n_max_range) )
    nmax_probs = np.zeros( len(n_max_range) )

    # Compute the maximum number of skills based on the confidence interval of the number of nodes
    for (i, nmax) in enumerate( n_max_range ):
    
        for l in range(nmax+1):
            skill_inclusion_probs[i] += poisson.pmf(l, λ) * comb(l, n_ν)
            # skill_inclusion_probs[i] += poisson.pmf(l, λ) * ( comb(nmax - n_ν, nmax - l) / comb(nmax, l) )
        
        # Normalize by the number of counts
        skill_inclusion_probs[i] /= comb(nmax, n_ν)

        # Add nmax prob
        nmax_probs[i] =  nmax_pdf(nmax, λ, N) 

    # Return highest probabilities so that 98% of the probability mass is contained
    prob_indexes = highest_probabilities_indices(nmax_probs)

    return skill_inclusion_probs[prob_indexes], nmax_probs[prob_indexes]

## helpers
def highest_probabilities_indices(prob_dist, mass_threshold=0.99):
    prob_dist = np.array(prob_dist)
    sorted_indices = np.argsort(prob_dist)[::-1]  # Sort indices based on probabilities in descending order
    # sorted_probs = np.sort(prob_dist)[::-1]  # Sort probabilities in descending order
    
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