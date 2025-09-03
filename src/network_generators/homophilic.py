import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import pairwise_distances
# local
from skills_and_vacancies import onehot_encode_skills

# Note: using Jaccard is justified at https://www.nature.com/articles/s44284-023-00009-1
# Note: the network model is from Talaga 2020 https://www.jasss.org/23/2/6.html
def generate_homophilic_network_edgelist(skills, k_avg, homophily=0.5, metric='jaccard', seed=None):
    """Generate a homophilic network based on node `skills`.
    """
    D = construct_distance_matrix(skills, metric)
    P = probability_matrix_talaga(D, k_avg, homophily)
    return sample_edgelist(P, seed=seed)

def construct_distance_matrix(skills, metric='jaccard'):
    """Construct the pairwise distance matrix from node positions."""
    # One-hot encode skills
    # n_skills = np.max([max(s) if len(s)>0 else 0 for s in skills])+1
    skills_onehot = onehot_encode_skills(skills)
    # Compute pairwise distances
    D = pairwise_distances(skills_onehot.astype(bool), metric=metric)
    np.fill_diagonal(D, 2) # insure self-distances are huge    
    return D

def probability_matrix_talaga(distance_matrix, k_avg, homophily):
    # number of nodes
    N = distance_matrix.shape[0]
    # minimum possible expected degree
    k_avg_min = p_ij_talaga(distance_matrix, np.inf, b=0).sum() / N
    if k_avg < k_avg_min:
        raise ValueError(f"Target average degree {k_avg} is less than minimum possible {k_avg_min:.2f} for given distance matrix.")
    
    # construct probability matrix
    b = find_characteristic_distance(distance_matrix, k_avg, N, homophily)
    P_talaga = p_ij_talaga(distance_matrix, homophily=homophily, b=b)
    np.fill_diagonal(P_talaga, 0)
    return P_talaga

# Model's p_ij (Talaga, 2020)
def p_ij_talaga(D, homophily=2, b=1): # SDA
    if homophily >= 20:
        return (D <= b).astype('int')
    return 1 / (1 + (D/b)**homophily)

def find_characteristic_distance(distance_matrix, k_avg, N, homophily, tol=1e-3):
    """Find the characteristic distance b that gives the desired average degree k_avg"""
    # Loss function for root finding
    def loss(b):
        return k_avg - 1/N * p_ij_talaga(distance_matrix, homophily=homophily, b=b).sum()
    # Find characteristic distance
    return brentq(loss, 1e-20, 1, xtol=tol, rtol=tol)

### GAUER 2016 (not used)
# def p_ij_gauer(λ, α, D, N=None):
#     return λ * α**D
# def λ_correction(k_avg, α, N):
#     return k_avg * np.log(α)**2 / ( 2*(N-1)*(α - 1 - np.log(α)) )


# other helpers
def sample_adjacency(P, rng=None, dtype=np.uint8):
    """
    Sample an adjacency matrix A with independent edge probabilities P.
    P: (N, N) array of probabilities in [0,1]
    """
    rng = np.random.default_rng(rng)
    A = (rng.random(P.shape) < P).astype(dtype)
    return A

# helpers
def sample_edgelist(P, seed=None, allow_self_loops=False):
    """Sample an edgelist from the probability matrix P.    
    """
    rng = np.random.default_rng(seed)    
    mask = rng.random(P.shape) < P
    if not allow_self_loops:
        np.fill_diagonal(mask, False)
        
    src, tgt = np.nonzero(mask)
    return pd.DataFrame({"source": src, "target": tgt})