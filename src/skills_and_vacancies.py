import numpy as np
import pandas as pd

## Skill generation helper ##
def sample_application_and_hiring_probs(nodes, λ, nν, correlated_skills=False, n_skills=None, return_skills=False, **kw):    
    """Sample application and hiring probabilities for each node in `nodes`.
        If `correlated_skills` is True, sample correlated skills.
        Returns:
            p_application: pd.Series with application probabilities indexed by node
            p_hiring: pd.Series with hiring probabilities indexed by node
    """

    if correlated_skills:
        n_groups = kw.pop('n_groups')
        alpha = kw.pop('alpha')
        skills = sample_skills_correlated(nodes, λ, n_groups=n_groups, alpha=alpha, n_skills=n_skills, **kw)
    else:
        skills = sample_skills(nodes, λ, n_skills=n_skills)

    if type(nodes) == int:
        nodes = range(nodes)        
    if n_skills is None:
        n_skills = max(len(s) for s in skills)    
        
    vacancy = sample_vacancy(nν, n_skills)
    pa = compute_application_probabilities(skills, vacancy)
    ph = compute_hiring_probabilities(skills, vacancy)

    # hack: always have someone hirable
    # if sum(ph) == 0:
    #     node_ix = np.random.choice( range(len(nodes)) )
    #     pa.iloc[node_ix] = 1.0
    #     ph.iloc[node_ix] = 1.0
    if return_skills:
        return pa, ph, skills
    return pa, ph

## Sample user skills
def sample_skills(nodes, λ, n_skills=None):
    """Sample agents' skillset.
    """
    if type(nodes) == int:
        nodes = range(nodes)
        
    # Sample number of skills per agent
    n_skills_per_user = sample_skills_per_agent(len(nodes), λ)    
    if n_skills is None:
        n_skills = max(n_skills_per_user)

    # Sample each agent skillset
    U = np.arange(n_skills) # universal skillset
    agent_skills = [set(np.random.choice(U, size=n, replace=False )) \
                    for n in n_skills_per_user]
    return pd.Series(agent_skills, index=nodes)

def sample_skills_per_agent(n_nodes, λ):
    """Sample number of skills per agent from a Poisson distribution with parameter λ"""
    skills_per_agent = np.random.poisson( λ, n_nodes )
    # Ensure each agent has at least one skill
    skills_per_user = np.clip(skills_per_agent, 1, None)
    return skills_per_user

def sample_vacancy(n, n_skills):
    n_skills = max(n, n_skills)
    return set( np.random.choice(range(n_skills), size=n, replace=False ) )    

## Skill helpers
# vacancy requirement fullfillment
def skill_compatibility(skills, vacancy):
    return len(skills.intersection(vacancy)) / len(vacancy)

# Hiring probabilities
def compute_application_probabilities(agent_skills, vacancy):
    def application_proba(skills, vacancy):
        return skill_compatibility(skills, vacancy)        
    return pd.Series(agent_skills).apply(application_proba, args=(vacancy,))
    
# Hiring probabilities
def compute_hiring_probabilities(agent_skills, vacancy):
    def hiring_proba(skills, vacancy):
        return skill_compatibility(skills, vacancy) == 1        
    return pd.Series(agent_skills).apply(hiring_proba, args=(vacancy,)).astype(int)

## Correlated skills
def sample_skills_correlated(
    nodes, # list of node indices (or int for number of nodes)
    λ,  # avg number of skills per agent
    n_groups, # number of skill groups to induce correlation
    alpha=0.5,  # correlation factor (the smaller the higher correlation)
    n_skills=None,
    groups=None,
    rng=None,
    **kw,
):
    """
    Correlated skills via group-weighted sampling without replacement.

    For each subject i:
      1) Draw θ_i ~ Dirichlet(alpha * 1_M). Smaller alpha => more concentrated => stronger correlation.
      2) Weight each skill k by θ_i[group(k)] / |group(k)|.
      3) Sample x[i] distinct skills w.r.t. those weights.

    Returns:
        onehot : (n, K) uint8 matrix
        indices (optional) : list of chosen index arrays per subject
    """
    rng = np.random.default_rng() if rng is None else rng

    if type(nodes) == int:
        nodes = range(nodes)
        
    # Sample number of skills per agent
    n_skills_per_user = sample_skills_per_agent(len(nodes), λ)
    if n_skills is None:
        n_skills = max(n_skills_per_user)
        
    groups = make_groups(n_skills, n_groups) if groups is None else groups    
    if groups.shape != (n_skills,):
        raise ValueError("`groups` must have shape (K,)")
    
    M_ = int(groups.max()) + 1
    if M_ != n_groups:
        # tolerate if user provided groups with fewer/extra labels, but keep M consistent
        n_groups = M_
        
    group_sizes = np.bincount(groups, minlength=n_groups).astype(float)
    if np.any(group_sizes == 0):
        raise ValueError("At least one group has no skills.")

    # Sample skills based on Dirichlet weights. Each group's presence is determined by alpha
    U = np.arange(n_skills) # universal skillset
    agent_skills = []
    for n in n_skills_per_user:
        theta = rng.dirichlet(alpha * np.ones(n_groups))
        weights = theta[groups] / group_sizes[groups]  # per-skill weights
        p = weights / weights.sum()
        agent_skills.append( set(rng.choice(U, size=n, replace=False, p=p)) )
    return pd.Series(agent_skills, index=nodes)

# helpers
def make_groups(K: int, M: int):
    """
    Partition K skills into M groups as evenly as possible.
    groups[k] ∈ {0,...,M-1} is the group of skill k.
    """
    groups = np.repeat(np.arange(M), repeats=int(np.ceil(K / M)))[:K]
    return groups.astype(int)

def majority_groups(S, m):
    nodes = S.index
    S = onehot_encode_skills(S) # doesn't change S globally
    n, k = S.shape
    sizes = np.full(m, k // m)
    sizes[:k % m] += 1
    idx = np.cumsum(sizes)[:-1]
    groups = np.split(S, idx, axis=1)

    counts = np.stack([g.sum(axis=1) for g in groups], axis=1)  # shape (n, m)
    # prefer the LAST group on ties
    majorities = (m - 1) - np.argmax(counts[:, ::-1], axis=1)
    return pd.Series(majorities, index=nodes)

def onehot_encode_skills(skills, dtype=np.uint8):
    """Onehot vectorized conversion of ragged skills lists to an (n, K) one-hot matrix.
    """
    n_nodes = len(skills)
    n_skills = np.max([max(s) if len(s)>0 else 0 for s in skills])+1    
    out = np.zeros((n_nodes, n_skills), dtype=dtype)
    
    # Build vectorized scatter indices
    lens = np.fromiter((len(s) for s in skills), count=n_nodes, dtype=int)
    if lens.sum() == 0:
        return out
    rows = np.repeat(np.arange(n_nodes), lens)
    cols = np.concatenate(skills.apply(list))
    out[rows, cols] = 1
    return out