import numpy as np
import networkx as nx
from community import best_partition
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

def distribution_sweep(X, xrange=None, n_kde_vals=500, return_distribution_matrix = False):
    if xrange is None: # TODO: do it for quantiles (outliers can make plots explode)
        xmin = min( [ min(x) for x in X.values()] )
        xmax = max( [ max(x) for x in X.values()] )
    else:
        xmin, xmax = xrange
    
    n_params = len( X )
    distribution_matrix = np.zeros((n_kde_vals, n_params-1))
    kde_input_vec = np.linspace(xmin, xmax, n_kde_vals)
    
    ymin, ymax = np.infty, -np.infty    
    for col, (param, x) in enumerate( X.items() ):

        ymin = min(ymin, param)
        ymax = max(ymax, param)
        
        if col == 0:
            continue
            
        # Initialize and fit kde object
        kde = gaussian_kde( x )
        # Compute the KDE histogram values
        kde_vals = kde( kde_input_vec )
        # Normalize
        kde_vals /= max(kde_vals)
        # Assign
        distribution_matrix[:, col-1] = kde_vals
    
    ## VIZ
    plt.figure( figsize=(12,4) )
    plt.imshow(distribution_matrix, extent=[ymin, ymax, xmax, xmin], cmap='viridis', aspect='auto')
    # plt.colorbar()
    
    if return_distribution_matrix:
        return plt.grid(False), distribution_matrix, (xmin, xmax), (ymin, ymax)
    else:
        return plt.grid(False)
    
## Network visualization
def visualize_communities(G, node_to_comm=None, pos=None, node_size=300, heterogeneous_sizes=True, with_labels=False):
    """
    Draws the graph with nodes colored by community.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    if node_to_comm is None:
        comm_ids = [0 for n in G.nodes()]
    else:
        comm_ids = [node_to_comm[n] for n in G.nodes()]
        
    # map community IDs to 0..C-1 for consistent colors
    unique_ids = {cid:i for i,cid in enumerate(sorted(set(comm_ids)))}
    colors = [unique_ids[cid] for cid in comm_ids]

    # PageRank for node sizes
    # pr = nx.pagerank(G)
    pr = dict( nx.degree(G) )
    pr_max = max(pr.values())
    if heterogeneous_sizes:
        sizes = [node_size * (pr[n] / pr_max)**2 + 1 for n in G]  # scale factor
    else:
        sizes = node_size // 3 

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap="tab20", alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.3)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    plt.tight_layout()