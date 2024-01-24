import numpy as np
import pandas as pd

def generate_erdos_renyi_edgelist(n_nodes, n_edges=None, prob_edge=None, avg_degree=None):
    """Generate an erdos-renyi (pandas) edgelist using numpy (and not networkx).
    """
    
    if avg_degree is not None:
        n_edges = int( n_nodes * avg_degree )

    if n_edges is None:
        n_edges = int(round(n_nodes * (n_nodes - 1) * prob_edge))    

    # Generate random pairs of connections
    el = np.random.randint(0, n_nodes, size=(n_edges, 2))
    
    # Assign them to a dataframe
    el = pd.DataFrame( el, columns=['source','target'] )

    # Drop multi-edges
    el = el.drop_duplicates(['source','target'])
    
    # Drop self-edges
    el = el[ el['source'] != el['target'] ]

    return el