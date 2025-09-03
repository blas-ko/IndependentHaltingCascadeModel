import numpy as np
from itertools import product
from pathlib import Path
# local
from model import IndependentHaltingCascade as IHC

### Helpers for parallel code-running ### 
# Basic function to run model and collect diagnosis statistcs
def run_model(
    edgelist,
    recommendation_probs,
    application_probs,
    hiring_probs,
    nodelist=None,
    initial_spreaders=None,
    k0_range=None,
    return_chain_ends=False,
):
    """Run the IHCM model and return aggregated results:
        - tree_depth: length of the recommendation chain
        - tree_size: number of unique nodes reached by recommendations and applications
        - tree_applicants_size: number of unique nodes that applied
        - tree_success: whether the task was completed (someone was hired)
    """
    # Instantiate model
    model = IHC(
        edgelist=edgelist,            
        recommendation_probs=recommendation_probs,
        application_probs=application_probs,
        hiring_probs=hiring_probs,
        nodelist=nodelist,
    )    

    # Sample initial spreader    
    if initial_spreaders is None:
        nodes = model.nodelist.index
        initial_spreaders = sample_nodes(nodes, 1, edgelist=edgelist, k_range=k0_range)

    # Run model
    result = model.simulate(initial_spreaders, return_chain=False)

    # Collect diagnosis statistics
    return model_diagnosis(result, return_chain_ends=return_chain_ends)

def model_diagnosis(result, return_chain_ends=False):
    """"Given the result of a model simulation, compute basic diagnosis statistics:
        - tree_depth: length of the recommendation chain
        - tree_size: number of unique nodes reached by recommendations and applications
        - tree_applicants_size: number of unique nodes that applied
        - tree_success: whether the task was completed (someone was hired)
    """
    nodes_reached = set( flatten(result['recommenders']) ).union( flatten(result['applicants']) ) 
    tree_applicants_size = len( flatten(result['applicants']) )    
    tree_size = len(nodes_reached)
    tree_depth = len(result['recommenders'])
    tree_success = result['task_completed']

    # If no recommendations nor applications in leaf nodes, that length wasn't reached.
    if ( len(result['recommenders'][-1]) == 0 ) & ( len(result['applicants'][-1]) == 0 ):
        tree_depth -= 1

    if return_chain_ends:
        initial_spreaders = result['recommenders'][0]
        hired = result['hires']
        return tree_depth, tree_size, tree_applicants_size, tree_success, (initial_spreaders, hired)

    return tree_depth, tree_size, tree_applicants_size, tree_success

# Wrap independent simulations into function
def perform_independent_model_simulations(
    n_simulations,
    new_edgelist_every=-1,
    edgelist_generator=None,
    edgelist=None,
    run_model_func=run_model,
    return_chain_ends=False,
    **kw,
):
    """Performs `n_simulations` independent model simulations.

    If edgelist is None, a function edgelist_generator() shall be provided to create new edgelists.
    Tip: Beforehand, create edgelist_generator = lambda: edgelist_generator(args...)
    If new_edgelist_every > 0, a new edgelist will be generated every `new_edgelist_every` simulations.
    Args:
        n_simulations: number of independent simulations to run
        new_edgelist_every: if >0, a new edgelist will be generated every `new_edgelist_every` simulations
        edgelist_generator: function that generates a new edgelist when called
        edgelist: if provided, the same edgelist will be used for all simulations
        run_model_func: function to run the model. It must accept edgelist as first argument, and **kw as keyword arguments.
    Returns:
        - tree_depths: list of tree_depth for each simulation
        - tree_sizes: list of tree_size for each simulation
        - tree_applicants_sizes: list of tree_applicants_size for each simulation
        - tree_successes: list of tree_success (boolean) for each simulation
    """

    tree_depths = []
    tree_sizes = []
    tree_applicants_sizes = []
    tree_successes = []
    tree_chain_ends = [] if return_chain_ends else None
    for i in range(n_simulations):        
        # Create new edgelist if generator provided
        if (i % new_edgelist_every == 0) & (edgelist_generator is not None):
            edgelist = edgelist_generator()  # TODO: Make it possible to pass args to generator       
        
        # run model
        result = run_model_func(edgelist=edgelist, **kw)
        if return_chain_ends:
            tree_depth, tree_size, tree_applicants_size, tree_success, chain_ends = result
            tree_chain_ends.append(chain_ends)
        else:
            tree_depth, tree_size, tree_applicants_size, tree_success = result
        # append results
        tree_depths.append( tree_depth )
        tree_sizes.append( tree_size )
        tree_applicants_sizes.append( tree_applicants_size )
        tree_successes.append( tree_success )

    if return_chain_ends:
        return tree_depths, tree_sizes, tree_applicants_sizes, tree_successes, tree_chain_ends
    return tree_depths, tree_sizes, tree_applicants_sizes, tree_successes

# helpers #
def sample_nodes(nodelist, n_nodes=1, edgelist=None, k_range=None,):
    """Sample n_nodes from nodes. If k_range and edgelist are given, one may condition on degree."""
    if k_range is not None:
        degrees = edgelist.groupby('source').size()
        kmin, kmax = k_range
        nodelist = degrees[ (degrees >= kmin) & (degrees < kmax) ].index.values
    return np.random.choice(nodelist, n_nodes, replace=False)

def flatten(arr):
    return [x for xs in arr for x in xs]

def reshape_list(a, n_rows, n_cols):
    return np.array([a[i*n_cols:(i+1)*n_cols] for i in range(n_rows)])

def nested_dict_from_list(a, xs, ys):
    pairs = list(product(xs, ys))
    nested = {x: {} for x in xs}
    for (x, y), val in zip(pairs, a):
        nested[x][y] = val
    return nested

def invert_nested_dict(d):
    inverted = {y: {} for y in d[list(d.keys())[0]].keys()}
    for x in d.keys():
        for y in d[x].keys():
            inverted[y][x] = d[x][y]
    return inverted

# saving and reading
def save_experiment(path, *items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as a 1-D object array to avoid ragged-shape errors
    np.save(path, np.array(items, dtype=object), allow_pickle=True)
    # print(f"Data succesfully saved at '{path}'")

def load_experiment(path):
    loaded = np.load(path, allow_pickle=True)   # returns a 1-D object ndarray
    return tuple(loaded.tolist())               # back to a Python tuple