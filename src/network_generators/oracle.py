import numpy as np
import pandas as pd

## Oracle system: star network with oracle at center
def generate_oracle(nodes, oracle_degree):
    """Creates an oracle node with access to `oracle_degree` contacts of the `nodelist`.
    """
    if type(nodes) == int:
        nodes = range(nodes)
    # Determine the neighbors of the oracle
    oracle_neighs = np.random.choice(nodes, oracle_degree, replace=False )
    # Create oracle as edgelist
    oracle = pd.DataFrame( columns=['source','target'] )
    oracle['target'] = oracle_neighs
    oracle['source'] = 'oracle'
    return oracle