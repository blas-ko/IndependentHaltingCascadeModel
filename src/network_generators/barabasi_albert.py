import networkx as nx
def generate_barabasi_albert(n_nodes, avg_degree):
    """Generate a Barabasi-Albert (pandas) edgelist using networkx.
    """
    G = nx.generators.barabasi_albert_graph(n_nodes, avg_degree)
    return nx.to_pandas_edgelist(G)