import pandas as pd
import graph_tool.all as gt

## Network reader
def import_graphtool_network(name, eprops=[], directed=False):

    # import network as graph-tool object
    g = gt.collection.ns[name]

    # Convert to edgelist
    el = pd.DataFrame( g.get_edges(eprops) )
    el.rename(columns={0:'source',1:'target'}, inplace=True)

    if not directed:
        el = to_undirected(el)

    # n_nodes = g.num_vertices()
    nodelist = g.get_vertices()
    return el, nodelist

def to_undirected(edgelist):
    el_reverted = edgelist.rename( columns={'source':'target','target':'source'} )
    el = pd.concat( [edgelist, el_reverted] )
    return el.drop_duplicates()