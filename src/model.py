import numpy as np
import pandas as pd

# Note: There's an `choose_initial_spreaders` function in `utils`. Check whether to leave it there or include it here.

# OPTIMIZATION IDEA:
# - Handle scipy.sparse matrices instead of the edgelist format. Slicing rows (sources) and columns (targets) should be more efficient for sparse matrices.
class IndependentHaltingCascade():

    def __init__(self, edgelist, nodelist=None, recommendation_probs=None, application_probs=None, hiring_probs=None) -> None:
        """<X>_probs must be ordered in the same way than edgelist, or at least be a series with edge_ids as indexes.

        Args:
            edgelist (pandas.dataframe): edgelist with columns ['source','target'] representing the network.
            nodelist (pandas.dataframe, optional): nodelist where node_ids should be the index with columns ['application_probs','hiring_probs']. Defaults to None.
            recommendation_probs ({float, itr}): Recommendation probabilities for each edge in the edgelist. Defaults to None.
            application_probs ({float, itr}): _description_. Defaults to None.
            hiring_probs ({float, itr}): _description_. Defaults to None.
        """                
        
        self.edgelist = edgelist.copy(deep=True)
        self.edgelist.rename( columns=dict(zip( edgelist.columns[:2], ['source','target'] )), inplace=True )
        
        if nodelist is None:
            nodelist = list( set( self.edgelist['source'] ).union(self.edgelist['target']) ) 
            nodelist = pd.DataFrame( index=nodelist )
        self.nodelist = nodelist
        
        if recommendation_probs is not None:
            self.edgelist['recommendation_probs'] = recommendation_probs
        if application_probs is not None:
            self.nodelist['application_probs'] = application_probs
        if hiring_probs is not None:
            self.nodelist['hiring_probs'] = hiring_probs
    
    # TODO: Create a more verbose output (dictionary with keys 'diffusion_steps', 'applicants_steps', 'hires')?
    def simulate(
            self,
            initial_spreaders, 
            max_diffusion_time=100,
            return_chain=False,
        ):
        """Simulates the Independent Halting Cascades (IHC) model using `initial_spreaders` as seed nodes.

        Args:
            initial_spreaders (list): list of seed node ids. The ids should be present in the 'source' column of `edgelist`.
            max_diffusion_time (int, optional): maximum number of diffusion steps to perform. Defaults to 100.
            return_chain (bool, optional): whether to return the full recommendation chain. Defaults to False.

        Returns:
            results (dict): dictionary with keys:
                - 'task_completed': whether the task was completed (someone was hired)
                - 'recommenders': list of lists, where each sublist contains the recommenders at each time step
                - 'applicants': list of lists, where each sublist contains the applicants at each time step
                - 'hires': list of nodes that were hired (if any)
                - 'recommenders_chain' (optional): list of dataframes, where each dataframe contains the recommendation edges at each time step
        """    

        # Create an internal (deep) copy of the edgelist (will be deleted at the end of the simulation)
        edgelist = self.edgelist.copy(deep=True)
        edgelist = edgelist.set_index('source')
        self.tmp_edgelist = edgelist
          
        # init 
        results = dict()       
        results['task_completed'] = False
        results['recommenders'] = []
        results['applicants'] = []
        results['hires'] = None

        t = 0
        spreaders = list(initial_spreaders)
        results['recommenders'].append( spreaders )
        results['applicants'].append( [] )
        if return_chain:            
            results['recommenders_chain'] = []
        
        # Get the different sources (could not correspond to the whole userlist if someone doesn't have an outedge)
        # Note: sources are the indexeses of the edgelist by construction
        sources = set( self.tmp_edgelist.index.unique() )
        
        # Realize diffusion process
        while (len(spreaders) > 0) and (t <= max_diffusion_time):            
            # Obtain the new recommendations
            # TODO: make the function take sources=None. We don't want to do the extra computation
            spreaders, applicants, hires, chains = self._step(spreaders, sources=sources, return_chain=return_chain)
            
            # Update collection variables            
            t+=1
            results['recommenders'].append( spreaders )
            results['applicants'].append( applicants )            
            #chains 
            if return_chain:
                results['recommenders_chain'].append( chains['recommenders_chain'] )

            if len(hires) > 0:
                # print("Someone was hired!")
                results['task_completed'] = True
                results['hires'] = hires
                break

        del self.tmp_edgelist
        return results
    
    def _step(self, spreaders, sources=None, edgelist=None, return_chain=False):
        """Takes one step of the independent halting cascade model based on the seed nodes `spreaders` and the `edgelist`.

        Args:
            edgelist (pandas.dataframe): network represented with an edgelist with columns ['source','target','recommendation_probability_col']
            initial_spreaders (list): list of seed node ids. The ids should be present in the 'source' column of `edgelist`.
            recommended_nodes (set): set of nodes in the network who have been recommended already, so we don't consider them.
            recommendation_prob_col (str, optional): name of the column in edgelist containing the edge recommendation probabilities. Defaults to 'recommendation_probability'.
            sources (list, optional): list of all possible spreaders in the network. nodes not in this list will be removed. If `sources==None`, no spreader is removed. Defaults to None.

        Returns:
            new_spreaders, applicant_nodes, hired_nodes, chains
        """    
        # Propagation step in a nutshell:
        #   - 1. Remove current spreaders from 'target'
        #   - 2. Compute new spreaders
        #   - 3. Remove current spreaders from 'sources'

        # 0. Check which spreaders can actually spread (i.e. has out neighbors)
        # Still needed? Maybe for directed networks only.
        if sources is not None:
            spreaders = [node for node in spreaders if node in sources]
            # spreaders[np.isin(spreaders, sources)] # if sources is a numpy array

        # 1. Remove current spreaders as targets (spreaders can't be recommended)
        if edgelist is None:        
            self.tmp_edgelist = self.tmp_edgelist[ ~self.tmp_edgelist['target'].isin(spreaders) ]
        else:
            self.tmp_edgelist = edgelist[ ~edgelist['target'].isin(spreaders) ]      

        # 2. Compute new spreaders
        #   - 2.1 Determine which users get recommended according coin tosses
        spreaders_mask = self.tmp_edgelist.index.isin(spreaders)
        # Get pairs of possible (spreaders, targets) Note: targets could be repeated if they are neighbors of many spreaders.
        spreaders_neighs = self.tmp_edgelist[spreaders_mask]
        recommendations = self._get_recommendations( edgelist=spreaders_neighs )
        recommendation_mask = recommendations['success']
        #   - 2.2 Get the set of newly recommended users
        recommended_nodes = list( recommendation_mask[recommendation_mask].index.unique() )
        #   - 2.3 Get the nodes that will apply for a job (and therefore not continue the chain)
        applicants_mask = self._get_applicants( recommended_nodes=recommended_nodes )
        applicant_nodes = list( applicants_mask[applicants_mask].index.unique() )
        #   - 2.4 Get the applicants that get hired (and therefore stop the chain)
        hired_nodes = self._get_hired( applicants=applicant_nodes )
        #   - 2.5 Get the new spreaders as those who didn't apply but got recommended
        new_spreaders = list( applicants_mask[~applicants_mask].index.unique() )

        # 3. Remove current spreaders as sources (old spreaders can't give recommendations anymore)
        #   - 3.1 Remove old spreaders
        self.tmp_edgelist = self.tmp_edgelist[~spreaders_mask]
        #   - 3.2 Remove applicants (they should not spread anymore)
        self.tmp_edgelist = self.tmp_edgelist[ ~self.tmp_edgelist['target'].isin(applicant_nodes) ]

        chains = {}
        if return_chain:
            chains['recommenders_chain'] = recommendations
        return new_spreaders, applicant_nodes, hired_nodes, chains

    # helpers
    def _get_applicants(self, recommended_nodes, random_state=None):
        
        # Generate random application thresholds
        application_thresholds = np.random.rand( len(recommended_nodes) )
        # Get the application probability of each of the nodes
        potential_applicants = self.nodelist.loc[recommended_nodes,'application_probs']
        # Determine which recommended nodes applies and which ones do not
        applicants_mask = potential_applicants > application_thresholds
        return applicants_mask

    def _get_hired(self, applicants, random_state=None):

        n_applicants = len(applicants)        
        if n_applicants == 0:
            return []
        else:
            # Generate random hiring thresholds
            hiring_thresholds = np.random.rand( len(applicants) )
            # Get the hiring probability of each of the nodes
            potential_hires = self.nodelist.loc[applicants,'hiring_probs']
            # Determine which applicants are hired
            hires_mask = potential_hires > hiring_thresholds
            hires = list( hires_mask[hires_mask].index.unique() )            
            return hires
    
    def _get_recommendations(self, edgelist, random_state=None):
        """Compute which users get recommended from a given edgelist with a column containing their recommendation probabilities.

        Args:
            edgelist (pandas.DataFrame): _description_

        Returns:
            recommendation_mask (pandas.Series): Boolean for each edge in `edgelist` of wheter the edge was recommended or not. 
        """
        # TODO: deal with random_state
                
        # Generate random recommendation thresholds
        recommendation_thresholds = np.random.rand( len(edgelist) )
        # Get the recommendation probability for each of the potential targets.
        # recommendation_probabilities = edgelist.set_index('target')['recommendation_probs']
        recommendation_probabilities = edgelist.reset_index().set_index('target')[['source','recommendation_probs']]
        # Determine which targets were recommended and which ones were not
        # successful_recommendations = recommendation_probabilities['recommendation_probs'] > recommendation_thresholds
        recommendation_probabilities['success'] = recommendation_probabilities['recommendation_probs'] > recommendation_thresholds

        # return successful_recommendations
        return recommendation_probabilities