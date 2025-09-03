import numpy as np
def p_incentives(pu, fv, β):    
    return (1 - np.exp(-β*pu*fv)) / (1 - np.exp(-β))

def p_incentives_inv(x, fv=1, β=1):
    return - np.log( 1 - x*(1-np.exp(-β)) )/β