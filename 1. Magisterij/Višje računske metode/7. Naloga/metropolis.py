import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import rand
from numba import jit

class Metropolis:
    def __init__(self, init_config, mcmove, evaluate, par, correl=None, rep=1):
        self.config = init_config
        self.shape = self.config.shape
        self.mcmove = mcmove
        self.eval = evaluate
        self.par = par
        self.correl = correl
        self.rep = rep

    def run(self, beta, steps):
        config = np.copy(self.config)
        e0, extra0 = self.eval(config, self.shape, beta, self.par)
        E = [e0]
        extra = [extra0]
        if self.correl != None:
            C = [self.correl(config)]
        for i in range(steps):
            config, dEnergy, dextra, Energy_list, H_list, V_list ,epsilon = self.mcmove(config, self.shape, beta, self.par, self.rep)
            E.append( E[-1] + dEnergy )
            extra.append( np.array(extra[-1]) + np.array(dextra) )
            if self.correl != None:
                C.append( self.correl(config) )
        if self.correl != None:
            return config, E, extra, C
        else:
            return config, E, extra, Energy_list, H_list, V_list ,epsilon