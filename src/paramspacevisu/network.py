# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:41:21 2013

@author: micha
"""

import pyNN.brian as pynn
import numpy
import cPickle

default_params = {
"wi": 0.01,
"we": 0.005,
"ri": 30.,
"re": 30.,
"p_conn_et": 0.1,
"p_conn_it": 0.1,
"num_e":200,
"num_i":50,
"num_t":30}

# vary re between 10 and 50, ri between 10 and 50 (5 steps)
# next: vary we between 0.001 and 0.005, wi between 0.005 and 0.015 (5 steps)

class ThreePops(object):
    """
    Three populations of neurons: Excitatory, inhibitory, and target.
    200 exc (poisson), 50 inh (poisson), 50 target (IF).
    convergence: random 10 % (exc to target and inh to target)
    weights: w_inh = 4*w_exc 
    """
    def __init__(self):
        pass
    
    def make_network(self, param_dict=default_params):
        """
        Construct the network according to the parameters specified.
        """
        pynn.setup()
        self.exc = pynn.Population(param_dict['num_e'], 
                                   pynn.SpikeSourcePoisson, 
                                   cellparams={'rate':param_dict['re']})
        self.inh = pynn.Population(param_dict['num_i'], 
                                   pynn.SpikeSourcePoisson, 
                                   cellparams={'rate':param_dict['ri']})
        self.target = pynn.Population(param_dict['num_t'], 
                                      pynn.IF_cond_exp)
        self.target.record(to_file=False)
        
        connector_et = pynn.FixedProbabilityConnector(
                            p_connect=param_dict["p_conn_et"],
                            weights=param_dict["we"])
        connector_ei = pynn.FixedProbabilityConnector(
                            p_connect=param_dict["p_conn_it"],
                            weights=param_dict["wi"])
        self.prj_et = pynn.Projection(self.exc, self.target, 
                  method=connector_et)
        self.prj_it = pynn.Projection(self.inh, self.target, 
                  method=connector_ei, target='inhibitory')
    
    def update_weights_and_rates(self, param_dict):
        self.exc.set('rate', param_dict['re'])
        self.inh.set('rate', param_dict['ri'])
        self.prj_et.setWeights(param_dict['we'])
        self.prj_it.setWeights(param_dict['wi'])
    
    def run_network(self, duration=1000.):
        """
        Run the network for the specified duration. Returns spikes of target 
        population in GDF format.
        """
        pynn.run(duration)
        spikes = self.target.getSpikes()
        return spikes
        
def make_the_simulation(params=default_params):
    # first panel: vary re and ri between 10 and 50 in five steps
    num_spikes = numpy.zeros((25,25))
    param_dicts = numpy.zeros((25,25), dtype=object)
    net = ThreePops()
    net.make_network(param_dict=params)
    lastnumspikes = 0
    for ire,re in enumerate(numpy.linspace(20.,40.,5)):
        for iri,ri in enumerate(numpy.linspace(20.,40.,5)):
            for iwe,we in enumerate(numpy.linspace(0.003,0.007,5)):
                for iwi,wi in enumerate(numpy.linspace(0.008,0.012,5)):
                    print "re:%.1f, ri:%.1f, we:%.4f, wi:%.4f"%(re,ri,we,wi)
                    params = default_params.copy()
                    params.update({'re':re, 'ri':ri, 'we':we, 'wi':wi})
                    index = (iwe*5 + ire, iwi*5+iri)
                    param_dicts[index] = params
                    net.update_weights_and_rates(params)
                    spikes = net.run_network()
                    currentnumspikes = len(spikes)
                    num_spikes[index] = (currentnumspikes - lastnumspikes)/params['num_t']
                    lastnumspikes = currentnumspikes
    cPickle.dump(num_spikes, open('numspikes.cPickle','w'))
    cPickle.dump(param_dicts, open('paramdicts.cPickle','w'))

    
