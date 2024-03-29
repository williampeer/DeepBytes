# Setup
- Please see: http://deeplearning.net/software/theano/install_ubuntu.html
- On Ubuntu 14.04 LTS, the included setup script may be run, which installs the packages and Theano for CPU-execution 
    using the most common options, as included in the link provided above.
    NOTE: This installs using sudo, and installs python and the packages in the global system scope. You might want to
    consider creating a testing environment for the packages, in case they will interfere with some of the system's
    previous variables and packages.

# DeepBytes

- kWTA now gets all > elements and all == elements for count(els)<k.
- input-ec weights are intialized only once, and have to be re-wired explicitly.
- neuronal turnover in the dg is to be called explicitly between training sets.
- Extracted patterns and pseudopatterns from hippocampal experiments are stored for later retrieval and consolidation
    to the neocortical module.


# HPC-parametrization

HPC-constructor:

- dims: number of neurons in the input-layer, ec-layer, dg-layer, ca3-layer, and output-layer
- connection_rate_input_ec: self-explanatory
- perforant_path: connection rate ec-dg and ec-ca3
- mossy_fibers: connection rate dg-ca3
- firing_rate_ec, firing_rate_dg, firing_rate_ca3: these decide the number of k active neurons in kWTA
- _gamma: learning rate in unbounded Hebbian learning
- _epsilon: steepness parameter in the transfer function, tanh(sum(in) / _epsilon)
- _nu: learning rate in the contrained Hebbian learning
- _turnover_rate: the relative frequency of neurons in the DG that are to be randomly re-initialized according to the
    firing rates and connection rates associated with the neurons
- _k_m, _k_r: damping factors for refractoriness in the equations for chaotic neurons, located in the ca3- and output-layers
- _a_i: constant outer stimuli / external input parameter, used in the chaotic neuron equations.
- _alpha: the scaling factor for refractoriness


# Storage

All of the chaotically recalled patterns, as well as corresponding (hippocampal) pseudopatterns may be retrieved from
storage using the Tools.py. These may then be used to rapidly consolidate the patterns to the neocortical module, and 
 obtain the associated goodness of fit value(s).