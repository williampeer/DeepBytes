# DeepBytes

- kWTA now gets all > elements and all == elements for count(els)<k.
- input-ec weights are intialized only once, and have to be re-wired explicitly.
- neuronal turnover in the dg is to be called explicitly between training sets.


# API
* HPC - constructor:
dims: number of neurons in the input-layer, ec-layer, dg-layer, ca3-layer, and output-layer
connection_rate_input_ec: self-explanatory
perforant_path: connection rate ec-dg and ec-ca3
mossy_fibers: connection rate dg-ca3
firing_rate_ec, firing_rate_dg, firing_rate_ca3: these decide the number of k active neurons in kWTA
_gamma: learning rate in unbounded Hebbian learning
_epsilon: steepness parameter in the transfer function, tanh(sum(in) / _epsilon)
_nu: learning rate in the contrained Hebbian learning
_turnover_rate: the relative frequency of neurons in the DG that are to be randomly re-initialized according to the
    firing rates and connection rates associated with the neurons
_k_m, _k_r: damping factors for refractoriness in the equations for chaotic neurons, located in the ca3- and output-layers
_a_i: constant outer stimuli / external input parameter, used in the chaotic neuron equations.
_alpha: the scaling factor for refractoriness
