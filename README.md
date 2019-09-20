# SpiNNaker Ear

code that supports running a model of a EAR from a PyNN script.

## Supports:
1. AutoPauseAndResumeCompatibility
2. Projections from PyNN models with Static Synapses.
3. Projections from the Ear model to other models.
4. Supports Recording of Spikes, Moc, and SpikeProbability through the
defacto PyNN population record interface.


## Does not support:
1. Projections from PyNN models with Structural or Plastic Synapses.
2. Limitations that are inherited from the defacto SpiNNaker populations.

## Summary
The EAR model models the outer ear, the inner ear, and the individual ear hair
cells. It was originally written by Robert James (https://github
.com/rjames91) as his PhD work and cleaned
up and supported by the SpiNNaker software team.

The original forked software can be found here:
https://github.com/rjames91/OME_SpiNN
