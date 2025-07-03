# 2DHD GPU
### About
Pseudo-spectral solver for the two-dimensional Navier Stokes equations. Written in Python, and uses CuPy to provide GPU acceleration.

### Santiago J. Benavides 

(santiago.benavides@upm.es)

Foundation of pseudospectral code is based on a 2D Fortran code written by Pablo Mininni (Universidad de Buenos Aires).

### Has three different modes
1. Full Navier Stokes (`main.py`)
2. Constrained Euler (`main_CE.py`). See [She & Jackson Phys. Rev. Lett. 70, 1255 (1993)](https://doi.org/10.1103/PhysRevLett.70.1255) for info on Constrained Euler, and [Zhou Phys. Fluids 5, 2511–2524 (1993)](https://doi.org/10.1063/1.858764) for info on how it is implemented here.
3. Phase-only (`main_phase_only.py`). See [Arguedas-Leiva et al. Phys. Rev. Research 4, L032035 (2022)](https://doi.org/10.1103/PhysRevResearch.4.L032035) for info on phase-only formulations.

### Capabilities: 
* Ensemble members
* Triad phase and amplitude statistics
