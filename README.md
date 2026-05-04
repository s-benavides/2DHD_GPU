# 2DHD GPU
### About
Pseudo-spectral solver for the two-dimensional Navier Stokes equations. Written in Python, and uses CuPy to provide GPU acceleration.

### Santiago J. Benavides 

(Santiago.Benavides@ed.ac.uk)

Foundation of pseudospectral code is based on a 2D Fortran code written by Pablo Mininni (Universidad de Buenos Aires).

### Has three different modes
1. Full Navier Stokes (`main.py`)
2. Constrained Euler (`main_CE.py`). See [She & Jackson Phys. Rev. Lett. 70, 1255 (1993)](https://doi.org/10.1103/PhysRevLett.70.1255) for info on Constrained Euler, and [Zhou Phys. Fluids 5, 2511–2524 (1993)](https://doi.org/10.1063/1.858764) for info on how it is implemented here.
3. Phase-only (`main_phase_only.py`). See [Arguedas-Leiva et al. Phys. Rev. Research 4, L032035 (2022)](https://doi.org/10.1103/PhysRevResearch.4.L032035) for info on phase-only formulations.

### Capabilities: 
* Ensemble members
* Triad phase and amplitude statistics

### Instructions
Create a directory following the structure (and naming convention) of `run_dir`, with three subdirectories named `run`,`ins`, and `outs`. To start a new run from scratch, make sure that `run/status.py` is all set to zero. Change the parameters in `params.py`, and simply start the run by executing `main.py` (or whatever version you'd like to run). To continue from a finished run, look at `time_field.txt`, copy the information for the latest output to `status.py` and don't forget to copy the corresponding `psi.XYZ.npy` from the `outs` directory into the `ins` directory. If you'd like to re-set the whole directory and start again from scratch, run `clean_new_run.sh`. 

If you'd like to collect triad statistics, the first step is to create a list of triads whose statistics you'd like to gather. This can be done using the `Triad_Generation.py` script contained in the `other_scripts` directory. Alternatively, you could produce your own list of triads, making sure to write each triad into a separate row containing `[kx ky px py]` as the columns. After creating the list of triads, copy the generated triad list (`triads_*.txt`) as `triads.txt` into the `ins` directory within the `run_dir` directory. Finally, make sure to change `triad_phase_hist` to `True` in the parameter file before starting the run.
