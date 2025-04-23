# import numpy as np
import cupy as np

######################################
### Phase only or full simulation? ###
######################################
phase_only = False

##################
### Resolution ###
##################
n = 1024    # Resolution
n_half = n//2+1
kcut = n/3.0 # float
ord = 4    # Runge-Kutta order
kmax = (kcut)**2  #     kmax: maximum truncation for dealiasing
tiny =  0.000001   #     tiny: minimum truncation for dealiasing

############
### Time ###
############
cfl = 0.5             # CFL safety factor
cfl_cad = 4           # Number of time steps before cfl is changed (saves some time).
H = 3.             # Number of wall-hours to run for.
step =  500000000     # Numer of steps in run   
cstep = 1000          # Number of steps between time series output
thstep = 500          # Number of steps between theta histogram updates
sstep = 500000        # Number of steps between spectra saves     
tstep = 5000000       # Number of steps between field output
# # For testing
# mult = 20
# step =  1000*mult     # Numer of steps in run   
# cstep = 10*mult          # Number of steps between time series output
# thstep = 10*mult          # Number of steps between theta histogram updates
# sstep = 100*mult        # Number of steps between spectra saves     
# tstep = 500*mult       # Number of steps between field output

########################
### Fluid parameters ###
########################
fp0 = 4.00                 # streamfunction forcing amplitude
u0 = 0.10                  # streamfunction ic amplitude
kdn = 35.0                 # lowest forced wavenumber
kup = 37.0                 # highest forcing wavenumber
nu = 3.125e-8              # viscosity, 0.001 for n=256 and nn=1
hnu = 0.7937005259840997   # hypoviscosity, 0.5 
nn = 2                     # order of dissipation
mm = 2                     # order of hypo-dissipation
seed = 123456              # random seed
iflow = 3                  # forcing choice (1 = sin(x)*sin(y), 2 = const inj, 3 = random forcing)
dt_corr = 0.00             # forcing correlation time
triad_phase_hist = False   # If true, then loads and updates histograms of triad phases
Nbins = 30                 # Sets the number of bins for the PDFs of thetas
alpha = 1.75                # Initial KE spectrum (from largest scale to kup) is KE(k) = k**(-alpha), with integrated, total KE = u0. 
beta = 1.75                 # Initial KE spectrum (from kup to smallest scales) is KE(k) = k**(-beta), with integrated, total KE = u0. ONLY USED IN PHASE ONLY VERSION
idir = '../ins'
odir = '../outs'
