# import numpy as np
import cupy as np

###########################
### Numerical precision ###
###########################
#Tf = np.float32
#Tc = np.complex64
#Ti = np.int16
Tf = np.float64
Tc = np.complex128
Ti = np.int32

################
### Ensemble ###
################
Nens = 10  # Number of ensemble members

##################
### Resolution ###
##################
n = 256    # Resolution
n_half = n//2+1
kcut = Tf(n/3.0) # float
ord = 2    # Runge-Kutta order
kmax = Tf((kcut)**2)  #     kmax: maximum truncation for dealiasing
tiny =  Tf(0.000001)   #     tiny: minimum truncation for dealiasing

############
### Time ###
############
cfl = Tf(0.5) # CFL safety factor
cfl_cad = 4           # Number of time steps before cfl is changed (saves some time).
H = np.inf            # Number of wall-hours to run for.
step =  np.inf        # Numer of steps in run   
cstep = 1000          # Number of steps between time series output
thstep = 100          # Number of steps between theta histogram updates
thtsstep = Nens*1000  # Number of steps between theta time series output
sstep = 500000        # Number of steps between spectra saves     
tstep = 5000000       # Number of steps between field output

########################
### Fluid parameters ###
########################
fp0 = Tf(4.00)                # streamfunction forcing amplitude
u0 = Tf(0.10)                  # streamfunction ic amplitude
kdn = Tf(8.0)                # lowest forced wavenumber
kup = Tf(9.0)                # highest forcing wavenumber
nu = Tf(2e-6)         # viscosity, 0.001 for n=256 and nn=1
hnu = Tf(0.5)   # hypoviscosity, 0.5 
nn = Ti(2)                     # order of dissipation 
mm = Ti(2)                     # order of hypo-dissipation
seed = 123456              # random seed
iflow = 1                  # forcing choice (1 = sin(x)*sin(y), 2 = const inj, 3 = random forcing)
dt_corr = 0.00             # forcing correlation time
triad_phase_hist =False   # If true, then loads and updates histograms of triad phases
Nbins = 30                 # Sets the number of bins for the PDFs of thetas
alpha = 1.75                # Initial KE spectrum (from largest scale to kup) is KE(k) = k**(-alpha), with integrated, total KE = u0. 
beta = 1.75                 # Initial KE spectrum (from kup to smallest scales) is KE(k) = k**(-beta), with integrated, total KE = u0. ONLY USED IN PHASE ONLY VERSION
idir = '../ins'
odir = '../outs'
