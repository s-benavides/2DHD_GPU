# import numpy as np
import cupy as np

###########################
### Numerical precision ###
###########################
#Tf = np.float32
#Tc = np.complex64
#Ti = np.int32
Tf = np.float64
Tc = np.complex128
Ti = np.int64

################
### Ensemble ###
################
Nens = 40  # Number of ensemble members

##################
### Resolution ###
##################
n = 256    # Resolution
n_half = n//2+1
kcut = Tf(n/3.0) # float
ord = 2    # Runge-Kutta order
kmax = Tf((kcut)**2)  #     kmax: maximum truncation for dealiasing
tiny =  Tf(0.000001)   #     tiny: minimum truncation for dealiasing
dec_dim = Tf(1.95)         # Fractional dimension for decimation (dec_dim = 2 does nothing)

############
### Time ###
############
cfl = Tf(0.5) # CFL safety factor
cfl_cad = 10           # Number of time steps before cfl is changed (saves some time).
H = np.inf            # Number of wall-hours to run for.
step =  np.inf        # Numer of steps in run   
cstep = 1000          # Number of steps between time series output
thstep = 200          # Number of steps between theta histogram updates
thtsstep = Nens*1000  # Number of steps between theta time series output
sstep = 500000        # Number of steps between spectra saves     
tstep = 5000000       # Number of steps between field output

########################
### Fluid parameters ###
########################
fp0 = Tf(2.50)            # streamfunction forcing amplitude
u0 = Tf(0.10)              # streamfunction ic amplitude
kdn = Tf(35.0)             # lowest forced wavenumber
kup = Tf(37.0)             # highest forcing wavenumber
nu = Tf(8.e-29)            # viscosity, 0.001 for n=256 and nn=1
hnu = Tf(0.7937)           # hypoviscosity, 0.5 
nn = Ti(8)                 # order of dissipation 
mm = Ti(2)                 # order of hypo-dissipation
seed = 123456              # random seed
iflow = 3                  # forcing choice (1 = sin(x)*sin(y), 2 = const inj, 3 = random forcing)
dt_corr = 0.00             # forcing correlation time
alpha = 1.75                # Initial KE spectrum (from largest scale to kup) is KE(k) = k**(-alpha), with integrated, total KE = u0. 
beta = 1.75                 # Initial KE spectrum (from kup to smallest scales) is KE(k) = k**(-beta), with integrated, total KE = u0. ONLY USED IN PHASE ONLY VERSION

################
### Analysis ###
################
triad_phase_hist = False    # If true, then loads and updates histograms of triad phases. If no triads.txt file exists, it will turn to false. NOTE: for triad statistics for fractal decimation, start with =True but with no triads.txt file, so that it builds P_frac correctly. Then to start gathering statistics, add triads.txt to the ins directory.  
Nbins = 30                 # Sets the number of bins for the PDFs of thetas
shell_transf=False    # Set to True if you want shell-to-shell energy transfers to be measured
binning = 'octave'   # options: 'round' or  'floor', which are linear, and 'octave' which is log binning with a base chosen by nshells (if nshells=None, defaulting to sqrt(2))
nshell = 40          # If nshell=None: linear bins will have nshell=ceil(kcut), log bins will have log(ceil(kcut))/log(sqrt(2)) shells. If nshells>0: linear bins will choose nshell logspaced modes for K but will compute all ceil(kcut) modes for Q (in T(K,Q)), log bins will change the base so that log(ceil(kcut))/log(base) = nshells
### Transfer function a la Aluie & Eyink, Phys. Fluids 21, 115108 (2009)#
shell_transf_AE = False
width = 2.0 # Choose base size for bins (e.g. [K/width,K] for one bin)
Kvals = np.logspace(1,np.log10(kup),10,dtype=Ti) # Choose K values for receiving bin
nP = None                              # Choose number of P values for donor bins. If nP = None, nP = np.linspace(1,kcut,int(ceil(kcut))) so that the T(K,P) curve looks smooth, like Fig. 8 in A&E 2009
###
idir = '../ins'
odir = '../outs'
