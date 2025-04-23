# BEFORE importing cupy 
import os

GPU_IDs = [1] # IDs of GPUs that are available (cross-check with gpustat in a terminal)
IDs_txt = ",".join(map(str, GPU_IDs)) # "ID[0],ID[1],ID[2],..."
os.environ["CUDA_VISIBLE_DEVICES"] = IDs_txt # Only these GPUS will be seen by the program after this line 

import numpy 
import cupy as np
import sys,os
from pathlib import Path
import time as time_wall
# Import subroutines
from subroutines import *

# Read input parameters
from parameter import *

######################
### INITIALIZATION ###
######################
# Set up log
log = open('./log.txt','a')
sys.stdout = log

# Create a 'RUNNING.txt' file
with open('./RUNNING.txt', 'w') as creating_new_txt_file:
   pass
print("Empty RUNNING File Created Successfully",flush=True)

# If the seed is fed through an argument, it supersedes the one provided by parameter.py
num_args = len(sys.argv)
if num_args>1:
    seed = int(sys.argv[1])
    print('Using new seed = %s' % seed, flush = True)

# Initialize random number generator (using numpy because it's faster)
rng = numpy.random.default_rng(seed)

# If recording triad statistics, load relevant information
if triad_phase_hist:
    # Load triads for histogram
    triads = np.loadtxt(idir+'/triads.txt')
    Ntriads = triads.shape[0]
    print("Gathering histogram statistics for %s triads." % Ntriads, flush = True)

    # Define histogram
    thetauuu = np.zeros((Nbins,Ntriads))
    # Define scriptK
    scriptK = np.zeros((2,Ntriads))
    # Set count to zero for scriptK average
    i_count = 0

    # Now process time-series triads
    triads_ts = np.loadtxt(idir+'/triads_ts.txt')
    Ntriads_ts = triads_ts.shape[0]
    print("Gathering temporal statistics for %s triads." % Ntriads_ts, flush = True)
    
# Builds the wave number and the square wave number matrixes
# In spectral space, index 0 is the ky axis, index 1 is the kx axis
# In real space, index 0 is the y axis, index 1 is the x axis
ka = np.fft.fftfreq(n,d=(1/n)) # ky
ka_half = np.fft.rfftfreq(n,d=(1/n)) # kx
KY,KX = np.meshgrid(ka,ka_half,indexing='ij')
ka2 = KX**2+KY**2
# Imaginary matrix
I = 1j*np.ones((n,n_half),dtype=np.complex128)

###########
### FFT ###
###########
# This is where the FFT planning is made in the fortran version. 
# At the moment, no FFT-related calls are made. 
# Eventually, with further optimization, we will likely add a planning call here.

##########################
### INITIAL CONDITIONS ###
##########################
# Read status.py
stat,time = np.loadtxt('./status.py') # stat is the output number
stat = int(stat)

if stat==0:
    dump = 0 # For use in spectra and transfers
    ini = 1 # Initial time-step
    timet = tstep
    timec = cstep
    times = sstep
    timeth = thstep

    # Stream function IC (random phase up to kup)
    ps = np.zeros((n,n_half),dtype=complex)
    # cond = (ka2<=kmax)&(ka2>=tiny)
    cond = (ka2<=kup**2)&(ka2>=4)
    phase = rng.uniform(low=-np.pi,high=np.pi,size=ps.shape)
    phase = np.asarray(phase)
    ps[cond] = (np.sqrt(ka2[cond])/kup)**((-alpha-3.0)/2.0) * (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))
    if phase_only: 
        cond = (ka2>kup**2)&(ka2<=kmax)
        ps[cond] = (np.sqrt(ka2[cond])/kup)**((-beta-3.0)/2.0) * (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))
    # Ensure 'realness' in the kx = 0 axis:
    ps[n_half:,0] = np.flip(np.conj(ps[1:n_half-1,0]))
    ps[0,0] = ps[n_half-1,0] = 0.0
    
    # Renormalize
    E = energy(ps,1,ka2)
    ps *= np.sqrt(2.0*u0/E)
else:
    ini = int((stat-1)*tstep)
    dump = float(ini)/float(sstep)+1
    times = 0
    timeth = 0
    timet = 0
    timec = 0
    timecorr = 0

    # Load the saved output file
    R1 = np.load(idir+'/ps.'+f'{int(stat):03}'+'.npy',allow_pickle=True)

    # FFT to get ps
    ps = np.fft.rfftn(R1)

    # If traid_phase_hist, then load the histogram array
    if triad_phase_hist:
        # Check to see if thetauuu file exists:
        my_file = Path(odir+'/thetauuu.npy')
        if my_file.is_file():
            # Load
            thetauuu[:] = np.load(my_file,allow_pickle=True)[:]

        # Check to see if scriptK file exists:
        my_file = Path(odir+'/scriptK.npy')
        if my_file.is_file():
            # Load
            scriptK[:] = np.load(my_file,allow_pickle=True)[:]

        # Also load the count file, if it exists:
        my_file = Path(odir+'/count.txt')
        if my_file.is_file():
            # Load
            i_count = int(np.loadtxt(my_file))
            print('Continuing average of scriptK, i_count = %i' % i_count, flush=True)


print('Starting from time-step %s and time %.3f.' % (ini,time), flush=True)

###############
### FORCING ###
###############
dt = CFL_condition(ps,KX,KY,I)
if iflow==1:
    ## OLD VERSION (from fortran code)
    # kdn=kup
    # if num_args>1:
    #     # If we're doing an ensemble run, where we change the seed by
    #     # feeding arguments to the execution, then we add a random
    #     # phase at the beginning of each run.
    #     phase1=rng.uniform(low=-np.pi,high=np.pi)
    #     phase2=rng.uniform(low=-np.pi,high=np.pi)
    # else:
    #     phase1=phase2=0.0
    # R1 = np.sin(2*np.pi*kup*np.mgrid[:n:1,:n:1][0]/n+phase1)+np.sin(2*kup*np.pi*np.mgrid[:n:1,:n:1][1]/n+phase2)
    # # FFT to get ps
    # fp = np.fft.rfftn(R1)
    # # Renormalize
    # fp[(ka2>kmax)&(ka2<tiny)]=0.0 
    # E = energy(fp,1,ka2)
    # fp *= fp0/np.sqrt(E)

    ## NEW VERSION
    # Stream function forcing (kdn to kup)
    fp = np.zeros((n,n_half),dtype=complex)
    cond = (ka2<=kup**2)&(ka2>=kdn**2)
    if num_args>1:
        # If we're doing an ensemble run, where we change the seed by
        # feeding arguments to the execution, then we add a random
        # phase at the beginning of each run.
        phase = rng.uniform(low=-np.pi,high=np.pi,size=fp.shape)
    else:
        # Otherwise, go with a random phase which is the same for every run.
        rng2 = np.random.default_rng(1)
        phase = rng2.uniform(low=-np.pi,high=np.pi,size=fp.shape)
    phase = np.asarray(phase)
    fp[cond] = (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))
    # Ensure 'realness' in the kx = 0 axis:
    fp[n_half:,0] = np.flip(np.conj(fp[1:n_half-1,0]))
    fp[0,0] = fp[n_half-1,0] = 0.0
    
    # Renormalize
    E = energy(fp,1,ka2)
    fp *= fp0/np.sqrt(E)
    
elif iflow==2:
    fp = const_inj(ps,ka2,rng)
elif iflow==3:
    fp = rand_force(dt,ka2,ka,ka_half,rng)
else:
    sys.exit('ERROR. The variable iflow must be either 1, 2, or 3. Stopping simulation.')

#################
### MAIN LOOP ###
#################
start_time=time_wall.time()
sim_end = start_time + 60*60*H # run for H hours
# import tqdm # FOR TESTING
# for t in tqdm.tqdm(range(ini,step)): # FOR TESTING
# for t in range(ini,step):
t = ini
while (time_wall.time() < sim_end)&(t<=step):
    if (t%cfl_cad)==0: # Update dt every cfl_cad time steps.
        dt = CFL_condition(ps,KX,KY,I)

    # Every 'cstep' steps, outputs global values.  
    # See the cond_check subroutine for details.
    if timec==cstep:
        timec = 0
        cond_check(ps,fp,time,ka2)
        if triad_phase_hist:
            # Output time series of triad energy and p hase for various triads.
            corr_check(ps,time,ka2,ka,ka_half,KX,KY,triads_ts)

    # Every 1000 steps, check if RUNNING.txt is present, otherwise end the stepping and save last outputs.
    if (t%1000)==0:
        if not os.path.isfile('./RUNNING.txt'):
            print("RUNNING.txt has been deleted. Stopping run. tstep = %s time = %s" % (t,time),flush=True)
            break

    # Random force
    if iflow==3:
        fp = rand_force(dt,ka2,ka,ka_half,rng)

    # Every 'sstep' steps, generates external files with the power spectrum
    if times==sstep:
        times = 0
        dump += 1 # Update specturm count
        spectrum(ps,dump,ka2)
        transfers(ps,dump,ka2,KX,KY,I)
        with open('./time_spec.txt', 'a') as f:
            f.write(f"{int(dump):04} {time:14.6F}\n")

    # Every 'thstep' steps, calculates and updates thetauuu histogram and scriptK online average (if triad_phase_hist is true)
    if ((timeth==thstep)&(triad_phase_hist)): 
        timeth = 0
        # Updates thetauuu, defined 'globally' using module theta_hist
        i_count,thetauuu,scriptK = thetauuu_calc(ps,triads,i_count,thetauuu,scriptK,ka,ka_half)
        
    # Every 'tstep' steps, stores the results of the integration
    if timet==tstep:
        timet = 0
        stat += 1
        # Write current state to file:
        R1 = np.fft.irfftn(ps)
        np.save(odir+'/ps.'+f'{int(stat):03}'+'.npy',R1)
        
        R1 = np.fft.irfftn(laplak2(ps,ka2))
        np.save(odir+'/ww.'+f'{int(stat):03}'+'.npy',R1)
        
        # If traid_phase_hist, then overwrites the current thetauuu histogram file. Updates scriptK average file.
        if triad_phase_hist:
            np.save(odir+'/thetauuu.npy',thetauuu)
            np.save(odir+'/scriptK.npy',scriptK)
            np.save(odir+'/i_count.npy',i_count)
            
        
        with open('./time_field.txt', 'a') as f:
            f.write(f"{int(stat):03} {time:14.6F}\n")
    
    ######## Runge-Kutta step 1
    C3 = np.copy(ps)
    
    ######## Runge-Kutta step 2
    for o in range(ord,0,-1):
        # Iflow2: change forcing to keep constant energy
        if iflow==2:
            fp = const_inj(C3,ka2,rng)
            
        # Nonlinear term
        nl = laplak2(C3,ka2) # Makes -w_2D
        nl = poisson(C3,nl,ka2,KX,KY,I) # Makes -curl(u_2D x w_2D)
        
        tmp1 = dt/float(o)
        C3 = NL(ps,nl,fp,tmp1,nu,hnu,nn,mm,ka2,kmax)
        
    ######## Runge-Kutta step 3
    ps = np.copy(C3)

    # Update times and counters
    t += 1 
    timet += 1
    times += 1
    timeth += 1
    timec += 1
    time += dt   
    
############## END OF MAIN LOOP ##############
end_time=time_wall.time()
print('Finished time-stepping loop. Total wall time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(t-ini)/(end_time-start_time)),flush=True)

# Save last time:
stat += 1
print("Saving files last time... Stat = %s, iteration = %s, time = %.4e" % (stat,t,time), flush=True)
# Write current state to file:
R1 = np.fft.irfftn(ps)
np.save(odir+'/ps.'+f'{int(stat):03}'+'.npy',R1)

R1 = np.fft.irfftn(laplak2(ps,ka2))
np.save(odir+'/ww.'+f'{int(stat):03}'+'.npy',R1)

# If traid_phase_hist, then overwrites the current thetauuu histogram file. Updates scriptK average file.
if triad_phase_hist:
    np.save(odir+'/thetauuu.npy',thetauuu)
    np.save(odir+'/scriptK.npy',scriptK)
    np.save(odir+'/i_count.npy',i_count)
    

with open('./time_field.txt', 'a') as f:
    f.write(f"{int(stat):03} {time:14.6F}\n")

# Delete RUNNING.txt if it hasn't already been deleted.
if os.path.isfile('./RUNNING.txt'):
    os.remove('./RUNNING.txt')

# Delete variables (might not be necessary...)
del ps,fp,R1,C3,ka2,KX,KY,nl

print('Finished saving. Exiting... \n \n',flush=True)
