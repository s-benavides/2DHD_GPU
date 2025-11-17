# BEFORE importing cupy 
import os

GPU_IDs = [4] # IDs of GPUs that are available (cross-check with gpustat in a terminal)
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

#########################
### PRECOMPUTE ARRAYS ###
#########################
# Builds the wave number and the square wave number matrixes
# In spectral space, index 0 is the kx axis, index 1 is the ky axis
# In real space, index 0 is the x axis, index 1 is the y axis
ka = np.asarray(np.fft.fftfreq(n,d=(1/n)),dtype=Tf) # kx
ka_half = np.asarray(np.fft.rfftfreq(n,d=(1/n)),dtype=Tf) # ky
KX,KY = np.meshgrid(ka,ka_half,indexing='ij')
ka2 = KX**2+KY**2
# Constant injection count:
cond = (np.sqrt(ka2)<kup)&(np.sqrt(ka2)>kdn)
Nf = np.sum(cond)
# Imaginary matrix
I = 1j*np.ones((n,n_half),dtype=Tc)
# For fractional dimension decimation (if dec_dim < 2), we build the projectorb
if dec_dim < 2.0:
    # First draw a random uniform distribution
    P_pre = np.asarray(rng.uniform(size=(Nens,n,n_half)),dtype=Tf)
    # Now apply the condition as a function of k
    P_frac = np.asarray((P_pre <= np.sqrt(ka2[None,:,:])**(dec_dim-2)),dtype=Tf)
    # Ensure 'realness' in the ky = 0 axis when projecting
    P_frac[:,n_half:,0] = np.flip(np.conj(P_frac[:,1:n_half-1,0]),axis=1)
    P_frac[:,0,0] = P_frac[:,n_half-1,0] = 0.0
    
# For shell integrating
inds_polar = []
for ii in range(n_half):
    kk = ii+1
    inds_polar.append(np.where(np.round(np.sqrt(ka2)).astype(Ti)==kk))

# If recording triad statistics, load relevant information
if triad_phase_hist:
    # Load triads for histogram
    triads = np.loadtxt(idir+'/triads.txt',dtype=Ti)
    Ntriads = triads.shape[0]
    print("Gathering histogram statistics for %s triads." % Ntriads, flush = True)

    # Define histogram
    thetauuu = np.zeros((Nbins,Ntriads),dtype=Ti)
    # Define scriptK and other averaged quantities
    scriptK = np.zeros((2,Ntriads),dtype=Tf)
    rhok = np.zeros((2,Ntriads),dtype=Tf)
    rhop = np.zeros((2,Ntriads),dtype=Tf)
    rhoq = np.zeros((2,Ntriads),dtype=Tf)
    Rkpq = np.zeros((2,Ntriads),dtype=Tf)
    Tkpq = np.zeros((2,Ntriads),dtype=Tf)
    # Set count to zero for averages
    i_count = 0

    # # Now process time-series triads
    # triads_ts = np.loadtxt(idir+'/triads_ts.txt',dtype=Ti)
    # Ntriads_ts = triads_ts.shape[0]
    # print("Gathering temporal statistics for %s triads." % Ntriads_ts, flush = True)
    
    # Compute index arrays, indKX,indKY,indPX, etc. X arrays have shape (Ntriads,n), Y arrays have shape (n,Ntriads)
    # To be used in thetauuu_calc
    indKX = np.zeros((Ntriads,n),dtype=Tf)
    indKY = np.zeros((n_half,Ntriads),dtype=Tf)
    indPX = np.zeros((Ntriads,n),dtype=Ti)
    indPY = np.zeros((n_half,Ntriads),dtype=Tf)
    indQX = np.zeros((Ntriads,n),dtype=Ti)
    indQY = np.zeros((n_half,Ntriads),dtype=Tf)
    kmag = np.zeros((Ntriads),dtype=Tf)
    pmag = np.zeros((Ntriads),dtype=Tf)
    qmag = np.zeros((Ntriads),dtype=Tf)
    for Ntr,triad in enumerate(triads):
        kx,ky,px,py = triad 
        qx = -kx-px
        qy = -ky-py
        # Magnitudes
        kmag[Ntr] = np.sqrt(kx**2+ky**2)
        pmag[Ntr] = np.sqrt(px**2+py**2)
        qmag[Ntr] = np.sqrt(qx**2+qy**2)
    
        # k
        if (ky>=0): # phi(kx,ky)
            sgn=1
        else: # -phi(-kx,-ky)
            ky=-ky
            kx=-kx
            sgn=-1
        indKX[Ntr,ka==kx] = sgn
        indKY[ka_half==ky,Ntr] = 1
    
        # p
        if (py>=0): # phi(px,py)
            sgn=1
        else: # -phi(-px,-py)
            py=-py
            px=-px
            sgn=-1
        indPX[Ntr,ka==px] = sgn
        indPY[ka_half==py,Ntr] = 1
        
        # q
        if (qy>=0): # phi(qx,qy)
            sgn=1
        else: # -phi(-qx,-qy)
            qy=-qy
            qx=-qx
            sgn=-1
        indQX[Ntr,ka==qx] = sgn
        indQY[ka_half==qy,Ntr] = 1
        
    # # To be used in corr_check (time-series of theta statistics)
    # indKX_ts = np.zeros((Ntriads_ts,n),dtype=Tf)
    # indKY_ts = np.zeros((n_half,Ntriads_ts),dtype=Tf)
    # indPX_ts = np.zeros((Ntriads_ts,n),dtype=Tf)
    # indPY_ts = np.zeros((n_half,Ntriads_ts),dtype=Tf)
    # indQX_ts = np.zeros((Ntriads_ts,n),dtype=Tf)
    # indQY_ts = np.zeros((n_half,Ntriads_ts),dtype=Tf)
    # kmag_ts = np.zeros((Ntriads_ts),dtype=Tf)
    # pmag_ts = np.zeros((Ntriads_ts),dtype=Tf)
    # qmag_ts = np.zeros((Ntriads_ts),dtype=Tf)
    # qxp_ts = np.zeros((Ntriads_ts),dtype=Tf)
    # for Ntr,triad in enumerate(triads_ts):
    #     kx,ky,px,py = triad 
    #     qx = -kx-px
    #     qy = -ky-py
        
    #     # qxp
    #     qxp_ts[Ntr] = qx*py-px*qy
        
    #     # Magnitudes
    #     kmag_ts[Ntr] = np.sqrt(kx**2+ky**2)
    #     pmag_ts[Ntr] = np.sqrt(px**2+py**2)
    #     qmag_ts[Ntr] = np.sqrt(qx**2+qy**2)
    
    #     # k
    #     if (ky>=0): # phi(kx,ky)
    #         sgn=1
    #     else: # -phi(-kx,-ky)
    #         ky=-ky
    #         kx=-kx
    #         sgn=-1
    #     indKX_ts[Ntr,ka==kx] = sgn
    #     indKY_ts[ka_half==ky,Ntr] = 1
    
    #     # p
    #     if (py>=0): # phi(px,py)
    #         sgn=1
    #     else: # -phi(-px,-py)
    #         py=-py
    #         px=-px
    #         sgn=-1
    #     indPX_ts[Ntr,ka==px] = sgn
    #     indPY_ts[ka_half==py,Ntr] = 1
        
    #     # q
    #     if (qy>=0): # phi(qx,qy)
    #         sgn=1
    #     else: # -phi(-qx,-qy)
    #         qy=-qy
    #         qx=-qx
    #         sgn=-1
    #     indQX_ts[Ntr,ka==qx] = sgn
    #     indQY_ts[ka_half==qy,Ntr] = 1

##########################
### INITIAL CONDITIONS ###
##########################
# Read status.py
stat,t,time = np.loadtxt('./status.py') # stat is the output number
stat = int(stat)
t = int(t)
ini = t

if stat==0:
    dump = 0 # For use in spectra and transfers
    t = 1 # Initial time-step
    timet = tstep
    timec = cstep
    times = sstep
    timeth = thstep
    timethts = thtsstep

    # Stream function IC (random phase up to kup)
    ps = np.zeros((Nens,n,n_half),dtype=Tc)
    phase = rng.uniform(low=-np.pi,high=np.pi,size=ps.shape)
    phase = np.asarray(phase,dtype=Tf)
    cond = (ka2<=kup**2)&(ka2>=tiny)
    ps = (np.sqrt(ka2[None,:,:])/kup)**((-alpha-3.0)/2.0) * (np.cos(phase) + 1j*np.sin(phase)) * cond[None,:,:]
    # If dec_dim < 2, project:
    if dec_dim < 2.0:
        ps *= P_frac
    # Ensure 'realness' in the ky = 0 axis:
    ps[:,n_half:,0] = np.flip(np.conj(ps[:,1:n_half-1,0]),axis=1)
    ps[:,0,0] = ps[:,n_half-1,0] = 0.0

    # Renormalize
    E = energy(ps,1,ka2)
    ps *= np.sqrt(2.0*u0/E[:,None,None])
else:
    dump = np.int64(np.float64(t)/np.float64(sstep))
    times = t%sstep
    timet = t%tstep
    timec = t%cstep
    timeth = t%thstep
    timethts = t%thtsstep

    # Load the saved output file
    R1 = np.load(idir+'/ps.'+f'{int(stat):03}'+'.npy')
    R1 = np.asarray(R1,dtype=Tf)

    # FFT to get ps
    ps = np.fft.rfftn(R1,axes=(1,2))
    
    Nens_t,_,_ = ps.shape
    if Nens_t<Nens:
        print('Nens in parameter.py does NOT match Nens from the input file. Changing Nens to match the input file. Nens = %i --> Nens = %i' % (Nens,Nens_t), flush=True)
        Nens = Nens_t
    elif Nens_t>Nens:
        print('Nens in parameter.py does NOT match Nens from the input file. Changing the shape of ps to match Nens. shape[0] = %i --> %i' % (ps.shape[0],Nens), flush=True)
        ps = ps[:Nens,:,:]

    # If traid_phase_hist, then load the histogram array
    if triad_phase_hist:
        # Check to see if thetauuu file exists:
        my_file = Path(odir+'/thetauuu.npy')
        if my_file.is_file():
            # Load
            thetauuu[:] = np.load(my_file)[:]

        # Check to see if scriptK file exists:
        my_file = Path(odir+'/scriptK.npy')
        if my_file.is_file():
            # Load
            scriptK[:] = np.load(my_file)[:]
        # Check to see if rhok file exists:
        my_file = Path(odir+'/rhok.npy')
        if my_file.is_file():
            # Load
            rhok[:] = np.load(my_file)[:]
        # Check to see if rhop file exists:
        my_file = Path(odir+'/rhop.npy')
        if my_file.is_file():
            # Load
            rhop[:] = np.load(my_file)[:]
        # Check to see if rhoq file exists:
        my_file = Path(odir+'/rhoq.npy')
        if my_file.is_file():
            # Load
            rhoq[:] = np.load(my_file)[:]
        # Check to see if Rkpq file exists:
        my_file = Path(odir+'/Rkpq.npy')
        if my_file.is_file():
            # Load
            Rkpq[:] = np.load(my_file)[:]
        # Check to see if Tkpq file exists:
        my_file = Path(odir+'/Tkpq.npy')
        if my_file.is_file():
            # Load
            Tkpq[:] = np.load(my_file)[:]

        # Also load the count file, if it exists:
        my_file = Path(odir+'/i_count.npy')
        if my_file.is_file():
            # Load
            i_count = int(np.load(my_file))
            print('Continuing averages, i_count = %i' % i_count, flush=True)


print('Starting from time-step %s and time %.3f.' % (t,time), flush=True)

###############
### FORCING ###
###############
dt = CFL_condition(ps,KX,KY,I)
if iflow==1:
    # Stream function forcing (kdn to kup)
    fp = np.zeros((Nens,n,n_half),dtype=Tc)
    cond = (ka2<=kup**2)&(ka2>=kdn**2)
    phase = rng.uniform(low=-np.pi,high=np.pi,size=fp.shape)
    phase = np.asarray(phase,dtype=Tf)
    fp = (np.cos(phase) + 1j*np.sin(phase)) * cond[None,:,:]
    # Ensure 'realness' in the ky = 0 axis:
    fp[:,n_half:,0] = np.flip(np.conj(fp[:,1:n_half-1,0]),axis=1)
    fp[:,0,0] = fp[:,n_half-1,0] = 0.0
    
    # If dec_dim < 2, project:
    if dec_dim < 2.0:
        fp *= P_frac
    
    # Renormalize
    E = energy(fp,1,ka2)
    fp *= fp0/np.sqrt(E[:,None,None])
elif iflow==2:
    fp = const_inj(ps,ka2,rng,Nf)
elif iflow==3:
    fp = rand_force(dt,ka2,ka,ka_half,rng)
else:
    sys.exit('ERROR. The variable iflow must be either 1, 2, or 3. Stopping simulation.')
    
#################
### MAIN LOOP ###
#################
start_time=time_wall.time()
sim_end = start_time + 60*60*H # run for H hours
while (time_wall.time() < sim_end)&(t<=step):
    if (t%cfl_cad)==0: # Update dt every cfl_cad time steps.
        dt = CFL_condition(ps,KX,KY,I)

    # Every 'cstep' steps, outputs global values.  
    # See the cond_check subroutine for details.
    if timec==cstep:
        timec = 0
        cond_check(ps,fp,time,ka2)

    # Every 1000 steps, check if RUNNING.txt is present, otherwise end the stepping and save last outputs.
    if (t%1000)==0:
        if not os.path.isfile('./RUNNING.txt'):
            print("RUNNING.txt has been deleted. Stopping run. tstep = %s, time = %s" % (t,time),flush=True)
            break

    # Random force
    if iflow==3:
        fp = rand_force(dt,ka2,ka,ka_half,rng)

    # Every 'sstep' steps, generates external files with the power spectrum
    if times==sstep:
        times = 0
        dump += 1 # Update spectrum count
        spectrum(ps,dump,ka2)
        transfers(ps,dump,ka2,KX,KY,I,inds_polar)
        with open('./time_spec.txt', 'a') as f:
            f.write(f"{int(dump):04} {time:14.6F}\n")

    # Every 'thstep' steps, calculates and updates thetauuu histogram and online averages (if triad_phase_hist is true)
    if ((timeth==thstep)&(triad_phase_hist)): 
        timeth = 0
        # Updates thetauuu
        i_count,thetauuu,scriptK,rhok,rhop,rhoq,Rkpq,Tkpq = thetauuu_calc(ps,i_count,thetauuu,scriptK,rhok,rhop,rhoq,Rkpq,Tkpq,indKX,indKY,indPX,indPY,indQX,indQY,kmag,pmag,qmag)
        
        
    # if ((timethts==thtsstep)&(triad_phase_hist)):
    #     timethts = 0
    #     # Output time series of triad energy and phase for various triads.
    #     corr_check(ps,time,ka2,KX,KY,I,indKX_ts,indKY_ts,indPX_ts,indPY_ts,indQX_ts,indQY_ts,kmag_ts,pmag_ts,qmag_ts,qxp_ts)
        
    # Every 'tstep' steps, stores the results of the integration
    if timet==tstep:
        timet = 0
        stat += 1
        # Write current state to file:
        R1 = np.fft.irfftn(ps,axes=(1,2))
        np.save(odir+'/ps.'+f'{int(stat):03}'+'.npy',R1)
        
        R1 = np.fft.irfftn(-laplak2(ps,ka2),axes=(1,2))
        np.save(odir+'/ww.'+f'{int(stat):03}'+'.npy',R1)
        
        # If traid_phase_hist, then overwrites the current thetauuu histogram file. Updates average files.
        if triad_phase_hist:
            np.save(odir+'/thetauuu.npy',thetauuu)
            np.save(odir+'/scriptK.npy',scriptK)
            np.save(odir+'/rhok.npy',rhok)
            np.save(odir+'/rhop.npy',rhop)
            np.save(odir+'/rhoq.npy',rhoq)
            np.save(odir+'/Rkpq.npy',Rkpq)
            np.save(odir+'/Tkpq.npy',Tkpq)
            np.save(odir+'/i_count.npy',i_count)
            
        with open('./time_field.txt', 'a') as f:
            f.write(f"{int(stat):03} {int(t)} {time:14.6F}\n")

    ######## Runge-Kutta step 1
    C3 = np.copy(ps)
    
    ######## Runge-Kutta step 2
    for o in range(ord,0,-1):
        # Iflow2: change forcing to keep constant energy
        if iflow==2:
            fp = const_inj(C3,ka2,rng,Nf)
            
        # Nonlinear term
        nl = laplak2(C3,ka2[None,:,:]) # Makes -w_2D
        nl = poisson(C3,nl,ka2,KX,KY,I) # Makes -curl(u_2D x w_2D)
        # If dec_dim < 2, project NL term into lattice
        if dec_dim < 2.0:
            nl *= P_frac
        
        tmp1 = dt/Tf(o)
        C3 = NL(ps,nl,fp,tmp1,nu,hnu,nn,mm,ka2[None,:,:],kmax)
        
    ######## Runge-Kutta step 3
    ps = np.copy(C3)

    # Update times and counters
    t += 1 
    timet += 1
    times += 1
    timec += 1
    timeth += 1
    timethts += 1
    time += dt   
    
############## END OF MAIN LOOP ##############
end_time=time_wall.time()
print('Finished time-stepping loop. Total wall time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(t-ini)/(end_time-start_time)),flush=True)

# Save last time:
stat += 1
print("Saving files last time... Stat = %s, iteration = %s, time = %.4e" % (stat,t,time), flush=True)
# Write current state to file:
R1 = np.fft.irfftn(ps,axes=(1,2))
np.save(odir+'/ps.'+f'{int(stat):03}'+'.npy',R1)

R1 = np.fft.irfftn(-laplak2(ps,ka2),axes=(1,2))
np.save(odir+'/ww.'+f'{int(stat):03}'+'.npy',R1)

# If traid_phase_hist, then overwrites the current thetauuu histogram file. Updates average files.
if triad_phase_hist:
    np.save(odir+'/thetauuu.npy',thetauuu)
    np.save(odir+'/scriptK.npy',scriptK)
    np.save(odir+'/rhok.npy',rhok)
    np.save(odir+'/rhop.npy',rhop)
    np.save(odir+'/rhoq.npy',rhoq)
    np.save(odir+'/Rkpq.npy',Rkpq)
    np.save(odir+'/Tkpq.npy',Tkpq)
    np.save(odir+'/i_count.npy',i_count)
    
with open('./time_field.txt', 'a') as f:
    f.write(f"{int(stat):03} {int(t)} {time:14.6F}\n")

# Delete RUNNING.txt if it hasn't already been deleted.
if os.path.isfile('./RUNNING.txt'):
    os.remove('./RUNNING.txt')

# Delete variables (might not be necessary...)
del ps,fp,R1,C3,ka2,KX,KY,nl
if triad_phase_hist:
    del indKX,indKY,indPX,indPY,indQX,indQY
    
print('Finished saving. Exiting... \n \n',flush=True)