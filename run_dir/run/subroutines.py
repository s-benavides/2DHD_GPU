import cupy as np
import numpy
# import numpy as np
from parameter import *
import warnings

######################
### Custom Kernels ###
######################
# Numerical precision:
if Tf == np.float32:
    float_name='float'
elif Tf==np.float64:
    float_name='double'

# """
# Two-dimensional derivative of the matrix 'a'

# ARGUMENTS
#  K : wave-vector for taking derivative 
#      K = KX if you want to take an x-derivative, etc.
#  a : input matrix
#  I : Imaginary matrix.

# RETURNS
#  b : the resulting matrix.
# """
derivk2 = np.ElementwiseKernel(
   ''+Tf.__name__+' K, '+Tc.__name__+' a,'+Tc.__name__+' I',
   ''+Tc.__name__+' b',
   'b = I*K*a',
   'derivk2')

# """
# Two-dimensional Laplacian of the matrix 'a'

# ARGUMENTS
#  a : input matrix
#  ka2: the square of the wave vector

# RETURNS
#  b : at the output contains the Laplacian d2a/dka2
# """
laplak2 = np.ElementwiseKernel(
   ''+Tc.__name__+' a,'+Tf.__name__+' ka2',
   ''+Tc.__name__+' b',
   'b = -ka2*a',
   'laplak2')

# """
# Multiplies and subtracts four matrices (for poission). Generic type so that it can be used with real or complex arrays.
# """
quad_diff = np.ElementwiseKernel(
   ''+Tf.__name__+' a,'+Tf.__name__+' b,'+Tf.__name__+' c,'+Tf.__name__+' d',
   ''+Tf.__name__+' e',
   'e = a*b-c*d',
   'quad_diff')

# """
# Multiplies and adds four matrices (for poission). Generic type so that it can be used with real or complex arrays.
# """
quad_plus = np.ElementwiseKernel(
   ''+Tf.__name__+' a,'+Tf.__name__+' b,'+Tf.__name__+' c,'+Tf.__name__+' d',
   ''+Tf.__name__+' e',
   'e = a*b+c*d',
   'quad_plus')

# """
# Dealiasing
# """
dealias = np.ElementwiseKernel(
    ''+Tc.__name__+' a, '+Tf.__name__+' ka2, '+Tf.__name__+' kmax',
    ''+Tc.__name__+' b',
'''
    if (ka2>kmax){
       b = 0;
    } else {
       b = a;
    }
''',
    'dealias')

# """
# Filter
# """
kfilt = np.ElementwiseKernel(
    ''+Tc.__name__+' a, '+Tf.__name__+' ka2, '+Tf.__name__+' kmin, '+Tf.__name__+' kmax',
    ''+Tc.__name__+' b',
'''
    if (ka2 > kmax || ka2 < kmin) {
       b = 0;
    } else {
       b = a;
    }
''',
    'kfilt')

# """
# Nonlinear term
# """
NL = np.ElementwiseKernel(
   ''+Tc.__name__+' ps,'+Tc.__name__+' nl,'+Tc.__name__+' fp, '+Tf.__name__+' dt, '+Tf.__name__+' nu, '+Tf.__name__+' hnu, '+Ti.__name__+' nn, '+Ti.__name__+' mm, '+Tf.__name__+' ka2, '+Tf.__name__+' kmax',
   ''+Tc.__name__+' out',
   f"""
    if (ka2 > kmax || ka2 == 0) {{
       out = 0;
    }} else {{
       out = (ps + ((-nl)/ka2+fp)*dt)/({float_name}(1.0) +(nu*pow({float_name}(ka2), {float_name}(nn)) + hnu*pow({float_name}(ka2), -{float_name}(mm)))*dt);
    }}
   """,
   'NL')

##########################
### Phase-only Kernels ###
# """
# Nonlinear term, only evolves the phases phi
# """
NL_phase_only = np.ElementwiseKernel(
   ''+Tc.__name__+' ps, '+Tf.__name__+' rho,'+Tf.__name__+' phi, '+Tc.__name__+' nl, '+Tf.__name__+' dt, '+Tf.__name__+' ka2, '+Tf.__name__+' kmax, '+Tc.__name__+' I',
   ''+Tf.__name__+' out',
   """
    if (ka2 > kmax || ka2 == 0 || rho == 0 ) {
       out = 0;
    } else {
       out = atan2(imag(ps), real(ps)) + imag(exp(-I*phi)*(-dt*nl/ka2)/rho);
    }
   """,
   'NL_phase_only')

# """
# Going from polar (rho,phi) to complex.
# """
polar_2_complex = np.ElementwiseKernel(
   ''+Tf.__name__+' rho,'+Tf.__name__+' phi, '+Tc.__name__+' I',
   ''+Tc.__name__+' out',
   'out = rho*exp(I*phi)',
   'polar_2_complex')

######################
### Other Kernels #### 
# """
# Used in corr_check
# """
dphi_dt = np.ElementwiseKernel(
   ''+Tc.__name__+' ps,'+Tc.__name__+' nl,'+Tf.__name__+' ka2, '+Tf.__name__+' kmax, '+Tc.__name__+' I',
   ''+Tf.__name__+' out',
   """
    if (ka2 > kmax || ka2 == 0 || abs(ps) == 0 ) {
       out = 0;
    } else {
       out = imag(exp(-I*atan2(imag(ps), real(ps)))*(-nl/ka2)/abs(ps));
    }
   """,
   'dphi_dt')

# """
# Averages calculations, to be used in thetauuu_calc
# """
scriptK_calc = np.ElementwiseKernel(
   ''+Tf.__name__+' kmag, '+Tf.__name__+' pmag, '+Tf.__name__+' qmag,'+Tf.__name__+' rhoks, '+Tf.__name__+' rhops, '+Tf.__name__+' rhoqs',
   ''+Tf.__name__+' out',
    f"""
    if (abs(rhoqs*rhops*rhoks) > 0) {{
       out = ((pow(qmag,{float_name}(2.0))-pow(pmag,{float_name}(2.0)))/(pow(kmag,{float_name}(2.0))))*abs((rhoqs*rhops)/(rhoks)) + ((pow(pmag,{float_name}(2.0))-pow(kmag,{float_name}(2.0)))/(pow(qmag,{float_name}(2.0))))*abs((rhoks*rhops)/(rhoqs)) + ((pow(kmag,{float_name}(2.0))-pow(qmag,{float_name}(2.0)))/(pow(pmag,{float_name}(2.0))))*abs((rhoks*rhoqs)/(rhops));
    }} else {{
        out = 0.0;
    }}
    """,
   'scriptK_calc')

averages_calc = np.ElementwiseKernel(
   ''+Tf.__name__+' rhoks,'+Tf.__name__+' rhops,'+Tf.__name__+' rhoqs,'+Tf.__name__+' thetas,'+Tf.__name__+' tmp',
   ''+Tf.__name__+' rhoks_tmp,'+Tf.__name__+' rhops_tmp,'+Tf.__name__+' rhoqs_tmp,'+Tf.__name__+' Rkpq,'+Tf.__name__+' Tkpq',
   """
   rhoks_tmp = tmp*rhoks;
   rhops_tmp = tmp*rhops;
   rhoqs_tmp = tmp*rhoqs;
   Rkpq = rhoks_tmp*rhops_tmp*rhoqs_tmp;
   Tkpq = Rkpq*cos(thetas);
   """,
   'averages_calc')

# """
# Calculate energy transfer term
# """
en_tran_calc = np.ElementwiseKernel(
   ''+Tc.__name__+' ps,'+Tc.__name__+' nl,'+Tf.__name__+' two',
   ''+Tf.__name__+' out',
   f"""
    out = two*real(ps*conj(nl));
    """,
   'en_tran_calc')

# """
# Calculate energy spectrum
# """
en_spec_calc = np.ElementwiseKernel(
   ''+Tc.__name__+' ps, '+Tf.__name__+' ka2, '+Tf.__name__+' two',
   ''+Tf.__name__+' out',
   f"""
    out = two*ka2*pow({float_name}(abs(ps)), {float_name}(2.0));
    """,
   'en_spec_calc')

###########################
### Spectral operations ###
###########################
def energy(C,kin,ka2):
    """
    Computes the mean kinetic or magnetic energy in 2D,
    and the mean square current density or vorticity.

    ARGUMENTS
    C  : input matrix with the scalar field (complex)
    kin: =0 computes the square of the scalar field
         =1 computes the energy
         =2 computes the current or vorticity
    ka2: the square of the wave vector

    RETURNS
    E  : at the output contains the energy
    """
    two = np.ones((n_half),dtype=Tf)
    two[1:] *= 2

    # We suppress warnings here because, if kin<0, the [0,0] element will be nan, since ka2 = 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warnings in this block
        E = np.nansum(two[None,None,:]*np.abs(C)**2*ka2[None,:,:]**kin,axis=(1,2))
    E /= n**4
    return E

def inerprod(a,b,kin,ka2):
    """
    ARGUMENTS
     a  : first  input matrix
     b  : second input matrix
     kin: = multiplies by the laplacian to this power
     ka2: the square of the wave vector

    RETURNS
     rslt : the inner product of the two matrices
    """ 
    two = np.ones((n_half),dtype=Tf)
    two[1:] *= 2

    # We suppress warnings here because, if kin<0, the [0,0] element will be nan, since ka2 = 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warnings in this block
        rslt = np.real(np.nansum((two[None,None,:]*ka2[None,:,:]**kin*a*np.conj(b)),axis=(1,2)))
    rslt /= n**4
    return rslt

def poisson(a,b,ka2,KX,KY,I):
    """
    Poisson bracket of the scalar fields A and B
    in real space.
    
    ARGUMENTS
     a: input matrix
     b: input matrix
     ka2: the square of the wave vector
     KX : wave-vector kx
     KY : wave-vector ky
     I : Imaginary matrix.

    RETURNS
     c: Poisson bracket {a,b} [output]
    """
    # da/dx * db/dy
    p1 = np.fft.irfftn(derivk2(KX[None,:,:],a,I[None,:,:]),axes=(1,2))
    p2 = np.fft.irfftn(derivk2(KY[None,:,:],b,I[None,:,:]),axes=(1,2))

    # da/dy * db/dx
    p3 = np.fft.irfftn(derivk2(KX[None,:,:],b,I[None,:,:]),axes=(1,2))
    p4 = np.fft.irfftn(derivk2(KY[None,:,:],a,I[None,:,:]),axes=(1,2))
    prod = quad_diff(p1,p2,p3,p4)
    
    out = np.fft.rfftn(prod,axes=(1,2))
    return dealias(out,ka2[None,:,:],kmax) 

##########################
### Run-time functions ###
##########################

def CFL_condition(ps,KX,KY,I):
    """
    Computes the time-step size.
    
    ARGUMENTS
     ps : the streamfunction
     KX : wave-vector kx
     KY : wave-vector ky
     I : Imaginary matrix.

    RETURNS
     dt : the timestep size
    """
    # Compute x and y derivatives
    vx = derivk2(KX[None,:,:],ps,I[None,:,:])
    vy = derivk2(KY[None,:,:],ps,I[None,:,:])
    # IFFT
    vx = np.fft.irfftn(vx,axes=(1,2))
    vy = np.fft.irfftn(vy,axes=(1,2))
    vel2_R = quad_plus(vx,vx,vy,vy)

    # Calculate max velocity magnitude among all ensembles
    max_vel = np.sqrt(np.max(vel2_R))
    
    dt = cfl / (kcut*max_vel + nu*kcut**(2*nn))
    
    return dt

def const_inj(ps,ka2,rng):
    """
    This subroutine assures that we inject constant energy.
    It is called when iflow == 2
    
    ARGUMENTS
     ps : streamfunction
     ka2: the square of the wave vector
     rng: random numbers

    RETURNS
     fp : forcing function
    """
    fp = np.zeros(ps.shape,dtype=Tc)
    cond = (ka2<=kup**2)&(ka2>=kdn**2)
    # Make operations passing [cond] instead of multiplyin by (cond) because the condition chooses very few modes, and hence the operation is faster in this case since it has to only multiply a few numbers of modes.
    fp[:,cond]=ps[:,cond]/(np.abs(ps[:,cond])+1.0)
    # Ensure 'realness' in the ky = 0 axis:
    fp[:,n_half:,0] = np.flip(np.conj(fp[:,1:n_half-1,0]),axis=1)
    fp[:,0,0] = fp[:,n_half-1,0] = 0.0

    # Rescale
    E = inerprod(ps,fp,1,ka2)
    # Random number
    tmp = rng.uniform(low=-1,high=1,size=ps.shape)
    tmp = np.asarray(tmp,dtype=Tf)*np.sqrt(ka2[None,:,:])

    fp *= fp0/E[:,None,None]
    fp[:,cond] += 1j*(tmp*ps)[:,cond]
    # Ensure 'realness' in the ky = 0 axis:
    fp[:,n_half:,0] = np.flip(np.conj(fp[:,1:n_half-1,0]),axis=1)
    fp[:,0,0] = fp[:,n_half-1,0] = 0.0
    
    return fp

# def rand_force(dt,ka2,ka,ka_half,rng):
#     """
#     This subroutine creates random forcing.
#     It is called when iflow == 3.
#     Based on forcing described in Chan et al. Phys. Rev. E 85, 036315 (2012) 

#     ARGUMENTS
#      dt : time step
#      ka2: the square of the wave vector
#      ka : wavenumbers
#      ka_half: half wavenumbers
#      rng: random numbers

#     RETURNS
#      fp : forcing function
#     """
#     ## Choose random vector of length kup and a random phase
#     # theta = np.arctan(ky/kx), theta between 0 and pi. Choosing this range so that ky > 0, which is the case for the rfftn in numpy/cupy.
#     theta = rng.uniform(low=0,high=numpy.pi,size=Nens)
#     # Complex phase of mode. Between -pi and pi.
#     phase = rng.uniform(low=-numpy.pi,high=numpy.pi,size=Nens)
#     # kx
#     kx = numpy.floor(kup*numpy.cos(theta)).astype(numpy.int16)
#     # ky 
#     ky = numpy.floor(kup*numpy.sin(theta)).astype(numpy.int16)

#     # Define norm
#     norm = numpy.power(n,2)*numpy.sqrt(fp0/dt)

#     # Transfer to device
#     phase = np.asarray(phase,dtype=Tf)
#     kx = np.asarray(kx,dtype=Ti)
#     ky = np.asarray(ky,dtype=Ti)

#     # Build fp
#     fp = np.zeros((Nens,n,n_half),dtype=Tc)
#     indx=np.zeros(Nens,dtype=Ti)
#     indx[kx>=0] = kx[kx>=0]
#     indx[kx<0] = n+kx[kx<0]
#     fp[np.arange(Nens),indx,ky] = norm*(np.cos(phase)+1j*np.sin(phase))/np.sqrt(ka2[None,indx,ky])
#     # Ensure 'realness' in the ky = 0 axis, but make sure not to remove the only mode that is nonzero.
#     fp[ky==0,n-indx[ky==0],0]=np.conj(fp[ky==0,indx[ky==0],0])
#     fp[:,0,0] = fp[:,n_half-1,0] = 0.0

#     return fp

def rand_force(dt,ka2,ka,ka_half,rng,cond_rand_force,counts):
    """
    This subroutine creates random forcing in the case when fractal Fourier decimation is implemented (dec_dim < 2).
    In this version, we are careful to choose forcing wavenumbers which are not zeroed out by the projection P_frac.
    It is called when iflow == 3.
    Based on forcing described in Chan et al. Phys. Rev. E 85, 036315 (2012) 

    ARGUMENTS
     dt : time step
     ka2: the square of the wave vector
     ka : wavenumbers
     ka_half: half wavenumbers
     rng: random numbers
     cond_rand_force: cumulative sum of increments at True entries, based on conditions (k2<kup**2)&(k2>kdn**2) and P_frac projection.
     counts: Number of possible wavenumber pairs (kx,ky) that are valid.

    RETURNS
     fp : forcing function
    """
    # Select a wavenumber pair for each ensemble (by multiplying counts by a random value (0,1) and rounding to the nearest int)
    r = np.asarray(rng.random(Nens),dtype=Tf)
    r = (r * counts).astype(Ti)
    # Complex phase of mode. Between -pi and pi.
    phase = np.asarray(rng.uniform(low=-numpy.pi,high=numpy.pi,size=Nens),dtype=Tf)
    
    # Now look for where this is by doing a cumsum of the condition matrix ( = cond_rand_force)
    target = r + 1  # 1-based target for cumsum (if we want position 0, we are looking for the first 'True', which will give a +1 in the cumsum)
    # Finding all of the entries which have target (between target and the next True value) then lets us choose when True happens
    equal_mask = (cond_rand_force == target[:, None])
    # Position of the chosen True within each flattened row
    pos = equal_mask.argmax(axis=1)
    # Convert flat index to 2D (i, j)
    i = pos // n_half
    j = pos % n_half

    # Define norm
    norm = numpy.power(n,2)*numpy.sqrt(fp0/dt)

    # Build fp
    fp = np.zeros((Nens,n,n_half),dtype=Tc)
    fp[np.arange(Nens),i,j] = norm*(np.cos(phase)+1j*np.sin(phase))/np.sqrt(ka2[None,i,j])
    # Ensure 'realness' in the ky = 0 axis:
    fp[:,n_half:,0] = np.flip(np.conj(fp[:,1:n_half-1,0]),axis=1)
    fp[:,0,0] = fp[:,n_half-1,0] = 0.0

    return fp

##############
### Output ###
##############

def cond_check(ps,fp,time,ka2):
    """
    Computes global quantities and saves them to a text file for time series.

    ARGUMENTS
     ps  : streamfunction
     fp  : forcing
     time: time
     ka2: the square of the wave vector

    RETURNS
     Nothing. Updates time series files.
    """
    # Energy budget
    en = energy(ps,1,ka2) # |u|^2
    if fp is None:
        inj = inerprod(ps,0.0*ps,1,ka2) # energy injection
    else:
        inj = inerprod(ps,fp,1,ka2) # energy injection
    diss = nu*energy(ps,nn+1,ka2) # dissipation
    hdiss = hnu*energy(ps,1-mm,ka2) # hypodissipation

    # Enstrophy budget
    enst = energy(ps,2,ka2) # |omega|^2
    if fp is None:
        inj_enst = inerprod(ps,0.0*ps,2,ka2) # enstrophy injection
    else:
        inj_enst = inerprod(ps,fp,2,ka2) # enstrophy injection
    diss_enst = nu*energy(ps,nn+2,ka2) # enstrophy dissipation
    hdiss_enst = hnu*energy(ps,2-mm,ka2) # enstrophy hypodissipation

    # Energy at forcing scale
    en_kf = energy(kfilt(ps,ka2[None,:,:],kup**2,2.01*kup**2),1,ka2)

    ### Save to file!
    # Open the file for appending
    with open('./energy_bal.txt', 'a') as f:
        # Write the formatted data to the file
        f.write(f"{time:23.14e} {np.nanmean(en):23.14e} {np.nanmean(inj):23.14e} {np.nanmean(diss):23.14e} {np.nanmean(hdiss):23.14e} {np.nanmean(en_kf):23.14e}\n")
    with open('./enstrophy_bal.txt', 'a') as f:
        # Write the formatted data to the file
        f.write(f"{time:23.14e} {np.nanmean(enst):23.14e} {np.nanmean(inj_enst):23.14e} {np.nanmean(diss_enst):23.14e} {np.nanmean(hdiss_enst):23.14e}\n")    
    return


def spectrum(ps,dump,ka2):
    """
    Computes the one-dimensional energy power spectrum (averaged over shells).
    
    ARGUMENTS
     ps: streamfunction
     dump: output number
     ka2: the square of the wave vector

    RETURNS
     Nothing. Saves to 'spectrum.XXXX.txt'.
    """
    # Keaton Burn's version (using histogram for shell-averaging)
    
    two = np.ones((n_half),dtype=Tf)
    two[1:] *= 2
    tmp = 1/n**4
    # Energy density
    E = en_spec_calc(ps,ka2[None,:,:],two[None,None,:])
    E *= np.pi*np.sqrt(ka2[None,:,:])*tmp # Multiply by Pi * K for integral 
    # Average over ensembles
    E = np.mean(E,axis=0)

    # Shell average of sqrt(ka2)
    bins = np.concatenate((np.array([0.0]),np.arange(1.5, n_half+1.5, 1)))
    hist_samples, _ = np.histogram(np.sqrt(ka2),bins=bins)
    
    # Shell average of E*sqrt(ka2) 
    pow_samples, _ = np.histogram(np.sqrt(ka2), bins=bins, weights=E)
    
    # E(k) = int |k| E  dtheta / int |k| dtheta
    Ek = pow_samples / hist_samples
    
    # Writes to file
    with open(odir+'/spectrum.'+f'{int(dump):04}'+'.txt', 'w') as f:
        for i in range(n_half):
            f.write(f"{Ek[i]:24.15E}\n")
        
    return

def transfers(ps,dump,ka2,KX,KY,I,inds_polar):
    """
    Computes the one-dimensional energy transfer and flux (averaged over shells).
    
    ARGUMENTS
     ps: streamfunction
     dump: output number
     ka2: the square of the wave vector
     KX : wave-vector kx
     KY : wave-vector ky
     I : Imaginary matrix.

    RETURNS
     Nothing. Saves to 'transfer.XXXX.txt' and 'fluxes.XXXX.txt'.
    """
    two = np.ones((n_half))
    two[1:] *= 2
    tmp = 1/n**4

    # Nonlinear term
    nl = laplak2(ps,ka2[None,:,:]) # Makes -w_2D
    nl = poisson(ps,nl,ka2,KX,KY,I) # Makes -curl(u_2D x w_2D)

    ### Energy and Enstrophy flux
    en_tran_tmp = en_tran_calc(ps,nl,two[None,None,:])*tmp
    enst_tran_tmp = en_tran_tmp*ka2[None,:,:]
    enst_tran = np.zeros((Nens,n_half),dtype=Tf)  
    en_tran = np.zeros((Nens,n_half),dtype=Tf)

    # Shell averaging
    for ii in range(n_half):
        rows, cols = inds_polar[ii]
        enst_tran[:,ii] = enst_tran_tmp[:,rows,cols].sum(axis=1)
        en_tran[:,ii] = en_tran_tmp[:,rows,cols].sum(axis=1)

    # Count zero as first bin
    cond = (np.round(np.sqrt(ka2)).astype(Ti)==0)
    enst_tran[:,0] += np.sum(enst_tran_tmp*cond[None,:,:],axis=(1,2))
    en_tran[:,0] += np.sum(en_tran_tmp*cond[None,:,:],axis=(1,2))
    # Ensemble average
    enst_tran = np.mean(enst_tran,axis=0)
    en_tran = np.mean(en_tran,axis=0)
        
    # Fluxes:
    pi_enst = np.cumsum(enst_tran)
    pi_en = np.cumsum(en_tran)
    
    # Writes to file
    with open(odir+'/transfer.'+f'{int(dump):04}'+'.txt', 'w') as f:
        for i in range(n_half):
            f.write(f"{enst_tran[i]:24.15E} {en_tran[i]:24.15E}\n")
        
    # Fluxes:
    pi_enst = np.cumsum(enst_tran)
    pi_en = np.cumsum(en_tran)
    with open(odir+'/fluxes.'+f'{int(dump):04}'+'.txt', 'w') as f:
        for i in range(n_half):
            f.write(f"{pi_enst[i]:24.15E} {pi_en[i]:24.15E}\n")
        
    return

def thetauuu_calc(ps,i_count,thetauuu,scriptK,rhok,rhop,rhoq,Rkpq,Tkpq,indKX,indKY,indPX,indPY,indQX,indQY,kmag,pmag,qmag):
    """
    Updates the histogram for theta and scriptK, as well as updates the count for the averaging. 

    ARGUMENTS
     ps  : streamfunction
     triads : list of triads loaded at beginning of simulation
     i_count : count for statistics
     thetauuu : the PDF of the triad phases
     scriptK : the average value and variance of the K coefficient
     rhok : the average value and variance of rhok
     rhop : the average value and variance of rhop
     rhoq : the average value and variance of rhoq
     Rkpq : the average value and variance of rhok*rhop*rhoq
     Tkpq : the average value and variance of rhok*rhop*rhoq*cos(theta)
     ind*: array used to isolate triads through dot products (to avoid for-loops).
     *mag: magnitudes of triads
    
    RETURNS
     [i_count, thetauuu, scriptK]
    """    
    Ntriads = indKX.shape[0]
    
    tmp = 1/n**2

    # Update counter:
    i_count += 1

    # Build the bin centers
    dtheta = 2*np.pi/Nbins
    bins_centered = -np.pi + dtheta/2 + dtheta*np.arange(Nbins)

    # Isolate individual triad rhos and phis
    rhos = np.abs(ps)
    rhoks = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indKX, rhos),indKY)) # Need abs to prevent rho<0 due to sgn
    rhops = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indPX, rhos),indPY)) # Need abs to prevent rho<0 due to sgn
    rhoqs = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indQX, rhos),indQY)) # Need abs to prevent rho<0 due to sgn
    phis = np.angle(ps)
    phiks = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indKX, phis),indKY)
    phips = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indPX, phis),indPY)
    phiqs = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indQX, phis),indQY)
    
    ####### thetauuu
    ## Define thetauuu
    thetas = phiks+phips+phiqs
    thetas = thetas - 2*np.pi*np.round(thetas/np.pi/2) # From [-pi,pi]
    # Remove thetas where one or more rhos == 0.
    valid_indices_ens, valid_indices_tri = np.where(np.abs(rhoks*rhoqs*rhops)>0)
    thetas_valid = thetas[valid_indices_ens,valid_indices_tri]
    # Find which 'bin' of the theta pdf it should go into and add one to the histogram
    # Shift thetas to the [0, 2pi) range
    shifted_thetas = (thetas_valid + np.pi) % (2 * np.pi)
    # Compute bin indices
    bin_indices = np.floor(shifted_thetas / dtheta).astype(int)
    # Combine bin + triad index into 1D index
    flat_idx = bin_indices * Ntriads + valid_indices_tri
    # Count with bincount (1D)
    counts = np.bincount(flat_idx, minlength=Nbins * Ntriads)
    # Reshape to (Nbins, Ntriads) and add to thetauuu
    thetauuu += counts.reshape((Nbins, Ntriads))
    
    ####### ScriptK average
    scriptK_tmp = scriptK_calc(kmag[None,:],pmag[None,:],qmag[None,:],rhoks,rhops,rhoqs)
    # Normalize based on grid
    scriptK_tmp = scriptK_tmp * tmp
    # Calculate ensemble mean
    scriptK_tmp = np.mean(scriptK_tmp,axis=0)
    
    ## Calculate the time mean of ensemble mean
    scriptK_avg_tmp = scriptK[0,:] + (scriptK_tmp - scriptK[0,:])/i_count
    
    ## Update the time variance of the ensemble mean
    scriptK[1,:] = scriptK[1,:] + ((scriptK_tmp-scriptK[0,:])*(scriptK_tmp-scriptK_avg_tmp) - scriptK[1,:])/i_count
    
    ## Update the mean
    scriptK[0,:] = scriptK_avg_tmp

    ######### rhos, R, and T averages
    # Calculate products on the GPU, and make sure to normalize based on grid
    # rhok_tmp = tmp * rhoks
    # rhop_tmp = tmp * rhops
    # rhoq_tmp = tmp * rhoqs
    # Rkpq_tmp = rhok_tmp * rhop_tmp * rhoq_tmp
    # Tkpq_tmp = Rkpq_tmp * np.cos(thetas)
    rhok_tmp,rhop_tmp,rhoq_tmp,Rkpq_tmp,Tkpq_tmp = averages_calc(rhoks,rhops,rhoqs,thetas,tmp)

    # Calculate ensemble mean
    rhok_tmp = np.mean(rhok_tmp,axis=0)
    rhop_tmp = np.mean(rhop_tmp,axis=0)
    rhoq_tmp = np.mean(rhoq_tmp,axis=0)
    Rkpq_tmp = np.mean(Rkpq_tmp,axis=0)
    Tkpq_tmp = np.mean(Tkpq_tmp,axis=0)
    
    ## Calculate the time mean of ensemble mean
    rhok_avg_tmp = rhok[0,:] + (rhok_tmp - rhok[0,:])/i_count
    rhop_avg_tmp = rhop[0,:] + (rhop_tmp - rhop[0,:])/i_count
    rhoq_avg_tmp = rhoq[0,:] + (rhoq_tmp - rhoq[0,:])/i_count
    Rkpq_avg_tmp = Rkpq[0,:] + (Rkpq_tmp - Rkpq[0,:])/i_count
    Tkpq_avg_tmp = Tkpq[0,:] + (Tkpq_tmp - Tkpq[0,:])/i_count
    
    ## Update the time variance of the ensemble mean
    rhok[1,:] = rhok[1,:] + ((rhok_tmp-rhok[0,:])*(rhok_tmp-rhok_avg_tmp) - rhok[1,:])/i_count
    rhop[1,:] = rhop[1,:] + ((rhop_tmp-rhop[0,:])*(rhop_tmp-rhop_avg_tmp) - rhop[1,:])/i_count
    rhoq[1,:] = rhoq[1,:] + ((rhoq_tmp-rhoq[0,:])*(rhoq_tmp-rhoq_avg_tmp) - rhoq[1,:])/i_count
    Rkpq[1,:] = Rkpq[1,:] + ((Rkpq_tmp-Rkpq[0,:])*(Rkpq_tmp-Rkpq_avg_tmp) - Rkpq[1,:])/i_count
    Tkpq[1,:] = Tkpq[1,:] + ((Tkpq_tmp-Tkpq[0,:])*(Tkpq_tmp-Tkpq_avg_tmp) - Tkpq[1,:])/i_count
    
    ## Update the mean
    rhok[0,:] = rhok_avg_tmp
    rhop[0,:] = rhop_avg_tmp
    rhoq[0,:] = rhoq_avg_tmp
    Rkpq[0,:] = Rkpq_avg_tmp
    Tkpq[0,:] = Tkpq_avg_tmp

    return i_count,thetauuu,scriptK,rhok,rhop,rhoq,Rkpq,Tkpq

# def corr_check(ps,time,ka2,KX,KY,I,indKX_ts,indKY_ts,indPX_ts,indPY_ts,indQX_ts,indQY_ts,kmag_ts,pmag_ts,qmag_ts,qxp_ts):
#     """
#     Calculates theta, dt(theta), the 'noise term' and scriptK for a set of triads, and writes these values to a time series file.

#     ARGUMENTS
#      ps  : streamfunction
#      time: time
#      ka2: the square of the wave vector
#      KX : wave-vector kx
#      KY : wave-vector ky
#      I : Imaginary matrix.
#      ind*: array used to isolate triads through dot products (to avoid for-loops).
#      *mag: magnitudes of triads

#     RETURNS
#      Nothing. Saves to file.
#     """
#     Ntriads_ts = indKX_ts.shape[0]

#     # Define the time series variable
#     corr_dat = np.zeros((4,Nens,Ntriads_ts),dtype=Tf)

#     # Normalization
#     tmp = 1/n**2

#     # For calculating dt(theta), to be used for the noise term.
#     nl = laplak2(ps,ka2[None,:,:]) # Makes -w_2D
#     nl = poisson(ps,nl,ka2,KX,KY,I) # Makes -curl(u_2D x w_2D)
#     dphidt = dphi_dt(ps,nl,ka2[None,:,:],kmax,I[None,:,:])

#     # Isolate individual triad rhos and phis
#     rhos = np.abs(ps)
#     rhoks = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indKX_ts, rhos),indKY_ts)) # Need abs to prevent rho<0 due to sgn
#     rhops = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indPX_ts, rhos),indPY_ts)) # Need abs to prevent rho<0 due to sgn
#     rhoqs = np.abs(np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indQX_ts, rhos),indQY_ts)) # Need abs to prevent rho<0 due to sgn
#     phis = np.angle(ps)
#     phiks = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indKX_ts, phis),indKY_ts)
#     phips = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indPX_ts, phis),indPY_ts)
#     phiqs = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indQX_ts, phis),indQY_ts)
#     dt_phiks = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indKX_ts, dphidt),indKY_ts)
#     dt_phips = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indPX_ts, dphidt),indPY_ts)
#     dt_phiqs = np.einsum('lik,ki->li',np.einsum('ij,ljk->lik', indQX_ts, dphidt),indQY_ts)

#     # Define thetauuu
#     theta = phiks + phips + phiqs
#     theta = theta - 2*np.pi*np.round(theta/np.pi/2) # From [-pi,pi]
#     corr_dat[0,:] = theta

#     # # Define triad energy
#     R_tr = np.abs(rhoks*rhops*rhoqs)*tmp**3 # Normalizing based on grid
#     corr_dat[1,:] = R_tr

#     # Define coefficient scriptK (in front of self-interaction term)
#     scriptK_tmp = scriptK_calc(kmag_ts[None,:],pmag_ts[None,:],qmag_ts[None,:],rhoks,rhops,rhoqs)
#     # Multilpy by -qxp to make it the coefficient of dt(theta)
#     scriptK_tmp = -qxp_ts[None,:] * scriptK_tmp
#     # Normalize based on grid
#     scriptK_tmp = scriptK_tmp * tmp
#     corr_dat[2,:] = scriptK_tmp  

#     # Define d theta / dt (to be used for noise calculation)
#     dt_theta = dt_phiks + dt_phips + dt_phiqs # No need to make periodic
#     corr_dat[3,:] = dt_theta

#     # Open file in append mode and write data
#     with open("triad_energy_phase.txt", "a") as f:
#         formatted_data = " ".join(f"{x:23.14E}" for x in np.hstack(([time], corr_dat.flatten())))
#         f.write(formatted_data + "\n")
    
#     return

