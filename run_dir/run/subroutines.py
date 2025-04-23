import cupy as np
import numpy
# import numpy as np
from parameter import *
import warnings

######################
### Custom Kernels ###
######################
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
   'float64 K, complex128 a,complex128 I',
   'complex128 b',
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
   'complex128 a,float64 ka2',
   'complex128 b',
   'b = -ka2*a',
   'laplak2')

# """
# Multiplies and subtracts four matrices (for poission). Generic type so that it can be used with real or complex arrays.
# """
quad_diff = np.ElementwiseKernel(
   'T a,T b,T c,T d',
   'T e',
   'e = a*b-c*d',
   'quad_diff')

# """
# Multiplies and adds four matrices (for poission). Generic type so that it can be used with real or complex arrays.
# """
quad_plus = np.ElementwiseKernel(
   'T a,T b,T c,T d',
   'T e',
   'e = a*b+c*d',
   'quad_plus')

# """
# Dealiasing
# """
dealias = np.ElementwiseKernel(
    'complex128 a, float64 ka2, float64 kmax',
    'complex128 b',
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
    'complex128 a, float64 ka2, float64 kmin, float64 kmax',
    'complex128 b',
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
   'complex128 ps,complex128 nl,complex128 fp, float64 tmp1, float64 nu, float64 hnu, float64 nn, float64 mm, float64 ka2, float64 kmax',
   'complex128 out',
   """
    if (ka2 > kmax || ka2 == 0) {
       out = 0;
    } else {
       out = (ps + ((-nl)/ka2+fp)*tmp1)/(1.0 +(nu*pow(ka2, nn) + hnu*pow(ka2, -mm))*tmp1);
    }
   """,
   'NL')

# """
# Used in corr_check
# """
dphi_dt = np.ElementwiseKernel(
   'complex128 ps,complex128 nl,float64 ka2, float64 kmax, complex128 I',
   'float64 out',
   """
    if (ka2 > kmax || ka2 == 0 || abs(ps) == 0 ) {
       out = 0;
    } else {
       out = imag(exp(-I*atan2(imag(ps), real(ps)))*(-nl/ka2)/abs(ps));
    }
   """,
   'dphi_dt')

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
    two = np.ones((n_half))
    two[1:] *= 2

    # We suppress warnings here because, if kin<0, the [0,0] element will be nan, since ka2 = 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warnings in this block
        E = np.nansum(two[None,:]*np.abs(C)**2*ka2**kin)
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
    two = np.ones((n_half))
    two[1:] *= 2

    # We suppress warnings here because, if kin<0, the [0,0] element will be nan, since ka2 = 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warnings in this block
        rslt = np.real(np.nansum((two[None,:]*ka2**kin*a*np.conj(b))))
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

    RETURNS
     c: Poisson bracket {a,b} [output]
    """
    # da/dx * db/dy
    p1 = np.fft.irfftn(derivk2(KX,a,I))
    p2 = np.fft.irfftn(derivk2(KY,b,I))

    # da/dy * db/dx
    p3 = np.fft.irfftn(derivk2(KX,b,I))
    p4 = np.fft.irfftn(derivk2(KY,a,I))
    prod = quad_diff(p1,p2,p3,p4)
    # prod = (prod - np.fft.irfftn(dx)*np.fft.irfftn(dy))  
    
    out = np.fft.rfftn(prod)
    return dealias(out,ka2,kmax) 

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
    vx = derivk2(KX,ps,I)
    vy = derivk2(KY,ps,I)
    vx = np.fft.irfftn(vx)
    vy = np.fft.irfftn(vy)
    # IFFT
    vel2_R = quad_plus(vx,vx,vy,vy)

    # Calculate max velocity magnitude
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
    fp = np.zeros(ps.shape,dtype=complex)
    cond = (ka2<=kup**2)&(ka2>=kdn**2)
    # Make operations passing [cond] instead of multiplyin by (cond) because the condition chooses very few modes, and hence the operation is faster in this case since it has to only multiply a few numbers of modes.
    fp[cond]=ps[cond]/(np.abs(ps[cond])+1.0)
    # Ensure 'realness' in the kx = 0 axis:
    fp[n_half:,0] = np.flip(np.conj(fp[1:n_half-1,0]))
    fp[0,0] = fp[n_half-1,0] = 0.0

    # Rescale
    E = inerprod(ps,fp,1,ka2)
    # Random number
    tmp = rng.uniform(low=-1,high=1,size=ps.shape)
    tmp = np.asarray(tmp)*np.sqrt(ka2)
    
    fp[cond] = fp[cond]*fp0/E + 1j*tmp[cond]*ps[cond]
    # Ensure 'realness' in the kx = 0 axis:
    fp[n_half:,0] = np.flip(np.conj(fp[1:n_half-1,0]))
    fp[0,0] = fp[n_half-1,0] = 0.0
    
    return fp

def rand_force(dt,ka2,ka,ka_half,rng):
    """
    This subroutine creates random forcing.
    It is called when iflow == 3.
    Based on forcing described in Chan et al. Phys. Rev. E 85, 036315 (2012) 

    ARGUMENTS
     dt : time step
     ka2: the square of the wave vector
     rng: random numbers

    RETURNS
     fp : forcing function
    """
    ## Choose random vector of length kup and a random phase
    # theta = np.arctan(kx/ky), theta between -pi/2 and pi/2. Choosing this range so that kx > 0, which is the case for us.
    theta = rng.uniform(low=-numpy.pi/2,high=numpy.pi/2)
    # Complex phase of mode. Between -pi and pi.
    phase = rng.uniform(low=-numpy.pi,high=numpy.pi)
    # kx
    kx = numpy.floor(kup*numpy.cos(theta)).astype(numpy.int32)
    # ky 
    ky = numpy.floor(kup*numpy.sin(theta)).astype(numpy.int32)

    # Define norm
    norm = numpy.power(n,2)*numpy.sqrt(fp0/dt)

    # Build fp
    fp = np.zeros((n,n_half),dtype=np.complex128)
    if ky>=0:
        indy = ky
    else:
        indy = n+ky
    fp[indy,kx] = norm*(np.cos(phase)+1j*np.sin(phase))/np.sqrt(ka2[indy,kx])
    # Ensure 'realness' in the kx = 0 axis, but make sure not to remove the only mode that is nonzero.
    if kx==0:
        fp[n-indy,0]=np.conj(fp[indy,0])
    fp[0,0] = fp[n_half-1,0] = 0.0

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
    inj = inerprod(ps,fp,1,ka2) # energy injection
    diss = nu*energy(ps,nn+1,ka2) # dissipation
    hdiss = hnu*energy(ps,1-mm,ka2) # hypodissipation

    # Enstrophy budget
    enst = energy(ps,2,ka2) # |omega|^2
    inj_enst = inerprod(ps,fp,2,ka2) # enstrophy injection
    diss_enst = nu*energy(ps,nn+2,ka2) # enstrophy dissipation
    hdiss_enst = hnu*energy(ps,2-mm,ka2) # enstrophy hypodissipation

    # Energy at forcing scale
    en_kf = energy(kfilt(ps,ka2,kup**2,2.01*kup**2),1,ka2)

    ### Save to file!
    # Open the file for appending
    with open('./energy_bal.txt', 'a') as f:
        # Write the formatted data to the file
        f.write(f"{time:23.14e} {en:23.14e} {inj:23.14e} {diss:23.14e} {hdiss:23.14e} {en_kf:23.14e}\n")
    with open('./enstrophy_bal.txt', 'a') as f:
        # Write the formatted data to the file
        f.write(f"{time:23.14e} {enst:23.14e} {inj_enst:23.14e} {diss_enst:23.14e} {hdiss_enst:23.14e}\n")    
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
    
    two = np.ones((n_half))
    two[1:] *= 2
    tmp = 1/n**4
    # Energy density
    E = np.pi*np.sqrt(ka2)*two[None,:]*np.abs(ps)**2*ka2*tmp # Multiply by Pi * K for integral 

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

def transfers(ps,dump,ka2,KX,KY,I):
    """
    Computes the one-dimensional energy transfer and flux (averaged over shells).
    
    ARGUMENTS
     ps: streamfunction
     dump: output number
     ka2: the square of the wave vector
     KX : wave-vector kx
     KY : wave-vector ky

    RETURNS
     Nothing. Saves to 'transfer.XXXX.txt' and 'fluxes.XXXX.txt'.
    """
    
    two = np.ones((n_half))
    two[1:] *= 2
    tmp = 1/n**4
    
    # Nonlinear term
    nl = laplak2(ps,ka2) # Makes -w_2D
    nl = poisson(ps,nl,ka2,KX,KY,I) # Makes -curl(u_2D x w_2D)
    
    ### Enstrophy flux
    enst_tran_tmp = two[None,:]*ka2*np.real(ps*np.conj(nl))*tmp 
    enst_tran = np.zeros((n_half))  
    
    ### Energy flux
    en_tran_tmp = two[None,:]*np.real(ps*np.conj(nl))*tmp
    en_tran = np.zeros((n_half))
    
    # Shell averaging
    for ii in range(n_half):
        kk = ii+1
        enst_tran[ii] = np.sum(enst_tran_tmp[np.round(np.sqrt(ka2)).astype(np.int64)==kk])
        en_tran[ii] = np.sum(en_tran_tmp[np.round(np.sqrt(ka2)).astype(np.int64)==kk])
    
    # Count zero as first bin
    enst_tran[0] += np.sum(enst_tran_tmp[np.round(np.sqrt(ka2)).astype(np.int64)==0])
    en_tran[0] += np.sum(en_tran_tmp[np.round(np.sqrt(ka2)).astype(np.int64)==0])
    
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
