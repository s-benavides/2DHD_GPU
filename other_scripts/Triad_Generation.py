import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib
import glob
import scipy.optimize as spt
import tqdm 
from scipy.stats import binned_statistic_2d
from scipy.stats import binned_statistic
rng = np.random.default_rng()

plt.style.use('default')
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["font.serif"] = "Times New Roman"

# Generate two sets of triads, one for each inertial range.

### Inverse cascade range
n = 512
kf = 24
Ntriads = 5000 
kmax = kf-1
print('Energy range: n = %i, kmax = %i' % (n,kmax))

#############################
kmax_int = np.int32(np.floor(kmax))
# Choose K_min. K_min = 1 if to be used for phase_only, and K_min = 3 for full models (avoiding hypodissipation range)
K_min = 3
# Choose bounds so that we can find various scalings. We want to _at least_ have factors of 8 for all triads. This means K_lim = P_lim = int(round(kmax/8)).
K_lim = kmax_int #int(np.round(kmax/2)) # using /8 for phase only, /4 in full model?
P_lim = kmax_int #int(np.round(kmax/2))

k_vecs = []
p_vecs = []
q_vecs = []
Ks = []
Ps = []
ratio1 = []
ratio2 = []
term = []

count = 0
count_wrong_1 = 0
count_wrong_2 = 0
count_wrong_3 = 0
count_wrong_4 = 0
count_wrong_5 = 0
for Ntr in tqdm.tqdm(range(1,Ntriads+1)):
    count+=1
    ##################
    ## Choose k vector
    # Random theta
    theta_k = np.random.uniform(low=-1,high=1)*np.pi/2  # theta_k = np.arctan(kx/ky), theta between -pi/2 and pi/2. Choosing this range so that kx > 0, which is the case for us.
    # Random magnitude
    mag_k = np.random.randint(K_min,K_lim)
    kx = np.round(mag_k*np.cos(theta_k))  # kx
    ky = np.round(mag_k*np.sin(theta_k)) # ky
    kmag = np.sqrt(kx**2 + ky**2)
    
    ##################
    ## Choose a random vector p, such that: (1) K<P, (2) P<Q, (3) P < min(K + P_max,kmax), and (4) Q < kmax
    ########
    rad_lim = np.min([kmax,P_lim])
    # Choose a random radius between kmag and rad_lim
    mag_p = np.random.uniform(low=kmag+0.5,high=rad_lim)
    # Intersection of P < Q line and mag_p circle
    theta_PQ_pos = + (np.pi - np.arccos(kmag/(2*mag_p)))
    theta_PQ_neg = - (np.pi - np.arccos(kmag/(2*mag_p)))
    if (kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2))<=1: # Possible thetas will be separated
        # Intersection of Q < kmax curve and mag_p circle
        theta_kmax_pos = np.arccos((kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2)))
        theta_kmax_neg = -np.arccos((kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2)))
        pos = np.random.randint(0,high = 2) # Choose a branch
        if pos==0: # neg
            low = theta_PQ_neg
            high = theta_kmax_neg
        else: # pos
            low=theta_kmax_pos
            high=theta_PQ_pos
    else:
        low=theta_PQ_neg
        high=theta_PQ_pos
    # theta_p = np.random.uniform(low=low+angle_res,high=high-angle_res) + np.angle(kx+1j*ky)
    theta_p = np.random.uniform(low=low,high=high) + np.angle(kx+1j*ky)
    px = np.round(mag_p*np.cos(theta_p))  # px
    py = np.round(mag_p*np.sin(theta_p)) # py
    pmag = np.sqrt(px**2+py**2)
    # Finally, find qx,qy
    qx = -kx -px
    qy = -ky -py
    qmag = np.sqrt(qx**2 + qy**2)

    ####### Check and correct for discreteness effects
    cross = (kx*py)-(ky*px)
    if cross==0:
        count_wrong_1 += 1
        # print('------cross = 0')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue

    if qmag>=kmax:
        count_wrong_2 += 1
        # print('---- (qx,qy) is outside the dealiased grid! ----')
        continue

    if pmag>=kmax:
        count_wrong_3 += 1
        # print('---- (px,py) is outside the dealiased grid! ----')
        continue
    
    if qmag<pmag:
        count_wrong_4+=1
        # print('------qmag<pmag')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue
    
    if kmag>=pmag:
        count_wrong_5 += 1
        # print('------ kmag = pmag')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue

    if (kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2))<=1: 
        ratio1.append(qmag/pmag)
    else:
        ratio2.append(qmag/pmag)
    Ks.append(kmag)
    Ps.append(pmag) 
    k_vecs.append([kx,ky])
    p_vecs.append([px,py])
    q_vecs.append([qx,qy])

k_vecs = np.array(k_vecs)
p_vecs = np.array(p_vecs)


# Create a unique list of triads 
# Because many are likely duplicated
triads_combined = np.hstack([k_vecs,p_vecs])
triads_unique = np.unique(triads_combined,axis=0)

### Forward Enstrophy cascade range
kmax = 90
print('Enstrophy range: n = %i, kmax = %i' % (n,kmax))
# Choose K_min. K_min = 1 if to be used for phase_only, and K_min = 3 for full models (avoiding hypodissipation range)
K_min = kf+1

#############################
kmax_int = np.int32(np.floor(kmax))
# Choose bounds so that we can find various scalings. We want to _at least_ have factors of 8 for all triads. This means K_lim = P_lim = int(round(kmax/8)).
K_lim = kmax_int #int(np.round(kmax/2)) # using /8 for phase only, /4 in full model?
P_lim = kmax_int #int(np.round(kmax/2))

k_vecs = []
p_vecs = []
q_vecs = []
Ks = []
Ps = []
ratio1 = []
ratio2 = []
term = []

count = 0
count_wrong_1 = 0
count_wrong_2 = 0
count_wrong_3 = 0
count_wrong_4 = 0
count_wrong_5 = 0
for Ntr in tqdm.tqdm(range(1,Ntriads+1)):
    count+=1
    ##################
    ## Choose k vector
    # Random theta
    theta_k = np.random.uniform(low=-1,high=1)*np.pi/2  # theta_k = np.arctan(kx/ky), theta between -pi/2 and pi/2. Choosing this range so that kx > 0, which is the case for us.
    # Random magnitude
    mag_k = np.random.randint(K_min,K_lim)
    kx = np.round(mag_k*np.cos(theta_k))  # kx
    ky = np.round(mag_k*np.sin(theta_k)) # ky
    kmag = np.sqrt(kx**2 + ky**2)
    
    ##################
    ## Choose a random vector p, such that: (1) K<P, (2) P<Q, (3) P < min(K + P_max,kmax), and (4) Q < kmax
    ########
    rad_lim = np.min([kmax,P_lim])
    # Choose a random radius between kmag and rad_lim
    mag_p = np.random.uniform(low=kmag+0.5,high=rad_lim)
    # Intersection of P < Q line and mag_p circle
    theta_PQ_pos = + (np.pi - np.arccos(kmag/(2*mag_p)))
    theta_PQ_neg = - (np.pi - np.arccos(kmag/(2*mag_p)))
    if (kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2))<=1: # Possible thetas will be separated
        # Intersection of Q < kmax curve and mag_p circle
        theta_kmax_pos = np.arccos((kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2)))
        theta_kmax_neg = -np.arccos((kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2)))
        pos = np.random.randint(0,high = 2) # Choose a branch
        if pos==0: # neg
            low = theta_PQ_neg
            high = theta_kmax_neg
        else: # pos
            low=theta_kmax_pos
            high=theta_PQ_pos
    else:
        low=theta_PQ_neg
        high=theta_PQ_pos
    # theta_p = np.random.uniform(low=low+angle_res,high=high-angle_res) + np.angle(kx+1j*ky)
    theta_p = np.random.uniform(low=low,high=high) + np.angle(kx+1j*ky)
    px = np.round(mag_p*np.cos(theta_p))  # px
    py = np.round(mag_p*np.sin(theta_p)) # py
    pmag = np.sqrt(px**2+py**2)
    # Finally, find qx,qy
    qx = -kx -px
    qy = -ky -py
    qmag = np.sqrt(qx**2 + qy**2)

    ####### Check and correct for discreteness effects
    cross = (kx*py)-(ky*px)
    if cross==0:
        count_wrong_1 += 1
        # print('------cross = 0')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue

    if qmag>=kmax:
        count_wrong_2 += 1
        # print('---- (qx,qy) is outside the dealiased grid! ----')
        continue

    if pmag>=kmax:
        count_wrong_3 += 1
        # print('---- (px,py) is outside the dealiased grid! ----')
        continue
    
    if qmag<pmag:
        count_wrong_4+=1
        # print('------qmag<pmag')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue
    
    if kmag>=pmag:
        count_wrong_5 += 1
        # print('------ kmag = pmag')
        # print('kx',kx,'ky',ky,'px',px,'py',py)
        continue

    if (kmax**2-mag_p**2-(kx**2+ky**2))/(2*mag_p*np.sqrt(kx**2+ky**2))<=1: 
        ratio1.append(qmag/pmag)
    else:
        ratio2.append(qmag/pmag)
    Ks.append(kmag)
    Ps.append(pmag) 
    k_vecs.append([kx,ky])
    p_vecs.append([px,py])
    q_vecs.append([qx,qy])
    
k_vecs = np.array(k_vecs)
p_vecs = np.array(p_vecs)

# Create a unique list of triads 
# Because many are likely duplicated
triads_combined = np.hstack([k_vecs,p_vecs])
# ADD TO PREVIOUS
triads_unique = np.vstack([triads_unique,np.unique(triads_combined,axis=0)])

np.savetxt('./triads_%i.txt' % n, triads_unique, fmt='%5i', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
