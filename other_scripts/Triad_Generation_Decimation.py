import numpy as np
import pathlib
import glob
import scipy.optimize as spt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
rng = np.random.default_rng()

plt.style.use('default')
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.serif"] = "Times New Roman"

# We need to import P_frac from the run we're interested in creating a triad list for!
idirs = [
   # '../512_Nens_10_DPRK2_avgs_dec_1d75_v2_triads/',
   '../512_Nens_10_DPRK2_avgs_dec_1d85_v2_triads/',
    # '../512_Nens_10_DPRK2_avgs_dec_1d95_triads/',
    ]

for idir_dir in idirs:
    dec_dim_name = idir_dir.split('_')[6]
    dec_dim = float((dec_dim_name).replace('d','.'))
    print(" ----- Working on %s ----- " % dec_dim_name,flush=True)
    
    # Load P_frac
    P_frac = np.load(idir_dir+'/ins/P_frac.npy')
    Nens,n,n_half = P_frac.shape
    
    ka = np.fft.fftfreq(n,d=(1/n)) # kx
    ka_half = np.fft.rfftfreq(n,d=(1/n)) # ky
    KX,KY = np.meshgrid(ka,ka_half,indexing='ij')
    ka2 = KX**2+KY**2
    
    # All are copies so we keep only one
    P_final = P_frac[1,:,:]

    phase_only = False

    Ntriads = 800000
    mult = n//256
    if phase_only:
        kmax = n//3
    else:
        # kmax = 12*mult # mid-range
        kmax = 36*mult # small-scale (hypo/hyper diss)
    print('n = %i, kmax = %i' % (n,kmax),flush=True)
    
    uniform_radii=True
    
    #############################
    kmax_int = np.int32(np.floor(kmax))
    # Choose K_min. K_min = 1 if to be used for phase_only, and K_min = 4 for full models (avoiding hypodissipation range)
    K_min = 4
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
    
    ##################
    ## Choose k vectors
    # First flatten:
    KX_flat = KX.reshape(n*n_half)
    KY_flat = KY.reshape(n*n_half)
    # Weight matrix (NOTE: can change this to have uniform kmag instead of uniform over 2D)
    # Uniform weights
    P_flat = (P_final*(ka2<K_lim**2)*(ka2>K_min**2)).reshape(n*n_half) # Conditions that must be true
    if uniform_radii:
        P_flat[1:] /= (np.sqrt(ka2).reshape(n*n_half))[1:]
    weights = P_flat/np.sum(P_flat) 
    # Randomly choose Ntriads k vectors
    inds = np.arange(n*n_half)
    inds_triads = rng.choice(inds,size=Ntriads,p=weights)
    kxs = KX_flat[inds_triads] # Shape (Ntriads)
    kys = KY_flat[inds_triads]
    kmags = np.sqrt(kxs**2 + kys**2)
    
    for ii in range(Ntriads):
        ##################
        ## Choose a random vectors p, such that: (1) K<P, (2) P<Q, (3) P < min(K + P_max,kmax), and (4) Q < kmax
        ########
        # This is a bit more challenging because each randomly chosen p will correspond to a (kx,ky) combination from the list
        # Meaning that the condition for each (px,py) will be different for each index.
        # Limits on radius, kmag<P<kmax
        kmag = kmags[ii]
        kx = kxs[ii]
        ky = kys[ii]
        cond_pre = P_final*(ka2<K_lim**2)*(ka2>(kmag+0.5)**2)
        # P<Q<kmax
        qa2_tmp = (KX+kx)**2 + (KY+ky)**2 # Possible q^2 
        cond_pre *= (ka2<qa2_tmp)*(qa2_tmp<K_lim**2) # Limits on Q
        if np.all(cond_pre==0.0):
            continue
        P_flat = (cond_pre).reshape(n*n_half) # Conditions that must be true
        if uniform_radii:
            P_flat[1:] /= (np.sqrt(ka2).reshape(n*n_half))[1:]
        weights = P_flat/np.sum(P_flat) 
        # Randomly choose one p vector per k vector
        inds_triads = rng.choice(inds,p=weights)
        px = KX_flat[inds_triads] # Shape (Ntriads)
        py = KY_flat[inds_triads]
        pmag = np.sqrt(px**2 + py**2)
    
        # Finally, find qx,qy
        qx = -kx -px
        qy = -ky -py
        qmag = np.sqrt(qx**2 + qy**2)
    
        
        cross = (kx*py)-(ky*px)
        wrong_now= False
        # Check to see if q is also on the grid
        if (qy<0):
            qy=-qy
            qx=-qx
        ind_q = np.where((qx==KX_flat)&(qy==KY_flat))[0][0]
        if P_final.reshape(n*n_half)[ind_q]==0.0:
            wrong_now = True
            continue
            
        if cross==0:
            count_wrong_1 += 1
            wrong_now = True
            continue
    
        if qmag>=kmax:
            count_wrong_2 += 1
            wrong_now = True
            continue
    
        if pmag>=kmax:
            count_wrong_3 += 1
            wrong_now = True
            continue
        
        if qmag<pmag:
            count_wrong_4+=1
            wrong_now = True
            continue
        
        if kmag>=pmag:
            count_wrong_5 += 1
            wrong_now = True
            continue
    
        Ks.append(kmag)
        Ps.append(pmag) 
        k_vecs.append([kx,ky])
        p_vecs.append([px,py])
        q_vecs.append([qx,qy])
        
    k_vecs = np.array(k_vecs)
    p_vecs = np.array(p_vecs)
    triads_combined = np.hstack([k_vecs,p_vecs])
    triads_unique = np.unique(triads_combined,axis=0)
    k_vecs_unique = np.unique(k_vecs,axis=0)
    p_vecs_unique = np.unique(p_vecs,axis=0)
    q_vecs_unique = np.unique(q_vecs,axis=0)
    
    inds_triads = np.arange(triads_unique.shape[0])
    inds_triads_subset = inds_triads # Full
    
    #############################
    kmax_int = np.int32(np.floor(kmax))
    
    triads_full = []
    as_full = []
    count = 1
    for triad in triads_unique[:,:]:
    # for triad in triads_unique[inds_triads_subset,:]:
        [qx,qy]=[-triad[0]-triad[2],-triad[1]-triad[3]]
        qmax = np.max([np.abs(qx),np.abs(qy)])
        nums = np.min([8,int(np.floor(kmax/qmax))])
        for ii in range(1,nums+1):
            ### Scale triad
            triad_tmp = triad*(ii)
            kx_tmp,ky_tmp = triad_tmp[:2]
            px_tmp,py_tmp = triad_tmp[2:]
            qx_tmp = -kx_tmp-px_tmp
            qy_tmp = -ky_tmp-py_tmp
    
            # Make sure they are on grid
            if (ky_tmp<0):
                ky_tmp=-ky_tmp
                kx_tmp=-kx_tmp
            if (py_tmp<0):
                py_tmp=-py_tmp
                px_tmp=-px_tmp
            if (qy_tmp<0):
                qy_tmp=-qy_tmp
                qx_tmp=-qx_tmp
            
            kmag = np.sqrt(kx_tmp**2 + ky_tmp**2)
            pmag = np.sqrt(px_tmp**2 + py_tmp**2)
            qmag = np.sqrt((kx_tmp+px_tmp)**2 + (ky_tmp+py_tmp)**2)
            if (kmag>kmax) or (pmag>kmax) or (qmag>kmax):
                continue
            else:
                ### Confirm triad is in P_frac
                ikx = np.where(ka==kx_tmp)[0][0]
                jky = np.where(ka_half==ky_tmp)[0][0]
                ipx = np.where(ka==px_tmp)[0][0]
                jpy = np.where(ka_half==py_tmp)[0][0]
                iqx = np.where(ka==qx_tmp)[0][0]
                jqy = np.where(ka_half==qy_tmp)[0][0]
                included = P_final[ikx,jky]*P_final[ipx,jpy]*P_final[iqx,jqy]
                if included==1.0:
                    triads_full.append(triad*(ii))
                    as_full.append(ii)

    
    triads_full = np.array(triads_full)
    as_full = np.array(as_full)
    
    print('Number of random set of "mother" triads: %i' % (triads_unique[inds_triads_subset,:].shape[0]),flush=True)
    print('Number of triads, including scaled: %i' % (triads_full.shape[0]),flush=True)
    print('Max scaling ("a") present in set: %i' % (np.max(as_full)),flush=True)
    # Remove all triads that just have a = 1 and that's it.
    as_final = as_full[:-1][np.abs(np.diff(as_full))>0]
    triads_final =  triads_full[:-1][np.abs(np.diff(as_full))>0]
    print('Final count = %i' % len(as_final),flush=True)

    if (len(as_final)>8000):
        # Removing pairs of 1 2 so we have a number close to 3's (or something a bit lower than current)
        inds_1 = np.where(as_final==1.0)[0]
        inds_pairs = []
        for ind in inds_1:
            if (ind+2)<len(as_final):
                if (as_final[ind+1]==2.0)&(as_final[ind+2]==1.0):
                    inds_pairs.append(ind)
        
        # Now choose some random set to remove
        inds_pairs_del_1 = rng.choice(inds_pairs,size=(len(inds_pairs)-np.sum(as_final==3)),replace=False) # Just 1 locations
        inds_pairs_del = np.concatenate([inds_pairs_del_1,inds_pairs_del_1+1]) # Now the 2 locations
        triads_final = np.delete(triads_final,inds_pairs_del,axis=0)
        as_final = np.delete(as_final,inds_pairs_del)
        print('Final count after removing random set of 1,2 pairs = %i' % np.shape(as_final),flush=True)

    runname = (idir_dir).split('avgs')[1][1:-1]
    if phase_only:
        name='phase_only'
    else:
        name='full'

    np.savetxt('./triad_lists/'+name+'/triads_%i_%s.txt' % (n,runname), triads_final, fmt='%5i', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
    np.savetxt('./triad_lists/'+name+'/as_%i_%s.txt' % (n,runname), as_final, fmt='%5i', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

    # Histogram
    q = as_final
    d = np.diff(np.unique(q)).min()
    left_of_first_bin = q.min() - float(d)/2
    right_of_last_bin = q.max() + float(d)/2
    # n, bins, patches = axs[jj,1].hist(q, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d),histtype='step',log=True,density=True,lw=2,
    #         color=cm.copper(cscale_qins(q_in/Ny,q_ins_Ny[2:5])))
    nn, bins = np.histogram(q, bins=np.arange(left_of_first_bin, right_of_last_bin + d, d),density=False)
    bins_c = (np.diff(bins)/2+bins[:-1])
    # nn[nn==0]=np.nan
    plt.semilogy(bins_c[nn!=0],nn[nn!=0],'.-',lw=2,)
    # plt.ylim(0,np.max(nn)*1.1)
    # plt.xlim(0,10)
    plt.ylabel('Histogram')
    plt.xlabel('Scaling, $a$')
    plt.savefig('./figs/triad_list_hist'+runname+'.png',dpi=150,bbox_inches='tight')
