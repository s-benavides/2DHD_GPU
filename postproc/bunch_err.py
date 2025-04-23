import numpy as np

# def bunch_err(data,err_ind = -1):
#     avg = np.nanmean(data)
#     err= np.nanstd(data)/np.sqrt(len(data))
#     avg_list = [avg]
#     err_list = [err]
#     list = np.ndarray.tolist(data)[:]
#     iter = int(np.log(len(list))/np.log(2))
#     for w in range(iter):
#         new_list=[]
#         while len(list)>1:
#             x=list.pop()
#             y=list.pop()
#             new_list.append((x+y)/2.)
#         list=new_list[:]   # the "[:]" is essential list is a copy of new_list.
#         avg_list.append(np.nanmean(list))
#         err_list.append(np.sqrt((np.sum((s - avg_list[w+1])**2 for s in list))/((float(len(list)))**2)))
#     if err_ind<0:
#         err_diff = np.diff(np.array(err_list))
#         if np.any(err_diff<0):
#             ind = np.where(err_diff<0)[0][0]
#         else:
#             ind = -1
#         error = err_list[ind]
#     else:
#         error = err_list[err_ind]

#     return error

def bunch_err(data,err_ind = -1,axis=0):
    # Order for reshaping
    if axis==0:
        order='C'
    else:
        order='F'
    
    # Length of list
    N = np.shape(data)[axis]
    
    # Take initial average and error
    avg = np.nanmean(data,axis=axis)
    err= np.nanstd(data,axis=axis)/np.sqrt(N)
    
    # Bunching algorithm
    avg_list = [avg]
    err_list = [err]
    
    dat_bunch = np.copy(data)
    iters = int(np.log2(N))
    for ii in range(iters):
        # Avg neighboring items together
        half1 = np.take(dat_bunch,np.arange(0,N,2),axis=axis)
        half2 = np.take(dat_bunch,np.arange(1,N,2),axis=axis)
        # In case they don't have the same size
        ind_max = np.min([np.shape(half1)[axis],np.shape(half2)[axis]])
        half1 = np.take(half1,np.arange(-ind_max,0),axis=axis)
        half2 = np.take(half2,np.arange(-ind_max,0),axis=axis)
        dat_bunch = np.add(half1,half2)/2
        N = np.shape(dat_bunch)[axis]
        
        # Calculate new avg and err
        new_avg = np.nanmean(dat_bunch,axis=axis)
        # print(new_avg.shape)
        new_avg_reshape = np.reshape(np.tile(new_avg,N),dat_bunch.shape,order=order)
        # print('.....',new_avg_reshape[:,0])
        new_err = np.sqrt( ( np.sum((dat_bunch - new_avg_reshape)**2,axis=axis) ) / (N**2))
        
        # Add the new mean and errors to the list
        avg_list.append(new_avg)
        err_list.append(new_err)
    
    if err_ind<0:
        err_diff = np.diff(np.array(err_list),axis=0)
        if np.array(err_list).ndim==1:
            ind = np.where(err_diff<0)[0][0]
            err_final = err_list[ind]
        else:
            indx,indy = np.where(err_diff<0)
            ind_arg = np.argsort(indy)
            indx = indx[ind_arg]
            indy= indy[ind_arg]
            err_final = np.zeros(len(err_list[0]))
            for ii in range(len(err_list[0])):
                ind_min = np.min(indx[indy==ii])
                err_final[ii] = np.array(err_list)[ind_min,ii]
    else:
        err_final = np.array(err_list)[err_ind,:]

    return err_final
