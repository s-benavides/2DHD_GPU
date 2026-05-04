"""
Use this script to remove all saved snapshots except for the latest N_save.
"""
import os,glob,pathlib
import numpy as np

idir='./'

N_save = 10

dirs = sorted(glob.glob(idir+'*Nens*'))

runs = []
for file in dirs:
    run = file.split('/')[1]
    runs.append(run)

print(runs)

for ii,run in enumerate(runs):
    path = dirs[ii]
    print("------------ Working on %s" % run)
    if os.path.exists(path+'/run/time_field.txt'):
        # Read last out
        ### Update status and copy ps file to ins folder
        tf = np.loadtxt(path+'/run/time_field.txt')
        try:
            stat = int(tf[-1][0])
            time = tf[-1][1]
        except:
            stat = int(tf[0])
            time = tf[1]
        print('Stat = %i' % stat)
        # Remove all but the last N_sve outputs
        if stat>N_save:
            for outnum in range(stat-N_save):
                #print('Deleting: %s' % (path+'/outs/ps.'+f'{int(outnum):03}'+'.npy'))
                if os.path.exists(path+'/outs/ps.'+f'{int(outnum):03}'+'.npy'):
                    print('Deleting: %s' % (path+'/outs/ps.'+f'{int(outnum):03}'+'.npy'))
                    os.remove(path+'/outs/ps.'+f'{int(outnum):03}'+'.npy')
                    os.remove(path+'/outs/ww.'+f'{int(outnum):03}'+'.npy')
        else:
            print('Five of fewer outputs.')
            continue
        
    else: 
        print('No time_field.txt file!')
        continue 
