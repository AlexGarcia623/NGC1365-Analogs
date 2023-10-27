import sys
import os
import time
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py
import illustris_python as il
from scipy.optimize import curve_fit

run = 'L35n2160TNG' #TNG50 - 1
# 
# base = '/orange/paul.torrey/zhemler/IllustrisTNG/' + run + '/' # Zach's directory
# out_dir = base + 'output/' # needed for Zach's directory

base = '/orange/paul.torrey/IllustrisTNG/Runs/' + run  # Paul's directory
out_dir = base


treeBase = '/orange/paul.torrey/IllustrisTNG/Runs/' + run #+'/postprocessing/'  # Paul's directory
# out_dir = base

snaps = [98] 

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

m_star_min = 8.5
m_star_max = 11.5
m_gas_min  = 8.5

def merger(run, base, out_dir, snap, treeBase):
    snap = snaps[0]
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    print scf
    z0       = (1.00E+00 / scf - 1.00E+00)
    print z0

    fields = ['SubhaloGasMetallicity', 'SubhaloPos', 'SubhaloMass', 'SubhaloVel', 'SubhaloSFR',
              'SubhaloMassType','SubhaloGasMetallicitySfr','SubhaloHalfmassRadType','SubhaloGrNr']
    sub_cat = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)

    mass = sub_cat['SubhaloMassType'] * 1.00E+10/h
    
    
#     subs = [526029, 526478, 537236]
#     names = ['TNG0052','TNG0053','TNG0070']
    
#     subs = [526478] #snap99
    subs = [524093,524098] #snap98
#     subs = [514617] #snap97
#     subs = [511757] #snap96
#     subs = [510184] #snap95
    names = ['TNG0053']
    plt.clf()
    i=0
    d1 = []
    d2 = []
    for sub in subs:
        single = il.groupcat.loadSingle(out_dir, snap, subhaloID = sub)
        
        if sub == subs[0]:
            d1 = single['SubhaloPos']*(scf/h)
        else:
            d2 = single['SubhaloPos']*(scf/h)
            
        print(single['SubhaloMassType'][4]*1.00E+10/h,single['SubhaloMassType'][1]*1.00E+10/h,str(sub))
        print(single['SubhaloMassType'][0]*1.00E+10/h+single['SubhaloMassType'][4]*1.00E+10/h,str(sub))
                    
    print(np.sqrt(np.sum((d1-d2)**2)))
    
    
merger(run, base, out_dir, snaps, treeBase)
