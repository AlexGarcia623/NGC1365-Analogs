# -*- coding: utf-8 -*-

import sys
import os
import h5py
import numpy                as np
import matplotlib           as mpl
mpl.use('agg')
mpl.rcParams['text.usetex']        = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
import matplotlib.pyplot as plt
import illustris_python as il

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

def center(pos0, centpos, boxsize = None):
    pos       = np.copy(pos0)
    pos[:,0] -= centpos[0]
    pos[:,1] -= centpos[1]
    pos[:,2] -= centpos[2]
    if (boxsize != None):
        pos[:,0][pos[:,0] < (-boxsize / 2.000E+00)] += boxsize 
        pos[:,0][pos[:,0] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,1][pos[:,1] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,1][pos[:,1] > ( boxsize / 2.000E+00)] -= boxsize
        pos[:,2][pos[:,2] < (-boxsize / 2.000E+00)] += boxsize
        pos[:,2][pos[:,2] > ( boxsize / 2.000E+00)] -= boxsize
    return pos
    
def calcrsfrio(pos0, sfr0):
    fraci = 5.000E-02
    fraco = 9.000E-01
    r0    = 1.000E+01
    rpos  = np.sqrt(pos0[:,0]**2.000E+00 + 
                    pos0[:,1]**2.000E+00 +
                    pos0[:,2]**2.000E+00 )
    sfr   = sfr0[np.argsort(rpos)]
    rpos  = rpos[np.argsort(rpos)]
    
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan
    sfrf   = np.cumsum(sfr)/sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxi   = idx0[(sfrf > fraci)]
    idxi   = idxi[0]
    rsfri  = rpos[idxi]
        
    dskidx = rpos < (rsfri + r0)
    sfr    =  sfr[dskidx]
    rpos   = rpos[dskidx]
    
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan
        
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxo   = idx0[(sfrf > fraco)]
    idxo   = idxo[0]
    rsfro  = rpos[idxo]
    return rsfri, rsfro
    
def calcincl(pos0, vel0, m0, ri, ro):
    rpos = np.sqrt(pos0[:,0]**2.000E+00 + 
                   pos0[:,1]**2.000E+00 +
                   pos0[:,2]**2.000E+00 )
    idx  = (rpos > ri) & (rpos < ro)
    pos  = pos0[idx]
    vel  = vel0[idx]
    m    =   m0[idx]
    
    hl = np.cross(pos, vel)
    L  = np.array([np.multiply(m, hl[:,0]), 
                   np.multiply(m, hl[:,1]), 
                   np.multiply(m, hl[:,2])])
    L  = np.transpose(L)
    L  = np.array([np.sum(L[:,0]),
                   np.sum(L[:,1]),
                   np.sum(L[:,2])])
    Lmag  = np.sqrt(L[0]**2.000E+00 +
                    L[1]**2.000E+00 +
                    L[2]**2.000E+00 )
    Lhat  = L / Lmag
    incl  = np.array([np.arccos(Lhat[2]), np.arctan2(Lhat[1], Lhat[0])])
    incl *= 1.800E+02 / np.pi
    if   incl[1]  < 0.000E+00:
         incl[1] += 3.600E+02
    elif incl[1]  > 3.600E+02:
         incl[1] -= 3.600E+02
    return incl
    
def dskcut(pos0, m0, rho0, T0, zcut = None):
    if (zcut != None):
        didx = (np.log10(T0) < (6.000E+00 + 2.500E-01 * np.log10(rho0))) & (np.abs(pos0[:,2]) < zcut)
    else:
        didx = np.log10(T0) < (6.000E+00 + 2.500E-01 * np.log10(rho0))
    return didx
    
def trans(arr0, incl0):
    arr      = np.copy( arr0)
    incl     = np.copy(incl0)
    deg2rad  = np.pi / 1.800E+02
    incl    *= deg2rad
    arr[:,0] = -arr0[:,2] * np.sin(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.cos(incl[0])
    arr[:,1] = -arr0[:,0] * np.sin(incl[1]) + (arr0[:,1] * np.cos(incl[1])                                                )
    arr[:,2] =  arr0[:,2] * np.cos(incl[0]) + (arr0[:,0] * np.cos(incl[1]) + arr0[:,1] * np.sin(incl[1])) * np.sin(incl[0])
    del incl
    return arr

def calcpix(gaspos, starpos, mgas, mstar, hrho, radvel, sfr, zm, zm9 = True):
    rmax   = +5.000E+01
    #pixl   = +2.500E-01
    pixl   = 1.000E-01
    # hcut   = -8.860E-01
    hcut   = -1.301E+00
    #hcut   = -2.000E+00
    pixa   = pixl**2.000E+00
    
    pixlims = np.arange(-rmax, rmax + pixl, pixl)
    pix     = len(pixlims) - 1
    pixcs   = pixlims[:-1] + (pixl / 2.000E+00)
    
    hidx = np.log10(hrho) > hcut
    xymstar, x, y = np.histogram2d(starpos[:,0], starpos[:,1], weights =       mstar, bins = [pixlims, pixlims])
    xymgas , x, y = np.histogram2d( gaspos[:,0],  gaspos[:,1], weights =        mgas, bins = [pixlims, pixlims])
    xysfr  , x, y = np.histogram2d( gaspos[:,0],  gaspos[:,1], weights =         sfr, bins = [pixlims, pixlims])
    xyprad , x, y = np.histogram2d( gaspos[:,0],  gaspos[:,1], weights = radvel*mgas, bins = [pixlims, pixlims])
    if (zm9):
        xyo, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx,4]), bins = [pixlims, pixlims])
        xyh, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx,0]), bins = [pixlims, pixlims])
    else:
        xh = 7.600E-01
        zo = 3.500E-01
        xyh, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = mgas[hidx] * xh                       , bins = [pixlims, pixlims])
        xyo, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx]) * zo, bins = [pixlims, pixlims])
    
    xymstar = np.transpose(xymstar)
    xymgas  = np.transpose( xymgas)
    xysfr   = np.transpose(  xysfr)
    xyo     = np.transpose(    xyo)
    xyh     = np.transpose(    xyh)
    xyprad  = np.transpose( xyprad)
    xymstar =     np.ravel(xymstar)
    xymgas  =     np.ravel( xymgas)
    xysfr   =     np.ravel(  xysfr)
    xyo     =     np.ravel(    xyo)
    xyh     =     np.ravel(    xyh)
    xyprad  =     np.ravel( xyprad)
    
    xymgas  /= pixa
    xymstar /= pixa
    xysfr   /= pixa
    xyo     /= pixa
    xyh     /= pixa
    xyprad  /= pixa
    
    return xymgas, xymstar, xysfr, xyo, xyh, xyprad

run = 'L35n2160TNG' #TNG50 - 1

base = '/orange/paul.torrey/zhemler/IllustrisTNG/' + run + '/' # Zach's directory
outdir = base + 'output/' # needed for Zach's directory

snap0 = 99
subs  = [526029, 526478, 537236]
names = ['TNG0052','TNG0053','TNG0070']

# lowmass = [14286 18517 21549 26704 27873 30634 33084 34981 37062 39082 40427 41086
#  42584 43887 44689 46227 46839 48060 48446 49084 49760 50327 51735 53450
#  55210 56184 57100 58442 58852 59237 60916 62302 65350 65760 71696 72410
#  72848 74078 74817 78684 81595 82393 85690 87339]

# highmass = [12907 14286 18517 20563 21549 26704 27873 33084 37062 39082 39760 40427
#  41086 42584 43887 44689 46227 46839 48060 48446 49760 50327 51735 53450
#  55210 56184 58442 58852 59237 60916 62302 65350 65760 71696 74078 85690]

# mpb    = il.sublink.loadTree(outdir, snap0, sub0, fields = ['SubhaloMass','SubfindID','SnapNum'], onlyMPB = True)
# msubid = mpb['SubfindID'  ]
# msnap  = mpb['SnapNum'    ]
# mmass  = mpb['SubhaloMass'] * (1.000E+10 / h)

xwdth    = 1.000E+02
# pixl     = 2.500E-01
pixl     = 1.00E-01
pixa     = pixl**2
rmax_tng = xwdth / 2.000E+00
dr       = 1.000E-01
pixlims  = np.arange(-rmax_tng, rmax_tng + pixl, pixl)
pix      = len(pixlims) - 1
pixcs    = pixlims[:-1] + (pixl / 2.000E+00)
rpix     = np.full((pix, pix), np.nan, dtype = float)
for r in range(0, pix):
    for c in range(0, pix):
        rpix[r,c] = np.sqrt(pixcs[r]**2 + pixcs[c]**2)
rpix = np.ravel(rpix)
rtng = np.arange(dr, rmax_tng, dr)
xcs, ycs = np.meshgrid(pixcs, pixcs)

h5f       = h5py.File('NGC1365Analogs.hdf5', 'w')

# nsnap      = len(msnap)
nrtng      = len( rtng)

# z0s        = np.full((nsnap), np.nan, dtype = float)

nsnap = 1

for i in range(len(subs)):
    print 'Starting %s' %names[i]
    map_mgass  = np.full((pix, pix), np.nan, dtype = float)
    map_mstars = np.full((pix, pix), np.nan, dtype = float)
    map_sfrs   = np.full((pix, pix), np.nan, dtype = float)
    map_os     = np.full((pix, pix), np.nan, dtype = float)
    map_hs     = np.full((pix, pix), np.nan, dtype = float)
    map_vr     = np.full((pix, pix), np.nan, dtype = float)
    map_pr     = np.full((pix, pix), np.nan, dtype = float)
    snap = snap0
    sub  = subs[i]
    name = names[i]
    
    h5map     = h5f.create_group(name)

#     print snap
#     if (snap < 25):
#         continue
    
    hdr = il.groupcat.loadHeader(outdir,snap)
    cat = il.groupcat.loadSubhalos(outdir,snap,fields=['SubhaloMassType', 'SubhaloSFR', 'SubhaloPos', 'SubhaloVel'])
    boxsize = hdr['BoxSize']
    scf     = hdr['Time']
    z0      = (1.000E+00 / scf - 1.000E+00)
    cat['SubhaloMassType'][       :, :] *= (1.00E+10 / h)
    submstar = np.log10(cat['SubhaloMassType'][sub,4])
    subsfr   = np.log10(cat['SubhaloSFR'][sub])
#     z0s[i]   = z0
    
    subpos   = cat['SubhaloPos'][sub]
    subvel   = cat['SubhaloVel'][sub]

    gasid    = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['ParticleIDs'      ])
    gaspos   = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['Coordinates'      ])
    gasvel   = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['Velocities'       ])
    gasmass  = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['Masses'           ])
    gassfr   = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['StarFormationRate'])
    gaszm    = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    gasrho   = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['Density'          ])
    gasxe    = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['ElectronAbundance'])
    gasu     = il.snapshot.loadSubhalo(outdir, snap, sub, 0, fields = ['InternalEnergy'   ])
    starid   = il.snapshot.loadSubhalo(outdir, snap, sub, 4, fields = ['ParticleIDs'      ])
    starpos  = il.snapshot.loadSubhalo(outdir, snap, sub, 4, fields = ['Coordinates'      ])
    starmass = il.snapshot.loadSubhalo(outdir, snap, sub, 4, fields = ['Masses'           ])
    bhid     = il.snapshot.loadSubhalo(outdir, snap, sub, 5, fields = ['ParticleIDs'      ])
    
    gaspos   = center(gaspos, subpos, boxsize)
    starpos  = center(starpos, subpos, boxsize)
    gaspos  *= (scf / h)
    starpos *= (scf / h)
    gasvel  *= np.sqrt(scf)
    gasvel  -= subvel
    gasmass *= (1.000E+10 / h)
    starmass *= (1.000E+10 / h)
    gasrho0  = gasrho / (scf)**3.00E+00
    gasrho  *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gasrho  *= (1.989E+33) / (3.086E+21**3.00E+00)
    gasrho  *= xh / mh
    gasmu    = 4.000E+00 / (1.000E+00 + 3.000E+00 * xh + 4.000E+00 * xh * gasxe)
    gastemp  = ((5.000E+00 / 3.000E+00) - 1.000E+00) * (gasu / kb) * (1.000E+10) * gasmu * mh
    gasoh    = np.log10((zo * gaszm / xh) * (1.000E+00 / 1.600E+01)) + 1.200E+01
    
    ri, ro = calcrsfrio(gaspos, gassfr)
    ro2    = 2.000E+00 * ro
        
    dskidx = dskcut(gaspos, gasmass, gasrho0, gastemp)
    incl   = calcincl(gaspos[dskidx], gasvel[dskidx], gasmass[dskidx], ri, ro)
    
    gaspos   = trans(gaspos , incl)
    starpos  = trans(starpos, incl)
    gasvel   = trans(gasvel , incl)
    
    gasridx  =  (( gaspos[:,0] > -xwdth / 2.000E+00) &
                 ( gaspos[:,1] > -xwdth / 2.000E+00) &
                 ( gaspos[:,2] > -xwdth / 2.000E+00) &
                 ( gaspos[:,0] <  xwdth / 2.000E+00) &
                 ( gaspos[:,1] <  xwdth / 2.000E+00) &
                 ( gaspos[:,2] <  xwdth / 2.000E+00) )
    starridx =  ((starpos[:,0] > -xwdth / 2.000E+00) &
                 (starpos[:,1] > -xwdth / 2.000E+00) &
                 (starpos[:,2] > -xwdth / 2.000E+00) &
                 (starpos[:,0] <  xwdth / 2.000E+00) &
                 (starpos[:,1] <  xwdth / 2.000E+00) &
                 (starpos[:,2] <  xwdth / 2.000E+00) )
        
    gasmass  =  gasmass[ gasridx]
    gaspos   =   gaspos[ gasridx]
    gasvel   =   gasvel[ gasridx]
    gassfr   =   gassfr[ gasridx]
    gaszm    =    gaszm[ gasridx]
    gasrho   =   gasrho[ gasridx]
    gasrho0  =  gasrho0[ gasridx]
    gastemp  =  gastemp[ gasridx]
    starpos  =  starpos[starridx]
    starmass = starmass[starridx]
    
    gasv     = (gasvel[:,0]**2 + gasvel[:,1]**2)**(0.5)
    
    velcopy  = gasvel.copy()
    theta    = np.arctan2(velcopy[:,1],velcopy[:,0])
    
    radvel   = (gaspos[:,0] * gasvel[:,0] + gaspos[:,1] * gasvel[:,1]) / np.sqrt(gaspos[:,0]**2 + gaspos[:,1]**2)
    
    xymgas, xymstar, xysfr, xyo, xyh, xyprad = calcpix(gaspos, starpos, gasmass, starmass,  gasrho, radvel, gassfr, gaszm, zm9 = False)
    
    map_mgass[ :,:] = np.reshape(xymgas , (pix, pix))
    map_mstars[:,:] = np.reshape(xymstar, (pix, pix))
    map_sfrs[  :,:] = np.reshape(xysfr  , (pix, pix))
    map_os[    :,:] = np.reshape(xyo    , (pix, pix))
    map_hs[    :,:] = np.reshape(xyh    , (pix, pix))
    map_pr[    :,:] = np.reshape(xyprad , (pix, pix))
    
    map_mgass[(map_mgass == 0)] = np.nan

    map_vr = map_pr/map_mgass
    
    print 'Saving Data \n'
    h5xcs     = h5map.create_dataset('x_map'           , dtype = float, data =        xcs)
    h5ycs     = h5map.create_dataset('y_map'           , dtype = float, data =        ycs)
    h5xymgas  = h5map.create_dataset('map_sig_gas'     , dtype = float, data =  map_mgass)
    h5xymstar = h5map.create_dataset('map_sig_star'    , dtype = float, data = map_mstars)
    h5xysfr   = h5map.create_dataset('map_sig_sfr'     , dtype = float, data =   map_sfrs)
    h5xyo     = h5map.create_dataset('map_sig_O'       , dtype = float, data =     map_os)
    h5xyh     = h5map.create_dataset('map_sig_H'       , dtype = float, data =     map_hs)
#     h5xyvrad  = h5map.create_dataset('map_sig_vrad'    , dtype = float, data =     map_vr)
h5f.close()