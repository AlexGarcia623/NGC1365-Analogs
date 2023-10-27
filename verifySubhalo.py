import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import illustris_python as il
import scipy.integrate as integrate

# set RC params
mpl.rcParams['figure.facecolor'] = 'black'
mpl.rcParams['axes.facecolor']='white'
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 7.5
mpl.rcParams['ytick.major.size'] = 7.5
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

os.environ['MANPATH']="/home/paul.torrey/local/texlive/2018/texmf-dist/doc/man:$MANPATH"
os.environ['INFOPATH']="/home/paul.torrey/local/texlive/2018/texmf-dist/doc/info:$INFOPATH"
os.environ['PATH']="/home/paul.torrey/local/texlive/2018/bin/x86_64-linux:/home/paul.torrey/local/texlive/2018/texmf-dist:$PATH"

mpl.rcParams['text.usetex']        = True
mpl.rcParams['font.family']        = 'serif'
mpl.rc('font',**{'family':'sans-serif','serif':['Times New Roman'],'size':15})
mpl.rc('text', usetex=True)

path = '/orange/paul.torrey/IllustrisTNG/Runs/'

run  = 'L35n2160TNG'

base_dir  = path + '/' + run
post_dir  = path + '/' + run + '/postprocessing'
tree_dir  = post_dir + '/trees/SubLink/'

subs  = [526029, 526478, 537236]
names = ['TNG0052','TNG0053','TNG0070']

which_sub = 0

SUBHALO_ID = subs[which_sub] 
print names[which_sub]
snap       = 99

h5file = h5py.File( names[which_sub] + '.hdf5', 'w' )
maps     = h5file.create_group( 'map' )
profiles = h5file.create_group( 'profile' )
redshift = h5file.create_group( 'snap' ) 

SF_GAS_FLAG = False

h      = 6.774E-01
xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

def maps_profiles(SF_GAS_FLAG=True):
    xwdth    = 1.000E+02
    pixl     = 2.500E-01
#     pixl     = 1.00E-01
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
    
    map_mgass  = np.full((pix, pix), np.nan, dtype = float)
    map_mstars = np.full((pix, pix), np.nan, dtype = float)
    map_sfrs   = np.full((pix, pix), np.nan, dtype = float)
    map_os     = np.full((pix, pix), np.nan, dtype = float)
    map_hs     = np.full((pix, pix), np.nan, dtype = float)
    
    sub = SUBHALO_ID
    
    hdr = il.groupcat.loadHeader(base_dir,snap)
    boxsize = hdr['BoxSize']
    scf     = hdr['Time']
    z0      = (1.000E+00 / scf - 1.000E+00)
    
    print('Loading Subhalos\n')
    fields = ['SubhaloMassType', 'SubhaloSFR', 'SubhaloPos', 'SubhaloVel']
    sub_cat = il.groupcat.loadSubhalos(base_dir, snap, fields = fields)
    
    sub_cat['SubhaloMassType'][:,:] *= (1.00E+10 / h)
    submstar = np.log10(sub_cat['SubhaloMassType'][sub,4])
    subsfr   = np.log10(sub_cat['SubhaloSFR'][sub])
    
    subpos   = sub_cat['SubhaloPos'][sub]
    subvel   = sub_cat['SubhaloVel'][sub]
    
    gasid    = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['ParticleIDs'      ])
    gaspos   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gasvel   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Velocities'       ])
    gasmass  = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Masses'           ])
    gassfr   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gaszm    = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    gasrho   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Density'          ])
    gasxe    = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['ElectronAbundance'])
    gasu     = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['InternalEnergy'   ])
    starid   = il.snapshot.loadSubhalo(base_dir, snap, sub, 4, fields = ['ParticleIDs'      ])
    starpos  = il.snapshot.loadSubhalo(base_dir, snap, sub, 4, fields = ['Coordinates'      ])
    starmass = il.snapshot.loadSubhalo(base_dir, snap, sub, 4, fields = ['Masses'           ])
    bhid     = il.snapshot.loadSubhalo(base_dir, snap, sub, 5, fields = ['ParticleIDs'      ])
    ZO       = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH       = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    
    print('Centering\n')
    gaspos   = center(gaspos , subpos, boxsize)
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
    gasoh    = np.log10((zo * gaszm / xh) * (1.000E+00 / 1.600E+01)) + 1.200E+01
    gasmu    = 4.000E+00 / (1.000E+00 + 3.000E+00 * xh + 4.000E+00 * xh * gasxe)
    gastemp  = ((5.000E+00 / 3.000E+00) - 1.000E+00) * (gasu / kb) * (1.000E+10) * gasmu * mh
    
    ri, ro = calcrsfrio(gaspos, gassfr)
    ro2    = 2.000E+00 * ro
    
    sfidx  = (gasrho > 1.300E-01)
    
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
    
    if (SF_GAS_FLAG):
        gasmass  =  gasmass[ gasridx & sfidx ]
        gaspos   =   gaspos[ gasridx & sfidx ]
        gasvel   =   gasvel[ gasridx & sfidx ]
        gassfr   =   gassfr[ gasridx & sfidx ]
        gaszm    =    gaszm[ gasridx & sfidx ]
        gasrho   =   gasrho[ gasridx & sfidx ]
        gasrho0  =  gasrho0[ gasridx & sfidx ]
        gastemp  =  gastemp[ gasridx & sfidx ]
        ZO       =       ZO[ gasridx & sfidx ] 
        XH       =       XH[ gasridx & sfidx ] 
    else:
        gasmass  =  gasmass[ gasridx ]
        gaspos   =   gaspos[ gasridx ]
        gasvel   =   gasvel[ gasridx ]
        gassfr   =   gassfr[ gasridx ]
        gaszm    =    gaszm[ gasridx ]
        gasrho   =   gasrho[ gasridx ]
        gasrho0  =  gasrho0[ gasridx ]
        gastemp  =  gastemp[ gasridx ]
        ZO       =       ZO[ gasridx ] 
        XH       =       XH[ gasridx ] 
    starpos  =  starpos[ starridx ]
    starmass = starmass[ starridx ]
    
    print('Calc Pixels\n')
    xymgas, xymstar, xysfr, xyo, xyh = calcpix(gaspos, starpos, gasmass, starmass,  gasrho, gassfr, gaszm, zm9 = False)
    
    xymgas  = np.reshape(xymgas , (pix, pix))
    xymstar = np.reshape(xymstar, (pix, pix))
    xysfr   = np.reshape(xysfr  , (pix, pix))
    xyo     = np.reshape(xyo    , (pix, pix))
    xyh     = np.reshape(xyh    , (pix, pix))
    
    print('Saving Maps\n')
    h5xcs     = maps.create_dataset('x_map'       , dtype = float,  data = xcs    )
    h5ycs     = maps.create_dataset('y_map'       , dtype = float,  data = ycs    )
    h5xymgas  = maps.create_dataset('map_sig_gas' , dtype = float,  data = xymgas )
    h5xymstar = maps.create_dataset('map_sig_star', dtype = float,  data = xymstar)
    h5xysfr   = maps.create_dataset('map_sig_sfr' , dtype = float,  data = xysfr  )
    h5xyo     = maps.create_dataset('map_sig_O'   , dtype = float,  data = xyo    )
    h5xyh     = maps.create_dataset('map_sig_H'   , dtype = float,  data = xyh    )
    
    
    gasrad  = np.sqrt( gaspos[:,0]**2 + gaspos[:,1]**2 + gaspos[:,2]**2 )
    starrad = np.sqrt(starpos[:,0]**2 +starpos[:,1]**2 +starpos[:,2]**2 )
    
    gasrad  = np.sqrt( gaspos[:,0]**2 + gaspos[:,1]**2 )
    starrad = np.sqrt(starpos[:,0]**2 +starpos[:,1]**2 )
    
    dr = 0.5
    print np.max(gasrad)
    rs = np.arange( 0.0, 30.0, dr )
    
    OH = ZO/XH * 1.00/16.00
    
    pf_oh       = np.ones( len(rs) ) * np.nan
    pf_sig_gas  = np.ones( len(rs) ) * np.nan
    pf_sig_star = np.ones( len(rs) ) * np.nan
    pf_sig_sfr  = np.ones( len(rs) ) * np.nan
    
    print('Creating profiles \n')
    for index, current_r in enumerate(rs):
        gasmask  = ((gasrad  > current_r) & (gasrad  < current_r + dr ))
        starmask = ((starrad > current_r) & (starrad < current_r + dr ))
    
        annulus_area = np.pi * ( (current_r+dr)**2 - (current_r)**2 )
        if round(current_r,1) % 5.0 == 0:
            print current_r, annulus_area
    
        if (sum(gasmask) > 10):
            pf_oh      [index] = np.log10( np.median( OH[gasmask] ) ) + 12
            pf_sig_gas [index] = np.sum( gasmass[gasmask] )   / annulus_area
            pf_sig_sfr [index] = np.median( gassfr[gasmask] ) / annulus_area
        if (sum(starmask) > 10):
            pf_sig_star[index] = np.sum( starmass[starmask] ) / annulus_area
            
    print('Saving Profiles\n')
    h5pfgas  = profiles.create_dataset('prf_sig_gas' , dtype = float,  data = np.log10(pf_sig_gas ))
    h5pfstar = profiles.create_dataset('prf_sig_star', dtype = float,  data = np.log10(pf_sig_star))
    h5pfsfr  = profiles.create_dataset('prf_sig_sfr' , dtype = float,  data = pf_sig_sfr )
    h5pfoh   = profiles.create_dataset('prf_oh'      , dtype = float,  data = pf_oh      )
    
    print('Saving redshift')
    h5snap = redshift.create_dataset( 'redshift', dtype = float, data = z0 )

    
def calcpix(gaspos, starpos, mgas, mstar, hrho, sfr, zm, zm9 = True):
    rmax   = +5.000E+01
    pixl   = +2.500E-01
#     pixl   = 1.000E-01
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
    if (zm9):
        xyo, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx,4]), bins = [pixlims, pixlims])
        xyh, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx,0]), bins = [pixlims, pixlims])
    else:
        xh = 7.600E-01
        zo = 3.500E-01
        xyh, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = mgas[hidx] * xh                       , bins = [pixlims, pixlims])
        xyo, x, y = np.histogram2d(gaspos[hidx,0], gaspos[hidx,1], weights = np.multiply(mgas[hidx], zm[hidx]) * zo, bins = [pixlims, pixlims])
    
    print('Done Calc Pixel\n')
    xymstar = np.transpose(xymstar)
    xymgas  = np.transpose( xymgas)
    xysfr   = np.transpose(  xysfr)
    xyo     = np.transpose(    xyo)
    xyh     = np.transpose(    xyh)
    xymstar =     np.ravel(xymstar)
    xymgas  =     np.ravel( xymgas)
    xysfr   =     np.ravel(  xysfr)
    xyo     =     np.ravel(    xyo)
    xyh     =     np.ravel(    xyh)
    
    xymgas  /= pixa
    xymstar /= pixa
    xysfr   /= pixa
    xyo     /= pixa
    xyh     /= pixa
    
    return xymgas, xymstar, xysfr, xyo, xyh
    
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

maps_profiles(SF_GAS_FLAG)
h5file.close()