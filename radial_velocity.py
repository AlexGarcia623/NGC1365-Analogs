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

base = '/orange/paul.torrey/zhemler/IllustrisTNG/' + run + '/' # Zach's directory
out_dir = base + 'output/' # needed for Zach's directory

snaps = [99] # Z = 0

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


def test(run, base, out_dir, snap):
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
    grp_cat = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub', 'Group_R_Crit200'])
    sub_cat['SubhaloMass'] *= 1.000E+10 / h
    sub_cat['SubhaloMassType'] *= 1.00E+10 / h

    subs     = np.array(grp_cat['GroupFirstSub'])

    sfms_idx = sfmscut(sub_cat['SubhaloMassType'][subs,4], sub_cat['SubhaloSFR'][subs])
    subs     = subs[(sub_cat['SubhaloMassType'][subs,4] > 1.000E+01**m_star_min) &
                 (sub_cat['SubhaloMassType'][subs,4] < 1.000E+01**m_star_max) &
                 (sub_cat['SubhaloMassType'][subs,0] > 1.000E+01**m_gas_min) &
                 (                                           sfms_idx) ]


    mass = sub_cat['SubhaloMassType']

    sub1 = [526029, 526478, 537236]
    idx = 0
    names = ['TNG0052','TNG0053','TNG0070']

#    print len(sub1)

    for sub in sub1:
      plt.clf()
      x,y,vr,gm  = pretty_picture(out_dir,snap,sub,sub_cat,box_size,scf,names[idx],mass[sub,4])
      rmax = +5.00E+01
      pixl = +2.50E-01
      pixa = pixl**2.00E+00

      pixlims = np.arange(-rmax, rmax + pixl, pixl)
      pix     = len(pixlims) - 1
      pixcs   = pixlims[:-1] + (pixl / 2.00E+00)
      
      xygm, xtra, ytra = np.histogram2d(x, y, weights = gm   , bins = [pixlims, pixlims])
      xypr, xtra, ytra = np.histogram2d(x, y, weights = vr*gm, bins = [pixlims, pixlims])

#      xyvr  = xypr/xygm

      xygm  = np.transpose(xygm)
      xypr  = np.transpose(xypr)  
      xygm  =     np.ravel(xygm)
      xypy  =     np.ravel(xypr)
 
      xygm /= pixa
      xypr /= pixa
 
      xwdth = 1.000E+02
      pixl  = 2.500E-01 
      rmax_tng = xwdth / 2.000E+00 
      pixlims  = np.arange(-rmax_tng, rmax_tng + pixl, pixl)
      pix      = len(pixlims) - 1

      map_prs = np.full((1,pix,pix),np.nan,dtype=float)
      map_gms = np.full((1,pix,pix),np.nan,dtype=float)
      map_prs = np.reshape(xypr,(pix,pix))
      map_gms = np.reshape(xygm,(pix,pix))

      map_vrs = np.full((1,pix,pix),np.nan,dtype=float)
      
      map_gms[(map_gms == 0)] = np.nan

      map_vrs = map_prs/map_gms

      plt.imshow(map_vrs,cmap='spectral')#,norm=mpl.colors.LogNorm())

      plt.gca().invert_yaxis()

#      plt.scatter(x,y,s=1.00E-03,c=vr,cmap='spectral',edgecolors='face')
#      plt.xlabel('x Position (kpc)')
#      plt.ylabel('y Position (kpc)')
      plt.colorbar()#label='Radial Velocity (km/s)')

      plt.title('%s Rotation Map' %names[idx])

      plt.savefig('./Pretty/%sRotationMap' %names[idx])

      idx += 1  


def pretty_picture(out_dir, snap, sub, sub_cat, box_size, scf, name, mass, fullGroup = False):
      sub_pos = sub_cat['SubhaloPos'][sub]
      sub_met = sub_cat['SubhaloGasMetallicity'][sub]
      sub_vel = sub_cat['SubhaloVel'][sub]
      GrNr    = sub_cat['SubhaloGrNr'][sub]

      if ~fullGroup:
        gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
        gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
        gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
        gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
        gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
        gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
        gas_eab   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['ElectronAbundance'])
        gas_ien   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['InternalEnergy'   ])
        ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
        XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
      else:
        print GrNr

      gas_pos    = center(gas_pos, sub_pos, box_size)
      gas_pos   *= (scf / h)
      gas_vel   *= np.sqrt(scf)
      gas_vel   -= sub_vel
      gas_mass  *= (1.000E+10 / h)
      gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
      gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
      gas_rho   *= xh / mh

      OH = ZO/XH * 1.00/16.00

      ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
      ro2    = 2.000E+00 * ro

      sf_idx = gas_rho > 1.300E-01
      incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

#      incl = [50.0,60.0]

      sfg = gas_rho > 0.13

#      gas_pos = gas_pos[:][(sfg)]
#      gas_vel = gas_vel[:][(sfg)]
#      gas_mass = gas_mass[(sfg)]

      gas_pos  = trans(gas_pos, incl)
      gas_vel  = trans(gas_vel, incl)
      

      gas_vr   = (gas_vel[:,0]**2 + gas_vel[:,1]**2)**(0.5)

      vel_copy = gas_vel.copy()

      theta    = np.arctan2(vel_copy[:,1],vel_copy[:,0])

      return gas_pos[:,0],gas_pos[:,1],gas_vr*np.sin(theta),gas_mass

def zoxh(out_dir, snap, sub, sub_cat, box_size):
  ZO = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metalicity'])

  OH = ZO/XH * 1.00/16.00

  return OH

def returnPosSfr(out_dir, snap, sub, sub_cat, box_size, scf):
  sub_pos = sub_cat['SubhaloPos'][sub]
  sub_met = sub_cat['SubhaloGasMetallicity'][sub]
  sub_vel = sub_cat['SubhaloVel'][sub]

  gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
  gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])

  gas_pos    = center(gas_pos, sub_pos, box_size)
  gas_pos   *= (scf / h)

  return gas_pos,gas_sfr

def line(data, p1, p2):
    return p1*data + p2

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

def calc_incl(pos0, vel0, m0, ri, ro):
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

def calc_rsfr_io(pos0, sfr0):
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
        return np.nan, np.nan
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
        return np.nan, np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idxo   = idx0[(sfrf > fraco)]
    idxo   = idxo[0]
    rsfro  = rpos[idxo]
    return rsfri, rsfro

def sfmscut(m0, sfr0):
    nsubs = len(m0)
    idx0  = np.arange(0, nsubs)
    non0  = ((m0   > 0.000E+00) &
             (sfr0 > 0.000E+00) )
    m     =    m0[non0]
    sfr   =  sfr0[non0]
    idx0  =  idx0[non0]
    ssfr  = np.log10(sfr/m)
    sfr   = np.log10(sfr)
    m     = np.log10(  m)

    idxbs   = np.ones(len(m), dtype = int) * -1
    cnt     = 0
    mbrk    = 1.020E+01
    mstp    = 2.000E-01
    mmin    = m_star_min
    mbins   = np.arange(mmin, mbrk + mstp, mstp)
    rdgs    = []
    rdgstds = []

    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        rdg   = np.median(ssfrb)
        idxb  = (ssfrb - rdg) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
        rdgs.append(rdg)
        rdgstds.append(np.std(ssfrb))

    rdgs       = np.array(rdgs)
    rdgstds    = np.array(rdgstds)
    mcs        = mbins[:-1] + mstp / 2.000E+00

    parms, cov = curve_fit(line, mcs, rdgs, sigma = rdgstds)
    mmin    = mbrk
    mmax    = 1.100E+01
    mbins   = np.arange(mmin, mmax + mstp, mstp)
    mcs     = mbins[:-1] + mstp / 2.000E+00
    ssfrlin = line(mcs, parms[0], parms[1])
    for i in range(0, len(mbins) - 1):
        idx   = (m > mbins[i]) & (m < mbins[i+1])
        idx0b = idx0[idx]
        mb    =    m[idx]
        ssfrb = ssfr[idx]
        sfrb  =  sfr[idx]
        idxb  = (ssfrb - ssfrlin[i]) > -5.000E-01
        lenb  = np.sum(idxb)
        idxbs[cnt:(cnt+lenb)] = idx0b[idxb]
        cnt += lenb
    idxbs    = idxbs[idxbs > 0]
    sfmsbool = np.zeros(len(m0), dtype = int)
    sfmsbool[idxbs] = 1
    sfmsbool = (sfmsbool == 1)
    return sfmsbool

def calcrsfr(pos0, sfr0, frac = 5.000E-01, ndim = 3):
    if (ndim == 2):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 +
                       pos0[:,1]**2.000E+00 )
    if (ndim == 3):
        rpos = np.sqrt(pos0[:,0]**2.000E+00 +
                       pos0[:,1]**2.000E+00 +
                       pos0[:,2]**2.000E+00 )
    sfr    = sfr0[np.argsort(rpos)]
    rpos   = rpos[np.argsort(rpos)]
    sfrtot = np.sum(sfr)
    if (sfrtot < 1.000E-09):
        return np.nan
    sfrf   = np.cumsum(sfr) / sfrtot
    idx0   = np.arange(1, len(sfr) + 1, 1)
    idx    = idx0[(sfrf > frac)]
    idx    = idx[0]
    rsfr50 = rpos[idx]
    return rsfr50

test(run, base, out_dir, snaps)
