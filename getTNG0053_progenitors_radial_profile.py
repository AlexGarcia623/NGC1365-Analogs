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

from matplotlib.colors import LogNorm

# set RC params
mpl.rcParams['figure.facecolor'] = 'white'
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

# os.environ['MANPATH']="/home/paul.torrey/local/texlive/2018/texmf-dist/doc/man:$MANPATH"
# os.environ['INFOPATH']="/home/paul.torrey/local/texlive/2018/texmf-dist/doc/info:$INFOPATH"
# os.environ['PATH']="/home/paul.torrey/local/texlive/2018/bin/x86_64-linux:/home/paul.torrey/local/texlive/2018/texmf-dist:$PATH"

mpl.rcParams['text.usetex']        = True
mpl.rcParams['font.family']        = 'serif'
mpl.rc('font',**{'family':'sans-serif','serif':['Times New Roman'],'size':15})
mpl.rc('text', usetex=True)

path = '/orange/paul.torrey/IllustrisTNG/Runs/'

run  = 'L35n2160TNG'

__tree_thing__ = '/orange/paul.torrey/IllustrisTNG/Runs/L35n2160TNG/postprocessing/trees/LHaloTree/trees_sf1_099.0.hdf5'

base_dir  = path + '/' + run
base_path = path + '/' + run
post_dir  = path + '/' + run + '/postprocessing'
tree_dir  = post_dir + '/trees/SubLink/'

snap = 99

subs  = [526029, 526478, 537236]
names = ['TNG0052','TNG0053','TNG0070']

which_sub = 1

# h5file = h5py.File( names[which_sub] + '.hdf5', 'w' )

SUBHALO_ID   = subs[which_sub]

MAJOR_MERGER_FRAC = 1.0/100.0
MERGER_MASS_TYPE  = 4 # 0 -> Gas, 1 -> DM, 4 -> Stars

file_num     = 0 
file_index   = None
file_located = False

RAW_SUBHALO_ID = SUBHALO_ID + int(snap * 1.00E+12)
while not file_located:    
    tree_file = h5py.File( tree_dir + 'tree_extended.%s.hdf5' %file_num, 'r' )
    
    subID = np.array(tree_file.get('SubhaloIDRaw'))
    
    overlap = RAW_SUBHALO_ID in subID
    
    if (overlap):
        file_located = True
        file_index   = np.where( subID == RAW_SUBHALO_ID )[0][0]
    else:
        file_num += 1


h  = 6.774E-01
xh = 7.600E-01
zo = 3.500E-01
mh = 1.6725219E-24
kb = 1.3086485E-16
mc = 1.270E-02

with h5py.File(__tree_thing__,'r') as f:
    for i in f:
        tree1 = f.get(i)
        redshifts = np.array(tree1.get('Redshifts'))
        break
        
snap_to_z = { snap: redshifts[i] for i, snap in enumerate(range(0,100)) }
z_to_snap = { round(redshifts[i],2): snap for i, snap in enumerate(range(0,100)) }

def maps_profiles(sub,snap,stellar_mass,SF_GAS_FLAG=True,main_prog=True):
    print(snap)
        
    hdr = il.groupcat.loadHeader(base_dir,snap)
    boxsize = hdr['BoxSize']
    scf     = hdr['Time']
    z0      = (1.000E+00 / scf - 1.000E+00)
    
    print('Loading Subhalo info\n')
    fields = ['SubhaloMassType', 'SubhaloSFR', 'SubhaloPos', 'SubhaloVel']
    sub_cat = il.groupcat.loadSubhalos(base_dir, snap, fields = fields)
    
    sub_cat['SubhaloMassType'][:,:] *= (1.00E+10 / h)
    submstar = np.log10(sub_cat['SubhaloMassType'][sub,4])
    subsfr   = np.log10(sub_cat['SubhaloSFR'][sub])
    
    subpos   = sub_cat['SubhaloPos'][sub]
    subvel   = sub_cat['SubhaloVel'][sub]
    
    starpos  = il.snapshot.loadSubhalo(base_dir, snap, sub, 4, fields = ['Coordinates'      ])
    gaspos   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gasvel   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Velocities'       ])
    gasmass  = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Masses'           ])
    gassfr   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gasrho   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['Density'          ])
    gasmet   = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    ZO       = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH       = il.snapshot.loadSubhalo(base_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    
    print('Centering\n')
    gaspos    = center(gaspos, subpos, boxsize)
    gaspos   *= (scf / h)
    gasvel   *= np.sqrt(scf)
    gasvel   -= subvel
    gasmass  *= (1.000E+10 / h)
    gasrho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gasrho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
    gasrho   *= xh / mh

    OH = ZO/XH * 1.00/16.00

    ri, ro = calc_rsfr_io(gaspos, gassfr)
    ro2    = 2.000E+00 * ro

    sfidx  = gasrho > 1.300E-01
    incl   = calcincl(gaspos[sfidx], gasvel[sfidx], gasmass[sfidx], ri, ro)

    gaspos  = trans(gaspos , incl)
    gasvel  = trans(gasvel , incl)
    starpos = trans(starpos, incl)
    

    if (SF_GAS_FLAG):
        gaspos   = gaspos [ sfidx ]
        gasvel   = gasvel [ sfidx ]
        gasmass  = gasmass[ sfidx ]
        gassfr   = gassfr [ sfidx ]
        gasrho   = gasrho [ sfidx ]
        gasmet   = gasmet [ sfidx ]
        ZO       = ZO     [ sfidx ]
        XH       = XH     [ sfidx ]
    
    print('Calc Gradient\n')
    
    gasrad = np.sqrt( gaspos[:,0]**2 + gaspos[:,1]**2 + gaspos[:,2]**2 )
    
    OH = np.log10( ZO / XH * (1.00/16.00) ) + 12.0
        
    dr = 1
    rs = np.arange(0,35,dr)
    Zs = np.ones( len(rs) ) * np.nan
    
    for index, r in enumerate(rs):
        mask = ( (gasrad > r) & (gasrad < (r + dr)) )
        if sum(mask) > 10:
            Zs[index] = np.median( OH[mask] )
            
    plt.clf()
    plt.plot( rs, Zs )
    
    plt.text( 0.8,0.9, r'$z=%s$' %round(snap_to_z[snap],2), transform=plt.gca().transAxes )
    
    if not main_prog:
        plt.text( 0.025,0.9 , r'${\rm Merger~ Partner}$', transform=plt.gca().transAxes )
    
    plt.xlabel( r'$R~{\rm (kpc)}$' )
    plt.ylabel( r'$\log ({\rm O/H}) + 12 ~{\rm (dex)}$' )
    
    plt.ylim(6.0,10.0)
    
    if main_prog:
        plt.savefig( 'gradients/' + '%s_main' %snap + '.pdf', bbox_inches='tight' )
    else:
        plt.savefig( 'gradients/' +'%s_partner_%s' %(snap,stellar_mass) + '.pdf', bbox_inches='tight' )
    
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

def main():
    rootID  = tree_file["SubhaloID"][file_index]
    fp      = tree_file["FirstProgenitorID"][file_index]
    snapNum = tree_file["SnapNum"][file_index]
    mass    = np.log10( tree_file["SubhaloMassType"][file_index,MERGER_MASS_TYPE] * 1.00E+10 / h )
    subfind = tree_file["SubfindID"][file_index]
    subhalo = tree_file["SubhaloID"][file_index]

    desired_snaps = [99,59,67,50,33,25,21,17,13]
    
    print( 'Subfind ID: %s' %subfind )
    print( 'Subhalo ID: %s' %subhalo )
    
    maps_profiles(subfind,snapNum,stellar_mass=mass,SF_GAS_FLAG=True,main_prog=True)

    # return
    while fp != -1:
        fpIndex = file_index + (fp - rootID)
        mass    = np.log10( tree_file["SubhaloMassType"][fpIndex,MERGER_MASS_TYPE] * 1.00E+10 / h )
        fpSnap  = tree_file["SnapNum"][fpIndex]
        subfind = tree_file["SubfindID"][fpIndex]
            
        if fpSnap in desired_snaps:
            maps_profiles(subfind,fpSnap,stellar_mass=mass,SF_GAS_FLAG=True,main_prog=True)

        nextProgenitor = tree_file["NextProgenitorID"][fpIndex]
        while nextProgenitor != -1:
            npIndex = file_index + (nextProgenitor - rootID)
            npMass  = np.log10( tree_file["SubhaloMassType"][npIndex,MERGER_MASS_TYPE] * 1.00E+10 / h )
            npSnap  = tree_file["SnapNum"][npIndex]
            npSubfind = tree_file["SubfindID"][npIndex]

            npHaloID = tree_file["SubhaloID"][npIndex]

#             if ((10**npMass / 10**mass) > MAJOR_MERGER_FRAC) and (npMass > 0):

#                 maps_profiles(npSubfind,npSnap,stellar_mass=npMass,SF_GAS_FLAG=True,main_prog=False)

            nextProgenitor = tree_file["NextProgenitorID"][npIndex]

        fp = tree_file["FirstProgenitorID"][fpIndex]
        
main()