import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

import copy
import psutil
import os
import subprocess
import sys

import tarfile
import re
import time

# Astropy:
from astropy.cosmology import FlatLambdaCDM 
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from skimage.transform import resize

# Scipy:
from scipy.integrate import simpson
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from photutils import CircularAperture, aperture_photometry


h=0.6774
c_AAs = 2.99792458e18
Lsun = 1.361e6   # erg/s/cm2
myzin = 0.0044983025884921465      # corresponding to a distance of 20Mpc using Planck18
myzout = 1.5
d_Lout = cosmo.luminosity_distance(myzout)
d_Lin = cosmo.luminosity_distance(myzin)
        
base = '/home/fernarua/Desktop/IAC/TNG_MockIFU/'


def filter_data(data, inp_data, radius, Stars=False):
    '''
    Function: Reduces the given input data to the FOV with the input radius.
    ------------------------------------------------------------
    Input: 
    Inp_data: the variable you want to reduce
    Coordinates: the rotated coordinates centered in the CM, to specific if we work with Gas, DM or Stars.
    Radius: Desidered limit radius in [ckpc].
    
    Output: 
    out_data: Filtered data.
    ------------------------------------------------------------
    '''
    Coordinates = np.column_stack((data.RC_x, data.RC_y, data.RC_z))
    Coord = Coordinates / h
    # To remove the WIND particles in the PartType4 class:
    if Stars == True:
        mask_stars = (data.GFM_StellarFormationTime >= 0)
        coord, data = Coord[mask_stars], inp_data[mask_stars]
        mask_radius = np.linalg.norm(coord, axis=1) <= radius
        out_data = data[mask_radius]

    else:
        mask_radius = np.linalg.norm(Coord, axis=1) <= radius
        out_data = inp_data[mask_radius]
    
    return out_data


def rodrigues_matrix(v_f, v_i=[0,0,1]):
    v_i = np.array(v_i) / np.linalg.norm(v_i)
    v_f = np.array(v_f) / np.linalg.norm(v_f)
    phi = np.arccos(np.dot(v_i, v_f)) 
    k = np.cross(v_i, v_f)
    K = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]])
    term_1 = np.eye(3)
    term_2 = K * np.sin(phi)
    term_3 = (1-np.cos(phi)) * (K @ K) 
    return term_1 + term_2 + term_3


def calc_age(FormationTime):
    from astropy.cosmology import Planck18 as cosmo
    z = 1/FormationTime - 1
    age = cosmo.age(z).value
    return age


def does_file_exist(filepath):
    exists = os.path.exists(filepath)
    return exists

 
def TNG_data_loader(subhalo_ID, snapshot, radius, v_f, Stars):
    '''
    Function to download only the data from TNG Stellar Particles
    that we will use in the computation of the Mock DataCube.
    This properties are:
    - Metallicity
    - Age
    - Mass
    - Rotated Coordinates
    - Rotated Velocities

    Arguments:
    ------------------------------------------------------------
    subhalo_ID: float
        ID of the TNG galaxy
        
    snapshot: float
        Snapshot of the galaxy. Conversion to redshift in:
        
        https://www.tng-project.org/data/downloads/TNG50-1/
    radius: float
        Radius in kpc of the FOV.
        
    v_f: 3d-array
        Vector of the inclination for the galaxy.

    Stars: Bool
        If True, store the needed data for the Stellar DataCube.
        If False store the Gas, DM and Stellar coordinates for representation.

    Return:
    ------------------------------------------------------------
    '''
    if Stars == True:
            
        datafolder = base + 'TNG_Galaxies/'
        # os.makedirs(datafolder, exist_ok=True)
        filename = str(subhalo_ID) + '_' + 'snap0' + str(snapshot) + '.hdf5'
        filepath = datafolder + filename 
            
        with h5py.File(filepath, 'r') as f:
            group = f['PartType4']
            
            GFM_Metallicity = group['GFM_Metallicity'][:]
            GFM_StellarFormationTime = group['GFM_StellarFormationTime'][:]
            GFM_InitialMass = group['GFM_InitialMass'][:]
            RC_x = group['RotatedCoordinates'][:,0]
            RC_y = group['RotatedCoordinates'][:,1]
            RC_z = group['RotatedCoordinates'][:,2]
            RV_x = group['RotatedVelocities'][:,0]
            RV_y = group['RotatedVelocities'][:,1]
            RV_z = group['RotatedVelocities'][:,2]
        
        data = pd.DataFrame({'GFM_Metallicity': GFM_Metallicity,
                             'GFM_StellarFormationTime': GFM_StellarFormationTime,
                             'GFM_InitialMass': GFM_InitialMass,
                             'RC_x': RC_x,
                             'RC_y': RC_y,
                             'RC_z': RC_z,
                             'RV_x': RV_x,
                             'RV_y': RV_y,
                             'RV_z': RV_z})
        
        Metallicity = filter_data(data, np.log10(GFM_Metallicity / 0.0127), radius, Stars=True)
        FormationTime = filter_data(data, GFM_StellarFormationTime, radius, Stars=True)
        Age = calc_age(FormationTime)
        Masses = filter_data(data, data.GFM_InitialMass * 1e10 / h, radius, Stars=True)
        RC_x0 = filter_data(data, data.RC_x, radius, Stars=True)
        RC_y0 = filter_data(data, data.RC_y, radius, Stars=True)
        RC_z0 = filter_data(data, data.RC_z, radius, Stars=True)
        RV_x0 = filter_data(data, data.RV_x, radius, Stars=True)
        RV_y0 = filter_data(data, data.RV_y, radius, Stars=True)
        RV_z0 = filter_data(data, data.RV_z, radius, Stars=True)
    
        RotatedCoord0 = np.column_stack((RC_x0, RC_y0, RC_z0))
        RotatedVel0 = np.column_stack((RV_x0, RV_y0, RV_z0))
    
        rotation_matrix = rodrigues_matrix(v_f)                              
        RotatedCoordinates = RotatedCoord0 @ rotation_matrix
        RotatedVelocities = RotatedVel0 @ rotation_matrix
    
        df = pd.DataFrame({'Metallicity': Metallicity,
                            'Age': Age,
                            'Masses': Masses,
                            'RC_x': RotatedCoordinates[:,0],
                            'RC_y': RotatedCoordinates[:,1],
                            'RC_z': RotatedCoordinates[:,2],
                            'RV_x': RotatedVelocities[:,0],
                            'RV_y': RotatedVelocities[:,1],
                            'RV_z': RotatedVelocities[:,2]})
    
        df.reset_index(drop=True, inplace=True)
    
        RotatedCoordinates = np.column_stack((df.RC_x, df.RC_y, df.RC_z))
        RotatedVelocities = np.column_stack((df.RV_x, df.RV_y, df.RV_z))
    
        h5datafolder = base + 'TNGdata_' + str(subhalo_ID) + '/'
        os.makedirs(h5datafolder, exist_ok=True)
        h5filename = (f'TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}')
        #filename = filename.replace('.', 'c')
        h5filename = h5filename + '.hdf5'
        h5filepath = h5datafolder + h5filename
        
        with h5py.File(h5filepath, "w") as f:
            header_grp = f.create_group("Header")
            header_grp.attrs["Subhalo_ID"] = subhalo_ID
            header_grp.attrs['Snapshot'] = snapshot
            header_grp.attrs["Radius"] = radius
            header_grp.attrs["RotationVector"] = v_f
            
            data_group = f.create_group("FilteredData")
            data_group.create_dataset('Metallicity', data=df.Metallicity)
            data_group.create_dataset('Age', data=df.Age)
            data_group.create_dataset('Mass', data=df.Masses)
            data_group.create_dataset('RotatedCoordinates', data=RotatedCoordinates)
            data_group.create_dataset('RotatedVelocities', data=RotatedVelocities)

    else:
         # In this case we are only interested in the coordinates:   
        datafolder = base + 'TNG_Galaxies/'
        #os.makedirs(datafolder, exist_ok=True)
        filename = str(subhalo_ID) + '_' + 'snap0' + str(snapshot) + '.hdf5'
        filepath = datafolder + filename 

        def load_group(filepath, part_type, radius, prefix):
            with h5py.File(filepath, 'r') as f:
                if part_type in f:
                    group = f[part_type]
                    RC_x = group['RotatedCoordinates'][:,0]
                    RC_y = group['RotatedCoordinates'][:,1]
                    RC_z = group['RotatedCoordinates'][:,2]
                    data = pd.DataFrame({'RC_x': RC_x,
                                         'RC_y': RC_y,
                                         'RC_z': RC_z})
                    RC_x0 = filter_data(data, data.RC_x, radius, Stars=False)
                    RC_y0 = filter_data(data, data.RC_y, radius, Stars=False)
                    RC_z0 = filter_data(data, data.RC_z, radius, Stars=False)
    
                    RotatedCoord0 = np.column_stack((RC_x0, RC_y0, RC_z0))
                    rotation_matrix = rodrigues_matrix(v_f)                              
                    RotatedCoordinates = RotatedCoord0 @ rotation_matrix

                    #df_name = f'df_{prefix}'
                    df = pd.DataFrame({'RC_x': RotatedCoordinates[:,0],
                                       'RC_y': RotatedCoordinates[:,1],
                                       'RC_z': RotatedCoordinates[:,2]})
                    df.reset_index(drop=True, inplace=True)
            return df
                    
        df_gas = load_group(filepath, 'PartType0', radius, 'gas')
        df_dm = load_group(filepath, 'PartType1', radius, 'dm')
        df_stars = load_group(filepath, 'PartType4', radius, 'stars')
        df = pd.concat([df_gas, df_dm, df_stars], axis=1)
        
        Gas_RotatedCoordinates = np.column_stack((df_gas.RC_x, df_gas.RC_y, df_gas.RC_z))
        DM_RotatedCoordinates = np.column_stack((df_dm.RC_x, df_dm.RC_y, df_dm.RC_z))
        Stars_RotatedCoordinates = np.column_stack((df_stars.RC_x, df_stars.RC_y, df_stars.RC_z))
        
        h5datafolder = base + 'TNGdata_' + str(subhalo_ID) + '/'
        os.makedirs(h5datafolder, exist_ok=True)
        h5filename = (f'All_TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}')
        #filename = filename.replace('.', 'c')
        h5filename = h5filename + '.hdf5'
        h5filepath = h5datafolder + h5filename
        
        with h5py.File(h5filepath, "w") as f:
            header_grp = f.create_group("Header")
            header_grp.attrs["Subhalo_ID"] = subhalo_ID
            header_grp.attrs['Snapshot'] = snapshot
            header_grp.attrs["Radius"] = radius
            header_grp.attrs["RotationVector"] = v_f
            
            data_group = f.create_group("FilteredData")
            data_group.create_dataset('Gas_RotatedCoordinates', data=Gas_RotatedCoordinates)
            data_group.create_dataset('DM_RotatedCoordinates', data=DM_RotatedCoordinates)
            data_group.create_dataset('Stars_RotatedCoordinates', data=Stars_RotatedCoordinates)


def read_hdf5file(subhalo_ID, snapshot, radius, v_f, Stars):
    '''
    Function to read the hdf5 File and open as a pandas table:

    Arguments:
    ----------
    '''
    if Stars == True:
        filepath = base + (f'TNGdata_{subhalo_ID}/TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')

        does_exist = does_file_exist(filepath)
        
        if not does_exist:
            TNG_data_loader(subhalo_ID, snapshot, radius, v_f, Stars)
            filepath = base + (f'TNGdata_{subhalo_ID}/TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')

        else:
            print('TNG_Data file exist, and doing nothing!')
        
        with h5py.File(filepath, 'r') as f:
            group = f['FilteredData']
            keys = list(group.keys())
            #for key in keys:
                #data = group[key][:]
            Metallicity = group['Metallicity'][:]
            Age = group['Age'][:]
            Mass = group['Mass'][:]
            RC_x = group['RotatedCoordinates'][:,0]
            RC_y = group['RotatedCoordinates'][:,1]
            RC_z = group['RotatedCoordinates'][:,2]
            RV_x = group['RotatedVelocities'][:,0]
            RV_y = group['RotatedVelocities'][:,1]
            RV_z = group['RotatedVelocities'][:,2]
    
        df = pd.DataFrame({'Metallicity': Metallicity,
                             'Age': Age,
                             'Mass': Mass,
                             'RC_x': RC_x,
                             'RC_y': RC_y,
                             'RC_z': RC_z,
                             'RV_x': RV_x,
                             'RV_y': RV_y,
                             'RV_z': RV_z})
        return df

    else:
        filepath = base + (f'TNGdata_{subhalo_ID}/All_TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')

        does_exist = does_file_exist(filepath)

        if not does_exist:
            TNG_data_loader(subhalo_ID, snapshot, radius, v_f, Stars)
            filepath = base + (f'TNGdata_{subhalo_ID}/All_TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')
        
        else:
            print('TNG_Data file exist, and doing nothing!')
        
        with h5py.File(filepath, 'r') as f:
            group = f['FilteredData']
            #keys = list(group.keys())
            #for key in keys:
                #data = group[key][:]
            Gas_RC_x = group['Gas_RotatedCoordinates'][:,0]
            Gas_RC_y = group['Gas_RotatedCoordinates'][:,1]
            Gas_RC_z = group['Gas_RotatedCoordinates'][:,2]
            DM_RC_x = group['DM_RotatedCoordinates'][:,0]
            DM_RC_y = group['DM_RotatedCoordinates'][:,1]
            DM_RC_z = group['DM_RotatedCoordinates'][:,2]
            Stars_RC_x = group['Stars_RotatedCoordinates'][:,0]
            Stars_RC_y = group['Stars_RotatedCoordinates'][:,1]
            Stars_RC_z = group['Stars_RotatedCoordinates'][:,2]

        max_length = max(len(Gas_RC_x), len(Gas_RC_y), len(Gas_RC_z), 
                     len(DM_RC_x), len(DM_RC_y), len(DM_RC_z),
                     len(Stars_RC_x), len(Stars_RC_y), len(Stars_RC_z))

        def pad_with_nan(arr, length):
            return np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=np.nan)
    
        Gas_RC_x = pad_with_nan(Gas_RC_x, max_length)
        Gas_RC_y = pad_with_nan(Gas_RC_y, max_length)
        Gas_RC_z = pad_with_nan(Gas_RC_z, max_length)
        DM_RC_x = pad_with_nan(DM_RC_x, max_length)
        DM_RC_y = pad_with_nan(DM_RC_y, max_length)
        DM_RC_z = pad_with_nan(DM_RC_z, max_length)
        Stars_RC_x = pad_with_nan(Stars_RC_x, max_length)
        Stars_RC_y = pad_with_nan(Stars_RC_y, max_length)
        Stars_RC_z = pad_with_nan(Stars_RC_z, max_length)

        df = pd.DataFrame({
        'Gas_RC_x': Gas_RC_x,
        'Gas_RC_y': Gas_RC_y,
        'Gas_RC_z': Gas_RC_z,
        'DM_RC_x': DM_RC_x,
        'DM_RC_y': DM_RC_y,
        'DM_RC_z': DM_RC_z,
        'Stars_RC_x': Stars_RC_x,
        'Stars_RC_y': Stars_RC_y,
        'Stars_RC_z': Stars_RC_z
        })
    
        return df



def Galaxy_ColorHistogram(subhalo_ID, snapshot, radius, v_f, num_bins, Stars=True):

    if Stars == True:
        
        filepath = base + (f'TNGdata_{subhalo_ID}/TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')
            
        TNG_data = read_hdf5file(subhalo_ID, snapshot, radius, v_f, Stars)
        x_min, x_max = TNG_data.RC_x.min(),  TNG_data.RC_x.max()
        y_min, y_max = TNG_data.RC_y.min(),  TNG_data.RC_y.max()
        x_bins = np.linspace(x_min, x_max, num_bins+1)
        y_bins = np.linspace(y_min, y_max, num_bins+1)
        z_bin = np.zeros((num_bins-1, num_bins-1))

        props = ['Metallicity', 'Age', 'RV_z']
        npydatafolder = base + (f'TNGdata_{subhalo_ID}/Histograms/')
        os.makedirs(npydatafolder, exist_ok=True)
        
        for prop in props:  
            npyfilename = (f'TNGhist_{prop}_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}bins.npy')
            npyfilepath = npydatafolder + npyfilename
            print('Starting with:', prop)

            for i in range(num_bins - 1):
                print(i, end = '\r')
                for j in range(num_bins - 1):
                    TNG_bin = TNG_data[(TNG_data.RC_x >= x_bins[i]) & (TNG_data.RC_x < x_bins[i+1]) & 
                                   (TNG_data.RC_y >= y_bins[j]) & (TNG_data.RC_y < y_bins[j+1])]
                    TNG_bin.reset_index(drop=True, inplace=True)
                    
                    if len(TNG_bin.Metallicity) == 0:
                        continue
                        
                    z_bin[i,j] = np.mean(TNG_bin[prop])
                    
            np.save(npyfilepath, z_bin)
        print('Histograms completed !')



def mk_hist_plots(subhalo_ID, snapshot, radius, v_f, Stars, vmin=None, vmax=None, *, num_bins=None, binsize=None):

    if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
        raise ValueError("You must provide either num_bins or binsize, but not both.")
    
    if num_bins is None:
        num_bins = calculate_num_bins(snapshot, radius, binsize)
    elif binsize is None:
        binsize = spaxel_size(snapshot, radius, radius)

    print('Number of bins: ', num_bins)
    print('Spaxel size: ', binsize)
    print('Radius: ', radius)
        
    if Stars == True:
        
        def load_hist_npy(prop, snapshot, radius, v_f):
            datafolder = base + (f'TNGdata_{subhalo_ID}/Histograms/')
            filename = (f'TNGhist_{prop}_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}bins')
            filepath = datafolder + filename + '.npy'
            return np.load(filepath)

        datafolder = (base + f'TNGdata_{subhalo_ID}/Histograms/')
        filename = (f'TNGhist_Metallicity_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}bins')
        filepath = datafolder + filename + '.npy'

        does_exist = does_file_exist(filepath)
        
        if not does_exist:
            print('Histogram does not exist, creating:')
            Galaxy_ColorHistogram(subhalo_ID, snapshot, radius, v_f, num_bins, Stars)

        
        x_axis = np.linspace(-radius, radius, num_bins)
        y_axis = np.linspace(-radius, radius, num_bins)
        
        hist_metallicity = load_hist_npy('Metallicity', snapshot, radius, v_f)
        hist_age = load_hist_npy('Age', snapshot, radius, v_f)
        hist_vel = load_hist_npy('RV_z', snapshot, radius, v_f)
        
        hist_metallicity[hist_metallicity==0.0] = -100
        
        fig, axs = plt.subplots(1, 3, figsize=(30,8))
        im1 = axs[0].imshow(hist_metallicity.T, extent = [-radius, radius, -radius, radius],vmin = vmin[0], vmax = vmax[0], cmap='viridis')
        cbar1 = fig.colorbar(im1, ax=axs[0], label = '[Fe / H]', pad=0.005, fraction=0.05)
        cbar1.ax.set_ylabel('[Fe / H]', fontsize = 18)
        axs[0].set_xlabel('x [kpc]', fontsize=18)
        axs[0].set_ylabel('y [kpc]', fontsize=18)
        axs[0].tick_params(labelsize=16)

        im2 = axs[1].imshow(hist_age.T, extent = [-radius, radius, -radius, radius],vmin = vmin[1], vmax = vmax[1], cmap='viridis')
        cbar2 = fig.colorbar(im2, ax=axs[1], label = 'Age [Gyr]', pad=0.005, fraction=0.05)
        cbar2.ax.set_ylabel('Age [Gyr]', fontsize = 18)
        axs[1].set_xlabel('x [kpc]', fontsize=18)
        axs[1].set_ylabel('y [kpc]', fontsize=18)
        axs[1].tick_params(labelsize=16)

        im4 = axs[2].imshow(hist_vel.T, extent = [-radius, radius, -radius, radius],vmin = vmin[2], vmax = vmax[2], cmap='seismic')
        cbar4 = fig.colorbar(im4, ax=axs[2], label = 'Tangenial Velocity [km/s]', pad=0.005, fraction=0.05)
        cbar4.ax.set_ylabel('Tangenial Velocity [km/s]', fontsize = 18)
        axs[2].set_xlabel('x [kpc]', fontsize=18)
        axs[2].set_ylabel('y [kpc]', fontsize=18)
        axs[2].tick_params(labelsize=16)

        plt.tight_layout()
        datafolder = base + (f'TNGdata_{subhalo_ID}/Histograms/Plots/')
        os.makedirs(datafolder, exist_ok=True)
        filename = (f'TNGhist_Stars_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}')
        filepath = datafolder + filename + '.png'
        plt.savefig(filepath)
        plt.show()

    else:
        

        filepath = (f'TNGdata_{subhalo_ID}/All_TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')
        TNG_data = read_hdf5file(subhalo_ID, snapshot, radius, v_f, Stars)
        titles = ['Gas', 'Dark Matter', 'Stars']
        RC_x = np.column_stack((TNG_data.Gas_RC_x, TNG_data.DM_RC_x, TNG_data.Stars_RC_x))
        RC_y = np.column_stack((TNG_data.Gas_RC_y, TNG_data.DM_RC_y, TNG_data.Stars_RC_y))
        RC_z = np.column_stack((TNG_data.Gas_RC_z, TNG_data.DM_RC_z, TNG_data.Stars_RC_z))
        
        fig, axs = plt.subplots(1, 3, figsize=(30, 8))
        radius_arcsec = num_bins * binsize
        print(radius_arcsec)

        scale = radius_arcsec / radius
        
        for j in range(3):

            ax = axs[j]
            arr_x = RC_x[:,j]
            RC_xnew = arr_x[~np.isnan(arr_x)] * scale 
            arr_y = RC_y[:,j]
            RC_ynew = arr_y[~np.isnan(arr_y)] * scale 
            
            x_min, x_max = RC_xnew.min(), RC_xnew.max() 
            y_min, y_max = RC_ynew.min(), RC_ynew.max()
            x_bins = np.linspace(x_min, x_max, num_bins)
            y_bins = np.linspace(y_min, y_max, num_bins)
          
            hist, xedges, yedges = np.histogram2d(RC_xnew, RC_ynew, bins=[x_bins, y_bins])
            im = axs[j].hist2d(RC_xnew, RC_ynew, bins = [x_bins, y_bins], norm=LogNorm(vmin=vmin, vmax=vmax), cmap='gray')
            cbar = fig.colorbar(im[3], ax=axs[j], pad=0.005, fraction=0.05)
            cbar.set_label('Density', fontsize=18)
            cbar.ax.tick_params(labelsize=16)
            
            ax.text(0.02, 0.97, (titles[j]), fontsize = 19, color='white', transform=ax.transAxes, verticalalignment='center', horizontalalignment='left')

            ax.text(0.02, 0.1, (f'Subhalo ID: {subhalo_ID}'), fontsize = 19, color='white', transform=ax.transAxes, verticalalignment='center', horizontalalignment='left')

            ax.text(0.02, 0.05, (f'Redshift: {snapshot_to_redshift(snapshot)}'), fontsize = 19, color='white', transform=ax.transAxes, verticalalignment='center', horizontalalignment='left')
        
            ax.set_xlabel('x [arcsec]', fontsize=18)
            ax.set_ylabel('y [arcsec]', fontsize=18)
            ax.tick_params(labelsize=16)
            ax.set_facecolor('black') 

        plt.tight_layout()
        datafolder = base + (f'TNGdata_{subhalo_ID}/Histograms/Plots/')
        os.makedirs(datafolder, exist_ok=True)
        filename = (f'TNGhist_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}bins')
        filepath = datafolder + filename + '.png'
        plt.savefig(filepath)
        plt.show()

' ---------------------------------------------------------------------------------------------------------------'



def openMILES_files(PathFile):
    '''
    Function to open MILES SSP models downloaded from:
    https://cloud.iac.es/index.php/s/aYECNyEQfqgYwt4?path=%2FMILES
    ----------
    Input:
    PathFile: path to the folder 
    ----------
    '''
    extracted_path = base + 'MILES_models/extracted_files'
    #os.makedirs(extracted_path, exist_ok=True)

    
    does_exist = does_file_exist(extracted_path)
    
    if not does_exist:
            
        with tarfile.open(PathFile, 'r:gz') as tar:
            tar.extractall(path=extracted_path)
        
        extracted_files = os.listdir(extracted_path)
  
    folder_path = base + 'MILES_models/extracted_files'
    files = sorted(os.listdir(folder_path))
    pattern = re.compile(r'Mku1.30Z([mp])([\d.]+)T([\d.]+)_iTp0.00+_baseFe\.fits')
    metallicities0 = []
    ages0 = []
    for file in files:
        match = pattern.match(file)
        if match:
            sign = -1 if match.group(1) == 'm' else 1
            metallicity = sign * float(match.group(2))
            age = float(match.group(3))
            metallicities0.append(metallicity)
            ages0.append(age)
    
    MILES_array0 = list(zip(metallicities0, ages0))
    MILES_data = np.array(MILES_array0)
    
    values_to_remove = [-2.27, -1.79, -1.49]
    mask_metal = ~np.isin(MILES_data[:, 0], values_to_remove)
    MILES_data = MILES_data[mask_metal]
    return MILES_data


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 ** 2  # in MB


def CynematicCorrection(RotatedVelocity_Z, wavelength0, spectrum, redshift):
    '''
    Function to correct wavelength shifts due to the perpendicular 
    motion of the stars.
    '''
    lightvel = 3e5  # km/s
    scalefactor = 1 / (1 + redshift)
    Velocity = RotatedVelocity_Z / np.sqrt(scalefactor)
    shifted_wavelengths = wavelength0 * (1 + Velocity / lightvel)
    # Interpolate the spectrum back to the original wavelength grid
    corrected_spectrum = np.interp(wavelength0, shifted_wavelengths, spectrum)
    return corrected_spectrum
    

def InterpolatedGrid(MILES_data):
    '''
    Creates the interpolated grid and flux with the MILES models.
    '''
    metallicities = np.unique(MILES_data[:,0])
    ages = np.unique(MILES_data[:,1])
    wavelength = np.linspace(3540.5, 7409.6, 4300)

    flux = np.zeros((len(ages), len(metallicities), len(wavelength)))    
    for i, age in enumerate(ages):
        for j, metallicity in enumerate(metallicities):
            sign = 'p' if metallicity >= 0 else 'm'
            metallicity = abs(metallicity)            
            metallicity_str = f"{metallicity:.2f}"
            age_str = f'{age:07.4f}'            
            fits_file = base + (f'MILES_models/extracted_files/Mku1.30Z{sign}{metallicity_str}T{age_str}_iTp0.00_baseFe.fits')
            
            if not os.path.exists(fits_file):
                print(f"Archivo no encontrado: {fits_file}")
                continue
        
            with fits.open(fits_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                crval1 = header['CRVAL1']
                cdelt1 = header['CDELT1']
                crpix1 = header['CRPIX1']                
                num_points = len(data)
                wavelength_array = crval1 + (np.arange(num_points) + 1 - crpix1) * cdelt1    
                flux[i, j, :] = np.interp(wavelength, wavelength_array, data)
    grid = (ages, metallicities)
    return grid, flux

    

def SyntheticSpectra(TNG_data, subhalo_ID, snapshot, radius, v_f, method, grid, flux, output_name, save_plots):
    '''
    Function: 
    Creates a numpy array with the corresponding SYNTHETIC SPECTRA of the TNG_data configuration.
    The projection vector can be changed in the TNGdata loading in the v_f input parameter.
    ----------
    Input:
    - TNG_data: Obtained from tng.TNGData_Cube
    - MILES_data: Downloaded from MILES stellar library.
    - method: 
        (1) Nearest: Asign the nearest age-metallicity value from MILES library to each of the 
        StellarParticles from the TNG simulation.
        (2) Interpolation: Creates an interpolated grid with spectra for the age-metallicity 
        values of MILES. 
        Note: In TNG we encounter metallicity values up to 1, while in MILES our maximum is 0.4.
        The interpolation method asigns the maximum metallicity allowed by MILES to TNG StellarParticles
        with metallicities greater than 0.4
        (3) Extrapolation: Same as interpolation but allows extrapolation up to 1 metallicity value.
    - If make_plots=True, we plot the obteined Synthetic Spectrum.

    Output:
    - wavelength and spectrum of the Synthetic Spectra.
    - Numpy array with this values to be stored in the folderpath.
    ----------
    '''
    MILES_data = openMILES_files(PathFile = base + 'MILES_models/extracted_files')
    if TNG_data is None:
        filepath = base + (f'TNGdata_{subhalo_ID}/TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')
        TNG_data = read_hdf5file(subhalo_ID, snapshot, radius, v_f, Stars=True)
        
    ''' ---------- NEAREST ---------- '''
    
    if method == 'Nearest':

        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
        TNG_1 = TNG_data.copy()
        TNG_1.Metallicity = np.array([find_nearest(MILES_data[:,0], metal) for metal in TNG_data.Metallicity])
        TNG_1.Age = np.array([find_nearest(MILES_data[:,1], age) for age in TNG_data.Age])
        
        total_spectrum = None
        num_points = None
        wavelength = np.linspace(3540.5, 7409.6, 4300)
        
        pattern = re.compile(r'Mku1.30Z([mp])([\d.]+)T([\d.]+)_iTp0.00_baseFe\.fits')
    
        for i in range(len(TNG_1.Metallicity)):
            metallicity, age, mass, velocity = TNG_1.Metallicity[i], TNG_1.Age[i], TNG_1.Mass[i], TNG_1.RV_z[i]
            
            sign = 'p' if metallicity >= 0 else 'm'
            metallicity = abs(metallicity)
            metallicity_str = f"{metallicity:.2f}"
            age_str = f'{age:07.4f}' 
            fits_file = f'./extracted_files/Mku1.30Z{sign}{metallicity_str}T{age_str}_iTp0.00_baseFe.fits'
        
            if not os.path.exists(fits_file):
                print(f"Archivo no encontrado: {fits_file}")
                continue
        
            with fits.open(fits_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                crval1 = header['CRVAL1']
                cdelt1 = header['CDELT1']
                crpix1 = header['CRPIX1']
        
                if total_spectrum is None:
                    total_spectrum = np.zeros_like(data)
                    num_points = len(data)
                    wavelength_array = crval1 + (np.arange(num_points) + 1 - crpix1) * cdelt1    
            
                total_spectrum += CynematicCorrection(velocity, wavelength_array, data, 0) * mass
    
        if save_plots == True:
            results = np.vstack((wavelength, total_spectrum))
            datafolder = base + (f'TNGdata_{subhalo_ID}/SyntheticSpectrum/')
            os.makedirs(datafolder, exist_ok=True)
            filename = (f'SS_{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}.npy')
            filepath = datafolder + filename  
            np.save(filepath, results)
            fig = plt.figure(figsize=(25, 10))
            plt.plot(wavelength, total_spectrum, color='blue')
            plt.xlabel('Wavelength')
            plt.title('Summed Spectrum')
            plt.show()
        
        return wavelength, total_spectrum

    ''' ---------- INTERPOLATION ---------- '''
    
    if method == 'Interpolation':
        
        wavelength = np.linspace(3540.5, 7409.6, 4300)
        if grid == False:
            grid, flux = InterpolatedGrid(MILES_data)

        total_int_spectra = np.zeros(len(wavelength))
        
        for i in range(len(TNG_data.Metallicity)):
            metallicity, age, mass, velocity = TNG_data.Metallicity[i], TNG_data.Age[i], TNG_data.Mass[i], TNG_data.RV_z[i]
            age_clipped = np.clip(age, grid[0].min(), grid[0].max())
            metallicity_clipped = np.clip(metallicity, grid[1].min(), grid[1].max())
            int_point = np.array([[age_clipped, metallicity_clipped]])
            int_spectra = interpn(grid, flux, int_point, method='linear', bounds_error=False, fill_value=None)
            total_int_spectra += CynematicCorrection(velocity, wavelength, int_spectra[0], 0) * mass
    
        if save_plots == True:
            results = np.vstack((wavelength, total_int_spectra))
            datafolder = base + (f'TNGdata_{subhalo_ID}/SyntheticSpectrum/')
            os.makedirs(datafolder, exist_ok=True)
            filename = (f'SS_{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}.npy')
            filepath = datafolder + filename  
            np.save(filepath, results)
            fig = plt.figure(figsize=(25, 10))
            plt.plot(wavelength, total_int_spectra, color='blue')
            plt.xlabel('Wavelength')
            plt.title('Interpolated Spectrum')
            plt.show()

        if output_name != None:
            results = np.vstack((wavelength, total_int_spectra))
            np.save(output_name, results)
    
        return wavelength, total_int_spectra
        
    ''' ---------- EXTRAPOLATION ---------- '''

    if method == 'Extrapolation':

        wavelength = np.linspace(3540.5, 7409.6, 4300)
        if grid == False:
            grid, flux = InterpolatedGrid(MILES_data)

        total_ext_spectra = 0
        
        for i in range(len(TNG_data.Metallicity)):
            metallicity, age, mass, velocity = TNG_data.Metallicity[i], TNG_data.Age[i], TNG_data.Mass[i], TNG_data.RV_z[i]
            ext_point = np.array([[age, metallicity]])
            ext_spectra0 = interpn(grid, flux, ext_point, method='linear', bounds_error=False, fill_value=None)
            total_ext_spectra += CynematicCorrection(velocity, wavelength, ext_spectra0[0], 0) * mass
                
        if save_plots == True:
            results = np.vstack((wavelength, total_ext_spectrum))
            datafolder = base + (f'TNGdata_{subhalo_ID}/SyntheticSpectrum/')
            os.makedirs(datafolder, exist_ok=True)
            filename = (f'SS_{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}.npy')
            filepath = datafolder + filename  
            np.save(filepath, results)
            fig = plt.figure(figsize=(25, 10))
            plt.plot(wavelength, total_ext_spectrum, color='blue')
            plt.xlabel('Wavelength')
            plt.title('Extrapolated Spectrum')
            plt.show()

        return wavelength, total_ext_spectra



def create_hdf5_datafile(filepath, n_bins, v_f, method, M=4300):
    '''
    Creates the empy 3d array for storing the DataCube results.
    '''
    with h5py.File(filepath, "w") as f:
        header_grp = f.create_group("Header")
        header_grp.attrs["n_bins"] = n_bins
        header_grp.attrs["v_f"] = np.array(v_f).astype(np.float32)
        header_grp.attrs["method"] = method

        body_group = f.create_group("Body")

        data_zeros = np.zeros((M, n_bins, n_bins), dtype = np.float32)
        body_group.create_dataset("Data", data=data_zeros, dtype=np.float32)

        check_zeros = np.zeros((n_bins, n_bins), dtype=np.bool)
        body_group.create_dataset("Checks", data=check_zeros, dtype=np.bool)

        data_mag = np.zeros((n_bins, n_bins), dtype = np.float32)
        body_group.create_dataset("Data_mag", data=data_mag, dtype=np.float32)
    return filepath
        


def create_new_savefile(filepath, snapshot, n_bins, v_f, radius, method, M=4300, overwrite=False):
    '''
    Function to create HDF5 file to store the data for the DataCube.
    '''
    does_exist = does_file_exist(filepath)

    if not does_exist:
        create_hdf5_datafile(filepath, n_bins, v_f, method, M=M)

    else:
        a = input('Overwrite savepath?')

        if a == 'yes':
        #if overwrite:
            print('Save file exists, but overwritting!')
            create_hdf5_datafile(filepath, n_bins, v_f, method, M=M)
            
        else:
            print("Save file exists, and doing nothing!")
    return filepath



def IFU_datacube(subhalo_ID, snapshot, radius, v_f, method='Interpolation',  *, binsize=None, num_bins=None):
    ''' Creates the IFUs:
    ---------------------
    Input:

    - subhalo_ID:
    - snapshot
    - radius
    - v_f
    - num_bins
    -method

    Output:
    - FITS file
    ---------------------
    '''

    if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
        raise ValueError("You must provide either num_bins or binsize, but not both.")
    
    if num_bins is None:
        num_bins = calculate_num_bins(snapshot, radius, binsize)
    elif binsize is None:
        binsize = spaxel_size(snapshot, radius, num_bins)

    print('The bin size is : ', binsize, 'arsec2')
    print('The number of bins is: ', num_bins)
    
    if snapshot == 99:
        filterpath = base +'Filters/SLOAN_SDSS.gprime_filter.dat'
        print('Zero point with Filter: SDSS g, and integrated magnitude 20 in 1 kpc.')
        mag = 18
        cal_radius = 1

    elif snapshot == 40:
        filterpath = base +'Filters/JWST_NIRCam.F162M.dat'
        mag = 24
        cal_radius = 1
        print('Zero point with filter: JWST F162M, and integrated magnitude 23 in 1 kpc.')

    flux_factor = flux_scale_factor(subhalo_ID, snapshot, cal_radius, v_f, method, filterpath, mag)
    print('Flux factor: ' , flux_factor)
    Start = time.time()
    
    print('Reading / creating TNG50 data file')
    filepath = base + (f'TNGdata_{subhalo_ID}/TNGdata_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.hdf5')
    TNG_data = read_hdf5file(subhalo_ID, snapshot, radius, v_f, Stars=True)
    
    wavelength = np.linspace(3540.5, 7409.6, 4300)
    synthetic_spectrum = np.zeros(len(wavelength))
    obs_wavelength = correct_by_redshift(wavelength, snapshot)

    print('Reading / creating MILES data file')
    MILES_data = openMILES_files(base + 'MILES_models/extracted_files/MILES_BASTI_KU_baseFe.tar.gz')

    x_min, x_max = TNG_data.RC_x.min(),  TNG_data.RC_x.max()
    y_min, y_max = TNG_data.RC_y.min(),  TNG_data.RC_y.max()
    x_bins = np.linspace(x_min, x_max, num_bins+1)
    y_bins = np.linspace(y_min, y_max, num_bins+1)

    hist, xedges, yedges = np.histogram2d(TNG_data.RC_x, TNG_data.RC_y, bins=[x_bins, y_bins])
    grid, flux = InterpolatedGrid(MILES_data)
    
    savefolder = base + (f'TNGdata_{subhalo_ID}/DC_SaveResults/')
    os.makedirs(savefolder, exist_ok=True)

    filename = (f'DCresults_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}_{num_bins}.hdf5')
    filepath = savefolder + filename
    savepath = create_new_savefile(filepath, snapshot, num_bins, v_f, radius, method)
    
    with h5py.File(savepath, 'a') as f:
    
        data = f['Body']['Data']
        data_mag = f['Body']['Data_mag']
        checks = f['Body']['Checks']

        k = -1
        started = False
    
        imin = 0
        imax = num_bins
        jmin = 0 
        jmax = num_bins
            
        for i in range(imin, imax):
            start = time.time()
            for j in range(jmin, jmax):
                k+=1
    
                if bool(checks[i,j]) == bool(0):
                    
                    if started == False:
                        print(f'Starting with cell {k}/{num_bins*num_bins}\nRow {k//num_bins}/{num_bins}')

                    started = True
                    
                    TNG_bin = TNG_data[(TNG_data.RC_x >= x_bins[i]) & (TNG_data.RC_x < x_bins[i+1]) & 
                           (TNG_data.RC_y >= y_bins[j]) & (TNG_data.RC_y < y_bins[j+1])]
        
                    TNG_bin.reset_index(drop=True, inplace=True)
    
                    if len(TNG_bin.Metallicity) == 0:
                        checks[i,j] = bool(1)
                        continue
                    
                    wavelength, spectrum = SyntheticSpectra(TNG_bin, 
                                                                  subhalo_ID,  
                                                                  snapshot,  
                                                                  radius,
                                                                  v_f,
                                                                  method,
                                                                  grid=False,
                                                                  flux=False,
                                                                  output_name = None,
                                                                  save_plots = False)
                    
                    """ Put in correct UNITS: ergs/s/cm2/AA/arcsec2 """
                    obs_spectrum = spectrum * flux_factor# / (binsize**2)
                    
                    """ A. MAGNITUDE IMAGE: """
                    mag_sup = superficial_magnitude(obs_spectrum, obs_wavelength, binsize, filterpath)
                    data_mag[i, j] = mag_sup
                        
                    """ B. DATA CUBE: """
                    data[:, i, j] = obs_spectrum
                    
                    checks[i,j] = bool(1)
                    
                else:
                    continue
                
            if started==True:
                end = time.time()
                print(f'{i}/{num_bins} Memory usage: {get_memory_usage()} MB ; Time: {end-start}', end='\r')
    
    """ Save in a FITS file """                

    dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/')
    os.makedirs(dcfolder, exist_ok=True)
    
    dcfilename = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
    dcfilepath = dcfolder + dcfilename

        
    with h5py.File(savepath, 'r') as hdf:
        data_cube = hdf['Body']['Data'][:]
        header = fits.Header()
       
        header['CTYPE1'] = 'X'
        header['CUNIT1'] = 'arcsec'
        header['CRPIX1'] = num_bins/2
        header['CRVAL1'] = 0
        header['CDELT1'] = binsize
        
        header['CTYPE2'] = 'Y'
        header['CUNIT2'] = 'arcsec'
        header['CRPIX2'] = num_bins/2
        header['CRVAL2'] = 0
        header['CDELT2'] = binsize

        header['CTYPE3'] = 'WAVELENGTH'
        header['CUNIT3'] = 'angstrom'
        header['CRPIX3'] = 1                                  
        header['CRVAL3'] = obs_wavelength[0]                  
        header['CDELT3'] = np.mean(np.diff(obs_wavelength))   
        header['SPECRES'] = 2.51

        header['BUNIT'] = 'erg/s/cm2/AA/arcsec2'

        hdu = fits.PrimaryHDU(data=np.float32(data_cube), header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(dcfilepath, overwrite=True)
    
    dcfilename_mag = (f'MI_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
    dcfilepath_mag = dcfolder + dcfilename_mag

    with h5py.File(savepath, 'r') as hdf:
        data_mag = hdf['Body']['Data_mag'][:]

        header = fits.Header()
        header['CTYPE1'] = 'X'
        header['CUNIT1'] = 'arcsec'
        header['CRPIX1'] = num_bins/2
        header['CRVAL1'] = 0
        header['CDELT1'] = binsize
        header['CTYPE2'] = 'Y'
        header['CUNIT2'] = 'arcsec'
        header['CRPIX1'] = num_bins/2
        header['CRVAL1'] = 0
        header['CDELT1'] = binsize

        hdu = fits.PrimaryHDU(data=data_mag, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(dcfilepath_mag, overwrite=True)
        
    print('FITS file created !')
    transfer_by_ssh(subhalo_ID, snapshot, radius, v_f, num_bins)
    
    End = time.time()
    print('Time : ', End-Start)



def transfer_by_ssh(subhalo_ID, snapshot, radius, v_f, num_bins, remote_user='fernarua', remote_server='delfin.ll.iac.es'):
     
    dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/')
    dcfilename = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
    dcfilepath = dcfolder + dcfilename

    base_diva = '/scratch/fernarua/TNG_MockIFU/'
    remote_folder = base_diva + (f'TNGdata_{subhalo_ID}/DataCubes/')
    remote_dir_command = f"ssh {remote_user}@{remote_server} 'mkdir -p {remote_folder}'"

    try:
        subprocess.run(remote_dir_command, shell=True, check=True)
        #print(f"Directory {remote_folder} created or already exists on {remote_server}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating remote directory: {e}")
        return

    remote_filepath = remote_folder + dcfilename
    scp_command = f"scp {dcfilepath} {remote_user}@{remote_server}:{remote_filepath}"

    try:
        subprocess.run(scp_command, shell=True, check=True)
        #print(f"File {dcfilepath} successfully copied to {remote_server}:{remote_filepath}")
        print('FITS file transfer to fernarua@delfin.ll.iac.es:/scratch/fernarua')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while copying file: {e}")


def create_calibration_file(subhalo_ID, snapshot, cal_radius, v_f, method='Interpolation'):
    
    TNG_data_loader(subhalo_ID, snapshot, cal_radius, v_f, Stars=True)
    TNG_data = read_hdf5file(subhalo_ID, snapshot, cal_radius, v_f, Stars=True)
    MILES_data = openMILES_files(PathFile = './MILES_models/extracted_files/MILES_BASTI_KU_baseFe.tar.gz')

    grid, flux = InterpolatedGrid(MILES_data)

    calfolder = base + (f'TNGdata_{subhalo_ID}/CalibrationTNGdata/')
    calname = (f'CalibTNGdata_snap0{snapshot}_{cal_radius}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.npy')
    filepath = calfolder + calname 
    
    SyntheticSpectra(None, subhalo_ID, snapshot, cal_radius, v_f, method, grid, flux, filepath, save_plots=False)
    return filepath
    

def superficial_magnitude(spectrum, wavelength, binsize, filterpath):
    ''' Calculate photometric magnitude. '''
    obs_spectrum = spectrum
    mag = integrated_magnitude(filterpath, wavelength, obs_spectrum)
    flux = 10**(-(mag + 48.6) / 2.5)
    flux_sup = flux #/ (binsize**2)
    mag_sup = -2.5*np.log10(flux_sup) - 48.6
    return mag_sup


def dat_to_txt(dat_file_path):
    txt_file_path = dat_file_path.replace('.dat', '.txt')
    does_exist = does_file_exist(txt_file_path)

    if not does_exist:
        with open(dat_file_path, 'r') as dat_file:
            data = dat_file.read()
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(data)
        return txt_file_path

    else:
        return txt_file_path


def integrated_magnitude(filterpath, lamS, spec):
    ''' The spec must be in F_nu '''
    txt_file_path = dat_to_txt(filterpath)

    if not os.path.isfile(txt_file_path):
        raise FileNotFoundError(f"The file {txt_file_path} does not exist.")

    try:
        lamF, filt = np.loadtxt(txt_file_path, unpack=True)
        
    except ValueError as e:
        raise ValueError(f"File {txt_file_path} is empty or improperly formatted: {e}")
        
    lamF, filt = np.loadtxt(txt_file_path, unpack=True)

    lamS = np.array(lamS, dtype=np.float32)
    lamF = np.array(lamF, dtype=np.float32)
    filt = np.array(filt, dtype=np.float32)
    spec = np.array(spec, dtype=np.float32)
    filt_int = np.interp(lamS, lamF, filt)
    
    I1 = simpson(spec*filt_int*lamS, x=lamS)         # Denominator
    I2 = simpson(filt_int/lamS, x=lamS)              # Numerator
    
    fnu = I1 / (I2 * c_AAs)
    mAB = -2.5*np.log10(fnu) - 48.6              #AB magnitude
    return mAB


def flux_scale_factor(subhalo_ID, snapshot, cal_radius, v_f, method, filterpath, mag, overwrite = False):

    calfolder = base + (f'TNGdata_{subhalo_ID}/CalibrationTNGdata/')
    os.makedirs(calfolder, exist_ok=True)
    calname = (f'CalibTNGdata_snap0{snapshot}_{cal_radius}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}.npy')
    filepath = calfolder + calname 
    
    does_exist = does_file_exist(filepath)

    if not does_exist:
        create_calibration_file(subhalo_ID, snapshot, cal_radius, v_f, method)

    else:
        a = input('Overwrite calibration file?')

        if a == 'yes':
        #if overwrite:
            print('Flux file exists, but overwritting!')
            create_calibration_file(subhalo_ID, snapshot, cal_radius, v_f, method)
            
        else:
            print("Flux file exists, and doing nothing!")
    
    Spectrum = np.load(filepath, allow_pickle = True)
    lamS, spec = Spectrum[0], Spectrum[1]
    spectrum = spec# * (lamS**2) / c_AAs
    
    TNG_mag = integrated_magnitude(filterpath, lamS, spectrum)
    Deltam = np.abs(TNG_mag - mag)
    
    FluxFactor = 10**(-Deltam/2.5)

    return FluxFactor
    # This is the factor by which we need to multiply our spectra


def spaxel_size(snapshot, radius, num_bins):
    """ Calculate the spaxel size """
    redshift = snapshot_to_redshift(snapshot)
    radius_Mpc = radius / 1000.0

    if snapshot == 99:
        D_A = 20
        
    else:
        D_A = cosmo.angular_diameter_distance(redshift).value
    
    theta_rad = radius_Mpc / D_A
    theta_arcsec = 2 * theta_rad * (180 * 3600) / np.pi
    spaxel_size = theta_arcsec / num_bins   # arsec / pix
    return spaxel_size
    

def calculate_num_bins(snapshot, radius, binsize):
    """ Calculate the number of bins given the spaxel size """
    redshift = snapshot_to_redshift(snapshot)
    radius_Mpc = radius / 1000.0  

    if snapshot == 99:
        D_A = 20 
        
    else:
        D_A = cosmo.angular_diameter_distance(redshift).value
    
    theta_rad = radius_Mpc / D_A 
    theta_arcsec = 2 * theta_rad * (180 * 3600) / np.pi 
    num_bins = theta_arcsec / binsize 
    return int(np.round(num_bins))


def snapshot_to_redshift(snapshot):
    '''
    Conversion between snapshots from TNG-50 to redshifts
    '''
    snapshots = np.linspace(0, 99, 100)
    redshifts = [20.05, 14.99, 11.98, 10.98, 10.00, 9.39, 9.00, 8.45, 8.01, 7.60, 7.24, 7.01, 6.49, 6.01, 5.85, 5.53, 5.23, 5.00, 4.66, 4.43, 4.18, 4.01, 3.71, 3.49, 3.28, 3.01, 2.90, 2.73, 2.58, 2.44, 2.32, 2.21, 2.10, 2.00, 1.90, 1.82, 1.74, 1.67, 1.60, 1.53, 1.50, 1.41, 1.36, 1.30, 1.25, 1.21, 1.15, 1.11, 1.07, 1.04, 1.00, 0.95, 0.92, 0.89, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.68, 0.64, 0.62, 0.60, 0.58, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.35, 0.33, 0.31, 0.30, 0.27, 0.26, 0.24, 0.23, 0.21, 0.20, 0.18, 0.17, 0.15, 0.14, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01, 0.00]
    snap_to_z = dict(zip(snapshots, redshifts))
    redshift = snap_to_z[snapshot]
    return redshift

def correct_by_redshift(wavelength, snapshot):
    redshift = snapshot_to_redshift(snapshot)
    obs_wavelength = wavelength * (1 + redshift)
    return obs_wavelength



""" ------------------ JAMES WEBB SPACE TELESCOPE ------------------ """

def JWST_sim(subhalo_ID, snapshot, radius, v_f, method = 'Interpolation', jwst_fov = False, *, binsize=None, num_bins=None):

    if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
        raise ValueError("You must provide either num_bins or binsize, but not both.")

    if jwst_fov == True:
        radius = (3 * (np.pi / 180) / 3600) * (cosmo.angular_diameter_distance(snapshot_to_redshift(snapshot)).to('kpc').value)
        num_bins = 30
        binsize = 3 / num_bins

    else:
        if num_bins is None:
            num_bins = calculate_num_bins(snapshot, radius, binsize)
        elif binsize is None:
            binsize = spaxel_size(snapshot, radius, num_bins)

    print('The radius is : ', radius, 'kpc')
    print('The bin size is : ', binsize, ' x ', binsize,  'arsec2')
    print('The number of bins is: ', num_bins)
    
    instpath = base +'Filters/jwst_nirspec_prism_disp.fits'
    if snapshot == 99:
        filterpath = base +'Filters/SLOAN_SDSS.gprime_filter.dat'

    elif snapshot == 40:
        filterpath = base +'Filters/JWST_NIRCam.F162M.dat'
    
    if snapshot == 99:
        datafolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/')
        dataname = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}_z1.50.fits')
        datapath = datafolder + dataname

        does_exist = does_file_exist(datapath)
        if not does_exist:
            displace_galaxy(subhalo_ID, snapshot, radius, v_f, 0.05, num_bins=num_bins)
            
        hdul = fits.open(datapath)
        data = hdul[0].data

        savefolder = base + (f'TNGdata_{subhalo_ID}/DC_SaveResults/JWST/')
        os.makedirs(savefolder, exist_ok=True)
        filename_jwst = (f'JWST_DCresults_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}_z1.50.hdf5')
        filepath_jwst = savefolder + filename_jwst
        savepath_jwst = create_new_savefile(filepath_jwst, snapshot, num_bins, v_f, radius, method, M = len(data[:,0,0]))

    
    else:
        datafolder = base + (f'TNGdata_{subhalo_ID}/DC_SaveResults/')
        dataname = (f'DCresults_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}_{num_bins}.hdf5')
        datapath = datafolder + dataname

        does_exist = does_file_exist(datapath)

        if not does_exist:
            IFU_datacube(subhalo_ID, snapshot, radius, v_f, num_bins = num_bins)
        
        with h5py.File(datapath, 'r') as f:
            data = f['Body']['Data'][:]

        savefolder = base + (f'TNGdata_{subhalo_ID}/DC_SaveResults/JWST/')
        os.makedirs(savefolder, exist_ok=True)
        filename_jwst = (f'JWST_DCresults_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.hdf5')
        filepath_jwst = savefolder + filename_jwst
        savepath_jwst = create_new_savefile(filepath_jwst, snapshot, num_bins, v_f, radius, method, M = len(data[:,0,0]))

    
    wavelength = np.linspace(3540.5, 7409.6, 4300)
    obs_wavelength0 = correct_by_redshift(wavelength, snapshot)
    obs_wavelength = np.linspace(obs_wavelength0.min(), obs_wavelength0.max(), data.shape[0])

    """ A. Spectral convolution: """
    with h5py.File(savepath_jwst, 'a') as f:
        data_cube = f['Body']['Data']
        data_mag = f['Body']['Data_mag']
        checks = f['Body']['Checks']

        noise = 0
        
        k = -1
        started = False
        for i in range(data.shape[1]):
            start = time.time()
            for j in range(data.shape[2]):
                k+=1
    
                if bool(checks[i,j]) == bool(0):
                    
                    if started == False:
                        print(f'Starting with cell {k}/{num_bins*num_bins}\nRow {k//num_bins}/{num_bins}')

                    started = True

                    obs_spectrum = data[:, i, j]
                    convolved_spectrum = spectral_convolution(obs_wavelength, obs_spectrum, instpath, snapshot)
            
                    mag_sup0 = superficial_magnitude(convolved_spectrum, obs_wavelength, binsize, filterpath)
                    cal_spectrum = data[:, int(num_bins/2), int(num_bins/2)]
                    median_spec = np.median(obs_spectrum)
                    
                    #noise = add_noise(obs_wavelength, median_spec) #* 29 / 24
                    final_spectrum = convolved_spectrum + noise
                    
                    data_cube[:, i, j] = final_spectrum
                    
                    mag_sup = superficial_magnitude(final_spectrum, obs_wavelength, binsize, filterpath)
                    data_mag[i, j] = mag_sup
                    
                    checks[i,j] = bool(1)
                                
                else:
                    continue
                    
            if started==True:
                end = time.time()
                print(f'{i}/{num_bins} JWST Memory usage: {get_memory_usage()} MB ; Time: {end-start}', end='\r')

    
    with h5py.File(savepath_jwst, 'r') as f:
        data_cube = f['Body']['Data'][:]
        data_mag = f['Body']['Data_mag'][:]

    """ B. Crop to JWST FOV: """
    obs_wavelength = np.linspace(obs_wavelength0.min(), obs_wavelength0.max(), data_cube.shape[0])
    jwst_pix = 30

    if num_bins < jwst_pix:
        raise ValueError(f'num_bins ({num_bins}) must be greater or equal to {jwst_pix}')

    if num_bins % jwst_pix == 0:
        size = num_bins // jwst_pix
        jwst_cube = data_cube.reshape((data_cube.shape[0], jwst_pix, size, jwst_pix, size))
        jwst_cube = jwst_cube.mean(axis=(2, 4))
        
        jwst_mag = data_mag.reshape((jwst_pix, size, jwst_pix, size))
        jwst_mag = jwst_mag.mean(axis=(1,3))

    else:
        jwst_cube = np.zeros((data_cube.shape[0], jwst_pix, jwst_pix))
        scale_pix = (data_cube.shape[1] * data_cube.shape[2]) / (float(jwst_pix * jwst_pix))
        print("Escalado:" , '{:06.3f}'.format(scale_pix))
        
        for i1 in range(data_cube.shape[0]):
            temp1 = np.log10(data_cube[i1,:,:])
            temp2 = resize(temp1, (jwst_pix, jwst_pix))
            temp3 = 10.**(temp2)
            #jwst_cube[i1,:,:] = temp3 * scale_pix
            jwst_cube[i1, :, :] = resize(data_cube[i1, :, :], (jwst_pix, jwst_pix))

        jwst_mag = np.zeros((data_mag.shape[0], data_mag.shape[1]))
        scale_pix = (data_mag.shape[0] * data_mag.shape[1]) / (float(jwst_pix * jwst_pix))
        temp1 = np.log10(data_mag[:,:])
        temp2 = resize(temp1, (jwst_pix, jwst_pix))
        temp3 = 10.**(temp2)
        #jwst_mag = temp3 * scale_pix
        jwst_mag = resize(data_mag, (jwst_pix, jwst_pix))
        
        print(scale_pix)
    
        
    """ C. Spatial convolution: """
    for idx, k in enumerate(obs_wavelength):
        fwhm_pix = gaussian_fwhm(k, binsize)
        sigma = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
        # print(f'For wavelength {k} sigma is: {sigma} and binsize: {binsize}')
        jwst_cube[idx, :, :] = gaussian_filter(jwst_cube[idx, :, :], sigma)

    for i in range(jwst_cube.shape[1]):
        for j in range(jwst_cube.shape[2]):
            final_spectrum = jwst_cube[:, i, j]
            mag_sup = superficial_magnitude(final_spectrum, obs_wavelength, binsize, filterpath)
            jwst_mag[i, j] = mag_sup
        print(f'{i}/{num_bins} 2nd JWST Memory usage: {get_memory_usage()} MB ', end='\r')

    
    dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/JWST/')
    os.makedirs(dcfolder, exist_ok=True)
    dcfilename = (f'JWST_DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
    dcfilepath = dcfolder + dcfilename

    header = fits.Header()
    header['CTYPE1'] = 'X'
    header['CUNIT1'] = 'arcsec'
    header['CRPIX1'] = num_bins/2
    header['CRVAL1'] = 0
    header['CDELT1'] = binsize
    
    header['CTYPE2'] = 'Y'
    header['CUNIT2'] = 'arcsec'
    header['CRPIX2'] = num_bins/2
    header['CRVAL2'] = 0
    header['CDELT2'] = binsize

    header['CTYPE3'] = 'WAVELENGTH'
    header['CUNIT3'] = 'angstrom'
    header['CRPIX3'] = 1                                  # Reference pixel
    header['CRVAL3'] = obs_wavelength[0]                  # Wavelength at the reference pixel
    header['CDELT3'] = np.mean(np.diff(obs_wavelength))   # Wavelength increment per pixel
    header['SPECRES'] = 2.51

    header['BUNIT'] = 'erg/s/cm2/AA/arcsec2'
    hdu = fits.PrimaryHDU(data=jwst_cube, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(dcfilepath, overwrite=True)

    dcfilename_mag = (f'JWST_MI_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}_{num_bins}.fits')
    dcfilepath_mag = dcfolder + dcfilename_mag

    header = fits.Header()
    header['CTYPE1'] = 'X'
    header['CUNIT1'] = 'arcsec'
    header['CRPIX1'] = num_bins/2
    header['CRVAL1'] = 0
    header['CDELT1'] = binsize
    header['CTYPE2'] = 'Y'
    header['CUNIT2'] = 'arcsec'
    header['CRPIX1'] = num_bins/2
    header['CRVAL1'] = 0
    header['CDELT1'] = binsize

    hdu = fits.PrimaryHDU(data=jwst_mag, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(dcfilepath_mag, overwrite=True)
    
    print('JWST FITS file created !')                


def gaussian_fwhm(wavelength, spaxel_size):
    ''' Calculate Gaussian standard deviation based on wavelength. '''
    min_fwhm = 0.03  # arcsec
    max_fwhm = 0.16  # arcsec
    min_wl = 10000   # A
    max_wl = 47000   # A
    fwhm_arcsec = min_fwhm + (max_fwhm - min_fwhm) * (wavelength - min_wl) / (max_wl - min_wl)
    fwhm_pix = fwhm_arcsec / spaxel_size
    return fwhm_pix



def calculate_fwmh(wavelength, instpath, snapshot):
    
    fwmh_MILES = 2.51  # Angstrom, 0.9 A/pix MILES
    fwmh_MILES_pix = 2.51
    redshift = snapshot_to_redshift(snapshot)

    with fits.open(instpath) as hdul:
        data = hdul[1].data
        header = hdul[0].header              
    
    wavelength_inst = data['WAVELENGTH'] * 1e4
    resolution_inst = data['R']
    wavelength_sized = np.linspace(wavelength.min(), wavelength.max(), 1001)
    R = np.interp(wavelength_inst, wavelength_sized, resolution_inst) 
    fwmh_inst =  wavelength_sized / (R * (1+redshift))
                           
    fwmh = np.sqrt(fwmh_inst**2 - fwmh_MILES_pix**2) 
    return fwmh, wavelength_sized


def spectral_convolution(wavelength, spectrum, instpath, snapshot):

    fwmh, wavelength_sized = calculate_fwmh(wavelength, instpath, snapshot)
    convolved_flux = np.zeros_like(spectrum)

    for i in range(len(wavelength_sized)):
        sigma = fwmh[i] / (2*np.sqrt(2 * np.log(2))) / 0.9
        convolved_flux = gaussian_filter1d(spectrum, sigma=sigma, mode='constant')
    return convolved_flux


def add_noise(wavelength, median_spec):
    random_noise = np.random.normal(0, 1, len(wavelength))
    noise = random_noise * median_spec / 5
    return noise
' ----------------------------------------------------------------------------------'



def displace_galaxy(subhalo_ID, snapshot, radius, v_f, sampling, *, num_bins=None, binsize=None):
    """
    @author: amonreal
    """
    
    if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
        raise ValueError("You must provide either num_bins or binsize, but not both.")
    
    if num_bins is None:
        num_bins = calculate_num_bins(snapshot, radius, binsize)
    elif binsize is None:
        binsize = spaxel_size(snapshot, radius, num_bins)

    instpath = base +'Filters/jwst_nirspec_prism_disp.fits'
    if snapshot == 99:
        filterpath = base +'Filters/SLOAN_SDSS.gprime_filter.dat'

    elif snapshot == 40:
        filterpath = base +'Filters/JWST_NIRCam.F162M.dat'
    if snapshot != 99:
        raise ValueError("You must provide the snapshot at z=0, i.e. snapshot=99")

    dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/')    
    dcfilename = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
    dcfilepath = dcfolder + dcfilename
    cubein = dcfilepath
    myzin = 0.0044983025884921465      # corresponding to a distance of 20Mpc using Planck18
    myzout = 1.5
    mypath = base + (f'TNGdata_{subhalo_ID}/DataCubes/')    
    mysampling = sampling
    
    def writemycube(mydata, myoutputfile, header, newcdelt2, lstepharmoni, liniharmoni):
        
        # hdu = fits.PrimaryHDU()
        # hdu.header['MYWORD'] = ('Value', 'MyComment')
        # hdu = fits.ImageHDU(mydata, name='DATA')

        header = fits.Header()
        header['CTYPE1'] = ('X', 'Mycomment')
        header['CTYPE2'] = ('Y', 'Mycomment')
        header['CTYPE3'] = ('WAVELENGTH', 'Mycomment')
        header['CUNIT1'] = ('arcsec', 'Mycomment')
        header['CUNIT2'] = ('arcsec', 'Mycomment')
        header['CUNIT3'] = ('angstrom', 'Mycomment')
        header['CDELT1'] = (newcdelt2 * 1e-3, 'Mycomment')
        header['CDELT2'] = (newcdelt2 * 1e-3, 'Mycomment')
        header['CDELT3'] = (lstepharmoni, 'Mycomment')
        header['CRVAL3'] = (liniharmoni, 'Mycomment')
        header['CRPIX3'] = (1, 'Mycomment')
        header['BUNIT'] = ('erg/s/cm2/AA/arcsec2', 'Mycomment')
        header['SPECRES'] = (mylamcube[0] / 2000., 'Provisional value for test')

        hdu = fits.PrimaryHDU(data=mydata, header=header)
        hdul = fits.HDUList([hdu])
        
        #hdulist = fits.HDUList(hdu0, hdu)
        hdul.writeto(myoutputfile, overwrite=True)

    dcfolder_out = base + (f'TNGdata_{subhalo_ID}/DataCubes/')    
    dcfilename_out = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}_z{myzout:4.2f}.fits')
    dcfilepath_out = dcfolder_out + dcfilename_out

    magfilename_out = (f'MI_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}_z{myzout:4.2f}.fits')
    magfilepath_out = dcfolder_out + magfilename_out
    
    cubeout_dc = os.path.splitext(cubein)[0] + '_z' + \
        '{:4.2f}'.format(myzout) + '.fits'

    cubeout_mag = 'MI_' + os.path.splitext(cubein)[0] + '_z' + \
        '{:4.2f}'.format(myzout) + '.fits'

    with fits.open(cubein) as hdul:
        data_cube = hdul[0].data
        header = hdul[0].header
    
    wcs = WCS(header)
    
    lamstart = header['CRVAL3']
    lamstep = header['CDELT3']
    npix = data_cube.shape[0]
    mylamcube = lamstart + np.arange(0,npix) * lamstep
    
    # Creating the redshifted lambda
    mylamcube_z = mylamcube * (1. + myzout) / (1. + myzin)
    # Creating a wavelength vector with the spectral range of Harmoni
    liniharmoni = mylamcube_z[0]
    lfinharmoni = mylamcube_z[-1]
    lstepharmoni = 2.0
    
    mylamHarmoni = np.arange(liniharmoni,lfinharmoni,lstepharmoni)
    mydata = np.zeros((len(mylamHarmoni),data_cube.shape[1],data_cube.shape[2]))
    
    # We have to interpolate (mylamcube_z,flux) to (mylamHarmoni,fluxinter)
    for i1 in range(data_cube.shape[1]):
        for i2 in range(data_cube.shape[2]):
            myflux = data_cube[:,i1,i2]
            f = interp1d(mylamcube_z,myflux, bounds_error=False, fill_value="extrapolate")
            mydata[:,i1,i2] = f(mylamHarmoni) #* myflux.shape[0] / data_cube.shape[0] 
            
            data_mag = superficial_magnitude(mydata[:, i1, i2], mylamHarmoni, binsize, filterpath)
            
    d_Lout = cosmo.luminosity_distance(myzout)
    d_Lin = cosmo.luminosity_distance(myzin)
    mydata = mydata / d_Lout**2 * d_Lin**2 #/ 1.E20
    d_Aout = cosmo.angular_diameter_distance(myzout).value
    d_Ain = cosmo.angular_diameter_distance(myzin).value

    cdeltx = header['CDELT1'] * 1e3 # * u.deg.to(u.mas)
    cdelty = header['CDELT2'] * 1e3 # u.deg.to(u.mas)
    
    newcdeltx = cdeltx * d_Ain / d_Aout
    newcdelty = cdelty * d_Ain / d_Aout

    print("Previous mas / pix of the input cube: ", \
          '{:06.3f}'.format(cdeltx), 'mas')
    print("New mas / pix for the input cube: ", \
          '{:06.3f}'.format(newcdeltx),  'mas')
    
    # These are the mas/pix of the entry cube mydata is still in flux/ spaxel
    print("Empezamos con el rebineado")
    newcdelt2 = mysampling * 1e3
    print(newcdelt2)
    xpix = int(mydata.shape[2] / newcdelt2 * newcdeltx)
    ypix = int(mydata.shape[1] / newcdelt2 * newcdelty)
    print(xpix, ypix)
    
    mydata2 = np.zeros((len(mylamHarmoni),ypix,xpix))
    mydata2_mag = np.zeros((ypix, xpix))
   
    escalado = (mydata.shape[1]*mydata.shape[2]) / (float(ypix*xpix))
    
    for i1 in range(len(mylamHarmoni)):
        temp1 = np.log10(mydata[i1,:,:])
        temp2 = resize(temp1, (ypix, xpix))
        temp3 = 10.**(temp2)
        mydata2[i1, :, :] = resize(mydata[i1, :, :], (ypix, xpix))
        
    mydata2_mag = resize(data_mag, (ypix, xpix))

    print("Fin rebineado")
    writemycube(mydata2, dcfilename_out, header, newcdelt2, lstepharmoni, liniharmoni)
    writemycube(mydata2_mag, magfilepath_out, header, newcdelt2, lstepharmoni, liniharmoni)

    print('FITS file created!')



def resize_binsize(binsize, snapshotin, snapshotout, radius):
    if snapshotin == 99:
        zin = 0.0044983025884921465
    else:
        zin = snapshot_to_redshift(snapshotin)
        
    zout = snapshot_to_redshift(snapshotout)
    
    d_Aout = cosmo.angular_diameter_distance(zout).value
    d_Ain = cosmo.angular_diameter_distance(zin).value
    
    newbinsize = binsize * d_Aout / d_Ain
    return newbinsize 



def plotting_fluxes(subhalo_ID, snapshot, radius, v_f, method='Interpolation', i_ls = False, normalization=False, jwst=False, harmoni=False, *, binsize=None, num_bins=None):

    if harmoni == False:
            
        if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
            raise ValueError("You must provide either num_bins or binsize, but not both.")
        
        if num_bins is None:
            num_bins = calculate_num_bins(snapshot, radius, binsize)
        elif binsize is None:
            binsize = spaxel_size(snapshot, radius, num_bins)
            
        print('Binsize: ', binsize)
        print('Number of bins before: ', num_bins)
    
        if jwst == True:
    
            dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/JWST/')
            dcfilename = (f'JWST_DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
            dcfilepath = dcfolder + dcfilename
            
            dcfilename_mag = (f'JWST_MI_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{method}_{num_bins}.fits')
            dcfilepath_mag = dcfolder + dcfilename_mag
    
    
        else:
            dcfolder = base + (f'TNGdata_{subhalo_ID}/DataCubes/')
            dcfilename = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
            dcfilepath = dcfolder + dcfilename
            
            dcfilename_mag = (f'MI_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}.fits')
            dcfilepath_mag = dcfolder + dcfilename_mag
    
        does_exist = does_file_exist(dcfilepath)
        if not does_exist:
            print(dcfilepath)
            raise ValueError('Datacube does not exist: run DataCube function to obtain it')
    
        print('Opening file: ', dcfilepath)
        ''' 1. Open data files:'''
        hdul = fits.open(dcfilepath)
        data_cube = hdul[0].data
        header = hdul[0].header
        crval1 = header['CRVAL3']
        cdelt1 = header['CDELT3']
        crpix1 = header['CRPIX3']
    
        hdul_mag = fits.open(dcfilepath_mag)
        data_mag = hdul_mag[0].data
    
        if snapshot == 99:
            #pixs = [(0,0), (0,-8), (0,-16)]
            pixs = [(0,0), (0, -20), (0,-40)] 
            rad = 5
    
        elif snapshot == 40:
            pixs = [(0,0), (0,-0.3), (0, -0.6)]
            rad = 0.08
    
        data_mag[data_mag==0] = 50
        data_mag[np.isinf(data_mag)] = 50
        data_mag[np.isnan(data_mag)] = 50
        
        nx, ny = data_mag.shape
    
        x_extent = (-nx/2 * binsize * h, nx/2 * binsize * h) 
        y_extent = (-ny/2 * binsize * h, ny/2 * binsize * h) 
    
        fig = plt.figure(figsize=([25,9]))
        gs = fig.add_gridspec(2, 2, width_ratios=[1,2], height_ratios=[3,1], wspace=0.1, hspace=0.1)
        
        #fig, axs = plt.subplots(2, 2, figsize=[24, 9], gridspec_kw={'width_ratios': [1, 2]}, constrained_layout=True)
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        
        # 1. Main plot:
        ax_main = fig.add_subplot(gs[:,0])
        im = ax_main.imshow(data_mag.T, cmap='gray_r', extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]], vmin=23, vmax=30)
        ax_main.text(0.02, 0.95, (f'Subhalo ID: {subhalo_ID}'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_main.text(0.02, 0.9, (f'Redshift: {snapshot_to_redshift(snapshot)}'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_main.text(0.02, 0.85, (f'Radius: {round(radius)} kpc'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')    
        ax_main.text(0.02, 0.8, (f'Bin scale: {binsize * 30 / num_bins:.2f} arcsec'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_main.text(0.02, 0.75, (f'LOS: {v_f}'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_main.text(0.02, 0.7, (r'10$^{M/M_{\odot}}$=10.70'), fontsize = 17, color='white', transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_main.set_xlabel('x [arcsec]', fontsize=18)
        ax_main.set_ylabel('y [arcsec]', fontsize=18)
        cbar = fig.colorbar(im, ax=ax_main, orientation='horizontal', pad=0.1, fraction=0.05)
        cbar.set_label(r"${\rm mag/arcsec^2}$", fontsize = 18)
        
        # 2. Top right:
        ax_1 = fig.add_subplot(gs[0, 1])
    
        # 3. Bottom right:
        ax_2 = fig.add_subplot(gs[1,1])
        
        for i in range(len(pixs)):
                
            aperture = CircularAperture(pixs[i], r=rad)
            ap_patches = aperture.plot(color=colors[i], lw=4, ax=ax_main)
            pix = pixs[i]
            
            xpix = int((pix[0] - x_extent[0]) / binsize)
            ypix = int((pix[1] - y_extent[0]) / binsize)
    
            xpix0 = int((0 - x_extent[0]) / binsize)
            ypix0 = int((0 - y_extent[0]) / binsize)
            
            spectrum0 = data_cube[:, xpix0, ypix0]
            
            if 0 <= xpix < data_cube.shape[1] and 0 <= ypix < data_cube.shape[2]:
                
                spectrum = data_cube[:, xpix, ypix]
                num_points = len(spectrum)
                wavelength = crval1 + (np.arange(num_points) + 1 - crpix1) * cdelt1
                max_spec = np.max(spectrum)
                ax_1.plot(wavelength, spectrum, color=colors[i])
                ax_2.plot(wavelength, (spectrum * spectrum0 / max_spec) /spectrum0  , color=colors[i])
              
            else:
                print(f"Warning: Pixel indices out of bounds for position {pix}: xpix={xpix}, ypix={ypix}")
    
        redshift = snapshot_to_redshift(snapshot)
        
        ax_1.plot(wavelength, spectrum, color=colors[i])
    
        ax_2.plot(wavelength, spectrum * spectrum0 / max_spec  , color=colors[i])
       
        ax_2.text(0.05, 0.85, r'H$\beta$', fontsize = 15, color='black', transform=ax_2.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_2.text(0.92, 0.85, r'Mg', fontsize = 15, color='black', transform=ax_2.transAxes, verticalalignment='center', horizontalalignment='left')
        ax_1.set_ylabel('Flux [erg/s/cm$^2$/$\AA$/arcsec$^2]$', fontsize=18)
        ax_2.set_xlabel('Observed wavelength[$\AA$]', fontsize=18)
        ax_main.tick_params(labelsize=15)
        ax_1.tick_params(labelsize=15)
        ax_2.tick_params(labelsize=15)
    
        plt.tight_layout()
        
        if jwst== True:
            plt.savefig(f'DC_JWST_{subhalo_ID}_{snapshot}_{round(radius)}.png')
    
        else:
            plt.savefig(f'DC_{subhalo_ID}__{snapshot}_{round(radius)}.png')
            
        plt.show()

    elif harmoni == True:

        if (num_bins is None and binsize is None) or (num_bins is not None and binsize is not None):
            raise ValueError("You must provide either num_bins or binsize, but not both.")
        
        if num_bins is None:
            num_bins = calculate_num_bins(snapshot, radius, binsize)
        elif binsize is None:
            binsize = spaxel_size(snapshot, radius, num_bins)
        
        dcfolder = '/home/fernarua/Downloads/'
        dcfilename = (f'DC_snap0{snapshot}_{round(radius)}kpc_{v_f[0]:0.1f}_{v_f[1]:0.1f}_{v_f[2]:0.1f}_{num_bins}_noiseless_obj.fits')
        dcfilepath = dcfolder + dcfilename
        
        hdul=fits.open(dcfilepath)
        data_cube = hdul[0].data
        header = hdul[0].header
        crval1 = header['CRVAL3']
        cdelt1 = header['CDELT3']
        crpix1 = header['CRPIX3']
        
        if snapshot == 99:
            #pixs = [(0,0), (0,-8), (0,-16)]
            pixs = [(-50, 0), (50,0)] 
            rad = 5
        
        elif snapshot == 40:
            pixs2 = [(0,0), (0,-0.02), (0,-0.04)]
            rad = 0.005
        
        fig, axs = plt.subplots(1, 2, figsize=[24, 9], gridspec_kw={'width_ratios': [1, 2]}, constrained_layout=True)

        x_extent = [-0.11, 0.11]
        y_extent = [-0.11, 0.11]   
        
        im = axs[0].imshow(data_cube[50, :, :].T, extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]))
        colors = ['tab:blue', 'tab:green', 'tab:orange']
        
        pixs = [(5, 5), (5, 3), (5,2)] 
        for i in range(len(pixs)):
                
            aperture = CircularAperture(pixs2[i], r=rad)
            ap_patches = aperture.plot(color=colors[i], lw=4, ax=axs[0])
            pix = pixs[i]
            
            xpix = int((pix[0] - x_extent[0]) / binsize - 27)
            ypix = int((pix[1] - y_extent[0]) / binsize - 27)

            xpix = pix[0]
            ypix=pix[1]
            
            if 0 <= xpix < data_cube.shape[1] and 0 <= ypix < data_cube.shape[2]:
                
                spectrum = data_cube[:, xpix, ypix]
                num_points = len(spectrum)
                wavelength = crval1 + (np.arange(num_points) + 1 - crpix1) * cdelt1
                
                axs[1].plot(wavelength, spectrum, color=colors[i])
            else:
                print(f"Warning: Pixel indices out of bounds for position {pix}: xpix={xpix}, ypix={ypix}")
                
        axs[0].text(0.02, 0.95, (f'Subhalo ID: {subhalo_ID}'), fontsize = 19, color='white', transform=axs[0].transAxes, verticalalignment='center', horizontalalignment='left')
        
        axs[0].text(0.02, 0.9, (f'Redshift: {snapshot_to_redshift(snapshot)}'), fontsize = 19, color='white', transform=axs[0].transAxes, verticalalignment='center', horizontalalignment='left')
        
        axs[0].text(0.02, 0.85, (f'Bin scale: {binsize:.3f} arcsec'), fontsize = 19, color='white', transform=axs[0].transAxes, verticalalignment='center', horizontalalignment='left')
        
        axs[0].set_xlabel('x [arcsec]', fontsize=18)
        axs[0].set_ylabel('y [arcsec]', fontsize=18)
        axs[0].tick_params(labelsize=16)
        axs[1].set_xlim(1.05, 1.3)
        axs[1].set_ylim(0, 0.002)
        axs[1].set_xlabel(r'Wavelength [micron]', fontsize=18)
        axs[1].set_ylabel('Electron', fontsize=18)
        
        
        
        axs[1].tick_params(labelsize=16)
        

        plt.tight_layout()
        plt.savefig('Prueba.png')
        plt.show()