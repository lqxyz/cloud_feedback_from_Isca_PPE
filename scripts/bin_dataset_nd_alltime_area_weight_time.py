# Divide the data into different bins according to
# omega500, ELF, LTS or EIS

from __future__ import print_function
import numpy as np
import xarray as xr
import os
import sys
from scipy import interpolate, stats
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning)
import copy

# Global variable for area mean
dst_varnm = ''

def weight_avg(group):
    # https://pbpython.com/weighted-average.html
    #print(dst_varnm)
    dg = group[dst_varnm]
    w = group['area_weight']
    # print(dg.dims)
    # print(w.dims)
    # print(dg)
    # print(w)
    # print('sum0', dg.sum(axis=-1, skipna=False))
    # print('sum1', (dg * w).sum(axis=-1, skipna=False))
    # print('sum2', w.sum(axis=-1, skipna=False))
    # if skipna is Ture, then the T at 1000hPa would be very small...
    #res = (dg * w).sum(axis=-1, skipna=True) / w.sum(axis=-1, skipna=True)
    # res = (dg * w).sum(axis=-1, skipna=False) / w.sum(axis=-1, skipna=False)
    # res2 = (dg * w).sum(axis=-1, skipna=True) / w.sum(axis=-1, skipna=True)
    res = (dg * w).sum(axis=-1, skipna=True) / (~xr.ufuncs.isnan(dg) * w).sum(axis=-1, skipna=True)
    # print('resut', res)
    # print('resut2', res2)
    # print('resut3', res3)
    #print(res)
    return res

def bin_4d_data(ds, bins, grp_time_var='year', bin_var_nm='ELF'):
    """
    ds: a xarray dataset containing several variables
    """
    global dst_varnm
    if grp_time_var is not None:
        ds_grp = ds.groupby(grp_time_var).mean('time')
        ntime = len(ds_grp[grp_time_var])

        nbins = len(bins)
        nlev = len(ds.pfull)
        pdf = np.ones((ntime, nbins-1)) * np.nan
        ds_bin_mean = {}

        for var in ds.variables:
            if len(ds[var].shape) == 4:
                ds_bin_mean[var] = np.ones((ntime, nlev, nbins-1)) * np.nan
            # if 'area_weight' in var:
            #     ds_bin_mean['area_sum'] = np.ones((nbins-1)) * np.nan
            if 'area_weight' in var:
                ds_bin_mean['area_sum'] = np.zeros((ntime, nbins-1))
    
        for i in range(ntime):
            if grp_time_var is not None:
                ds_i = ds_grp.isel({grp_time_var: i})
            else:
                ds_i = ds.isel({'time':i})
            pdf[i,:] = np.histogram(ds_i[bin_var_nm], bins=bins, density=True)[0]
            # grouped = ds_i.groupby_bins(bin_var_nm, bins).mean()
            # grouped_t = grouped.transpose()
            # for var in ds.variables:
            #     if len(ds[var].shape) == 4:
            #         ds_bin_mean[var][i,:,:] = grouped_t.variables.get(var)
            grouped = ds_i.groupby_bins(bin_var_nm, bins)
            for var in ds.variables:
                if len(ds[var].shape) == 4:
                    dst_varnm = var
                    g_w = grouped.apply(weight_avg)
                    g_t = g_w.transpose('pfull', bin_var_nm+'_bins')
                    # g_t = grouped.mean().transpose('pfull', bin_var_nm+'_bins').variables.get(var)
                    ds_bin_mean[var][i,:,:] = g_t
                    #print(ds.pfull[0:5], g_t[0:5,:])
                if 'area_weight' in var:
                    g_s = grouped.sum()
                    ds_bin_mean['area_sum'][i,:] = g_s.variables.get('area_weight')
    else:
        ds_bin_mean = {}
        pdf = np.histogram(ds[bin_var_nm], bins=bins, density=True)[0]
        # grouped = ds.groupby_bins(bin_var_nm, bins).mean(skipna=True)
        # grouped_t = grouped.transpose()
        # for var in ds.variables:
        #     if len(ds[var].shape) == 4:
        #         ds_bin_mean[var] = grouped_t.variables.get(var)
        grouped = ds.groupby_bins(bin_var_nm, bins)
        for var in ds.variables:
            if len(ds[var].shape) == 4:
                dst_varnm = var
                g_w = grouped.apply(weight_avg)
                g_t = g_w.transpose('pfull', bin_var_nm+'_bins')
                ds_bin_mean[var][:,:] = g_t #.variables.get(var)
            if 'area_weight' in var:
                g_s = grouped.sum()
                ds_bin_mean['area_sum'][:] = g_s.variables.get('area_weight')

    return pdf, ds_bin_mean

def bin_3d_data(ds, bins, grp_time_var='year', bin_var_nm='ELF'):
    """
    ds: a xarray dataset containing several variables
    """
    global dst_varnm

    if grp_time_var is not None:
        ds_grp = ds.groupby(grp_time_var).mean('time')
        ntime = len(ds_grp[grp_time_var])

        nbins = len(bins)
        pdf = np.ones((ntime, nbins-1)) * np.nan
        ds_bin_mean = {}

        for var in ds.variables:
            if len(ds[var].shape) == 3:
                ds_bin_mean[var] = np.ones((ntime, nbins-1)) * np.nan
            # if 'area_weight' in var:
            #     ds_bin_mean['area_sum'] = np.ones((nbins-1)) * np.nan
            if 'area_weight' in var:
                ds_bin_mean['area_sum'] = np.zeros((ntime, nbins-1))

        for i in range(ntime):
            if grp_time_var is not None:
                ds_i = ds_grp.isel({grp_time_var: i})
            else:
                ds_i = ds.isel({'time':i})
            pdf[i,:] = np.histogram(ds_i[bin_var_nm], bins=bins, density=True)[0]
            
            # grouped = ds_i.groupby_bins(bin_var_nm, bins).mean()
            # for var in ds.variables:
            #     if len(ds[var].shape) == 3:
            #         ds_bin_mean[var][i,:] = grouped.variables.get(var)
            grouped = ds_i.groupby_bins(bin_var_nm, bins)
            for var in ds.variables:
                if len(ds[var].shape) == 3:
                    dst_varnm = var
                    g_w = grouped.apply(weight_avg)
                    ds_bin_mean[var][i,:] = g_w #.variables.get(var)
                if 'area_weight' in var:
                    g_s = grouped.sum()
                    ds_bin_mean['area_sum'][i,:] = g_s.variables.get('area_weight')
    else:
        ds_bin_mean = {}
        pdf = np.histogram(ds[bin_var_nm], bins=bins, density=True)[0]
        # grouped = ds.groupby_bins(bin_var_nm, bins).mean(skipna=True)
        # for var in ds.variables:
        #     v = grouped.variables.get(var)
        #     if len(ds[var].shape) == 3:
        #         ds_bin_mean[var] = v
        nbins = len(bins)
        for var in ds.variables:
            if len(ds[var].shape) == 3:
                ds_bin_mean[var] = np.ones((nbins-1)) * np.nan
            if 'area_weight' in var:
                ds_bin_mean['area_sum'] = np.zeros((nbins-1))

        grouped = ds.groupby_bins(bin_var_nm, bins)
        for var in ds.variables:
            if len(ds[var].shape) == 3:
                dst_varnm = var
                g_w = grouped.apply(weight_avg)
                ds_bin_mean[var][:] = g_w #.variables.get(var)
            if 'area_weight' in var:
                g_s = grouped.sum()
                ds_bin_mean['area_sum'] = g_s.variables.get('area_weight')

    return pdf, ds_bin_mean

def select_4d_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year',
      four_d_varnames=None):

    for vn in four_d_varnames:
        if vn not in bin_data_dict.keys():
            bin_data_dict[vn] = ds_m[vn]

    ds_bin_m = xr.Dataset(bin_data_dict,
                coords={'time': ds_m.time, 'pfull': ds_m.pfull, 
                        'lat': ds_m.lat, 'lon': ds_m.lon})

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)
    if 'lts' in bin_var_nm.lower():
        for varname, da in ds_bin_m.data_vars.items():
            ds_bin_m[varname] = da.where(ds_bin_m.lts>min(bins))
    
    pdf_m, ds_bin_mean_m = bin_4d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm= bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1] + bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'pfull':ds_m.pfull, 'bin':bins_coord}
        dims = (grp_time_var, 'pfull', 'bin')
    else:
        coords = {'pfull':ds_m.pfull, 'bin':bins_coord}
        dims = ('pfull', 'bin')

    return pdf_m, ds_bin_mean_m, dims, coords

def select_3d_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year',
      three_d_varnames=None):

    for vn in three_d_varnames:
        if vn not in bin_data_dict.keys():
            bin_data_dict[vn] = ds_m[vn]

    # ds_bin_m = xr.Dataset(bin_data_dict, coords={'time': ds_m.time, 'lat': ds_m.lat, 'lon': ds_m.lon})
    coords = {}
    for d in ds_m[vn].dims:
        coords[d] = ds_m[d]
    ds_bin_m = xr.Dataset(bin_data_dict, coords=coords)

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)
    if 'lts' in bin_var_nm.lower():
        for varname, da in ds_bin_m.data_vars.items():
            ds_bin_m[varname] = da.where(ds_bin_m.lts>min(bins))
    
    pdf_m, ds_bin_mean_m = bin_3d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm=bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1] + bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'bin':bins_coord}
        dims = (grp_time_var, 'bin')
    else:
        coords = {'bin':bins_coord}
        dims = ['bin',]

    return pdf_m, ds_bin_mean_m, dims, coords

def select_3d_obs_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='ELF', land_sea='ocean', grp_time_var='year'):
    three_d_varnames = [ 'toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                         'low_cld_amt', 'mid_cld_amt', 'high_cld_amt',
                         'tot_cld_amt', ]# 'cwp']
    for vn in three_d_varnames:
        bin_data_dict[vn] = ds_m[vn]

    ds_bin_m = xr.Dataset(bin_data_dict, coords={'time': ds_m.time, 'lat': ds_m.lat, 'lon': ds_m.lon})

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        # https://apps.ecmwf.int/codes/grib/param-db?id=172
        # Portion of land area
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)

    pdf_m, ds_bin_mean_m = bin_3d_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm=bin_var_nm)

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1]+bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], 'bin':bins_coord}
        dims = (grp_time_var, 'bin')
    else:
        coords = {'time':ds_bin_m.time, 'bin':bins_coord}
        dims = ('time', 'bin')

    return pdf_m, ds_bin_mean_m, dims, coords

def bin_obs_data(ds, s_lat=-30, n_lat=30, bin_var_nm='omega500',
        grp_time_var='year', bins=np.arange(0,1.1,0.1), land_sea='global', land_mask_dir='./data/'):

    """ Return binned data for isca dataset based on certain variable
        such as vertical pressure velocity at 500hPa (omega500), ELF,
        EIS and LTS...
    """
    ds_m = ds.where(np.logical_and(ds.lat>=s_lat, ds.lat<=n_lat), drop=True)

    ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
    ds_mask = ds_mask.where(np.logical_and(ds_mask.lat>=s_lat,ds_mask.lat<=n_lat), drop=True)
    #ds_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    bin_data_dict = {'omega500': ds_m.omega500} 

    vars_dict = {}

    ## 3d variables
    bin_data_dict2 = copy.deepcopy(bin_data_dict)
    pdf_m, ds_bin_mean_m, dims, coords2 = select_3d_obs_data(ds_m, bin_data_dict2, ds_mask,
        bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var)
    for key, val in ds_bin_mean_m.items():
        vars_dict[key] = (dims, val)
    
    vars_dict['pdf'] = (dims, pdf_m)
    ds_bin_mean_m_array = xr.Dataset(vars_dict, coords=coords2)

    return ds_bin_mean_m_array

def get_percentile(dt):
    dims = dt.dims
    coords = {}
    for d in dims:
        coords[d] = dt.coords[d]
    dt_percentile = stats.rankdata(dt) / np.size(dt) * 1e2  # Units: %
    dt_percentile = np.reshape(dt_percentile, dt.shape)
    dt_percentile = xr.DataArray(dt_percentile, dims=dims, coords=coords)
    return dt_percentile

def get_monthly_percentile(dt):
    dims = dt.dims
    coords = {}
    for d in dims:
        coords[d] = dt.coords[d]
    try:
        times = dt.time
    except:
        times = dt.month
    else:
        print('No time dim')

    dt_percentile = np.ones_like(dt) * np.nan
    for i in range(len(times)):
        i_dt = dt[i,]
        i_dt_percentile = stats.rankdata(i_dt) / np.size(i_dt) * 1e2  # Units: %
        dt_percentile[i,] = np.reshape(i_dt_percentile, i_dt.shape)
    dt_percentile = xr.DataArray(dt_percentile, dims=dims, coords=coords)
    return dt_percentile

def bin_isca_exp_nd_data(ds, s_lat=-30, n_lat=30, bin_var_nm='omega500', bin_var=None,
        grp_time_var='year', bins=np.arange(0,1.1,0.1), land_sea='global', land_mask_dir='./data',
        nd_varnames=None, nd=4):

    """ Return binned data for isca dataset based on certain variable
        such as vertical pressure velocity at 500hPa (omega500), ELF,
        EIS and LTS...
    """
    ds_m = ds.where(np.logical_and(ds.lat>=s_lat, ds.lat<=n_lat), drop=True)

    ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
    ds_mask = ds_mask.where(np.logical_and(ds_mask.lat>=s_lat,ds_mask.lat<=n_lat), drop=True)
    #ds_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    omega_coeff = 3600. * 24. / 100.
    try:
        omega500_m = ds_m.omega500 * omega_coeff
    except:
        omega_m = ds_m.omega * omega_coeff

        try:
            omega500_m = omega_m.sel(pfull=500)
        except:
            omega500_m = omega_m.interp(pfull=500)
        else:
            fint = interpolate.interp1d(np.log(ds_m.pfull), omega_m, kind='linear', axis=1)
            omega500_m = fint(np.log(np.array([500])))
            omega500_m = xr.DataArray(omega500_m[:,0,:,:], coords=[ds_m.time, ds_m.lat, ds_m.lon],
                    dims=['time', 'lat', 'lon'])

    bin_data_dict = {}

    # Add area info to dataset
    lats = ds_m.lat
    nlon = len(ds_m.lon)
    coslat = np.cos(np.deg2rad(lats))
    coslat2 = coslat / np.sum(coslat) / nlon
    # summing this over lat and lon = 1
    area_wts = np.moveaxis(np.tile(coslat2, [nlon, 1]), 0, 1)

    latlon_dims = ('lat', 'lon')
    latlon_coords = {}
    for d in latlon_dims:
        latlon_coords[d] = ds_m[d]
    area_wts = xr.DataArray(area_wts, dims=latlon_dims, coords=latlon_coords)
    bin_data_dict['area_weight'] = area_wts

    if bin_var is None:
        bin_data_dict['omega500'] = omega500_m
        #if 'lts' in bin_var_nm.lower():
        try:
            bin_data_dict['lts'] = ds_m.lts
        except:
            print('No LTS')
            #exit
        #if 'eis' in bin_var_nm.lower():
        try:
            bin_data_dict['eis'] = ds_m.eis
        except:
            print('No EIS')
            #exit
        #if 'elf' in bin_var_nm.lower():
        try:
            bin_data_dict['ELF'] = ds_m.ELF
        except:
            print('No ELF')
            #exit
    else:
        omega500_obs_t = np.ones_like(omega500_m) * np.nan
        omega500_obs_lat_range = bin_var.where(np.logical_and(bin_var.lat>=s_lat, bin_var.lat<=n_lat), drop=True)
        for t in range(len(ds_m.time)):
            omega500_obs_t[t,:,:] = omega500_obs_lat_range
        omega500_obs_t = xr.DataArray(omega500_obs_t, coords=[ds_m.time, ds_m.lat, ds_m.lon],
                dims=['time', 'lat', 'lon'])
        bin_data_dict['omega500'] = omega500_obs_t

    # Add percentile for each variable
    bin_data_dict_tmp = copy.deepcopy(bin_data_dict)
    for key, val in bin_data_dict_tmp.items():
        if 'area' not in key:
            val_percentile = get_monthly_percentile(val) #get_percentile(val)
            bin_data_dict[key + '_percentile'] = val_percentile

    bin_data_dict2 = copy.deepcopy(bin_data_dict)

    ## ====================== 4d variables ====================== ##
    if nd == 4:
        if nd_varnames is None:
            nd_varnames = ['cf', 'rh', 'sphum', 'qcl_rad', 'omega', 'temp'], # 'theta',
                #'soc_tdt_lw', 'soc_tdt_sw', 'soc_tdt_rad', ] #  'diff_m', 'diff_t'
        pdf_m, ds_bin_mean_m, dims, coords2 = select_4d_data(ds_m, bin_data_dict2, ds_mask,
                bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var,
                four_d_varnames=nd_varnames)

    ## ====================== 3d variables ====================== ##
    if nd == 3:
        if nd_varnames is None:
            nd_varnames = ['soc_olr', 'soc_olr_clr', #'flux_lhe', 'flux_t',
                            'toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                            'tot_cld_amt', 'low_cld_amt', 'mid_cld_amt', 'high_cld_amt',
                            'temp_2m', 't_surf', 'cwp'] #, 'soc_tot_cloud_cover','z_pbl'
        pdf_m, ds_bin_mean_m, dims, coords2 = select_3d_data(ds_m, bin_data_dict2, ds_mask,
            bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var,
            three_d_varnames=nd_varnames)
    
    vars_dict = {}

    for key, val in ds_bin_mean_m.items():
        if 'area' in key:
            if 'sum' in key:
                if len(dims)>1:
                    #vars_dict[key] = (dims[-1], val)
                    vars_dict[key] = ((dims[0], dims[-1]), val)
                else:
                    vars_dict[key] = ((dims[-1]), val)
            else: # neglect the area_weight
                pass
        else:
            vars_dict[key] = (dims, val)

    # print(dims)
    # for key, val in ds_bin_mean_m.items():
    #     if 'area' in key:
    #         vars_dict[key] = ((dims[0], dims[-1]), val)
    #     else:
    #         vars_dict[key] = (dims, val)

    dims2 = tuple([d for d in dims if d != 'pfull'])
    vars_dict['pdf'] = (dims2, pdf_m)
    # print(coords2)
    ds_bin_mean_m_array = xr.Dataset(vars_dict, coords=coords2)

    return ds_bin_mean_m_array
