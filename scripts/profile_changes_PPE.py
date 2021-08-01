#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from analysis_functions import add_datetime_info 
import matplotlib.pyplot as plt
import proplot as plot
import warnings
warnings.simplefilter(action='ignore')

def add_contour_label(ax, ds, lats_2d, plevel_2d, varnm='cf',
             levels=np.arange(0,110,10), label_interval=2, coeff=1):
    # class nf(float):
    #     def __repr__(self):
    #         s = f'{self:.1f}'
    #         return f'{self:.0f}' if s[-1] == '0' else s

    # Label levels with specially formatted floats
    # if plt.rcParams["text.usetex"]:
    #     fmt = r'%r \%%'
    # else:
    #     fmt = '%r %%'

    # Add contour
    dims = tuple([d for d in ds[varnm].dims if d != 'pfull' and d != 'lat'])
    cs_label = ax.contour(lats_2d, plevel_2d, ds[varnm].mean(dims, skipna=True)*coeff, 
                levels=levels, colors='gray', linewidths=1)  
    # Recast levels to new class 
    fmt = {}  
    for l in cs_label.levels:
        #print(l)
        fmt[l] = str(int(l)) #str(nf(l))
    
    # cs_label.levels = [nf(val) for val in cs_label.levels[::label_interval]] 
    # [::label_interval]
    #print(cs_label.levels[::label_interval])
    ax.clabel(cs_label, cs_label.levels[::label_interval], inline=True, fmt=fmt) # fontsize=10

def get_final_nyr_mean(dt, n=2):
    return dt[-12*n:, :, :].mean('time')

def zonal_mean_profile_change_inc_reff(ds1, ds2, figname='test.pdf',
    add_stippling=False, ds_arr1=None, ds_arr2=None, stippling_ratio=0.8):
    # ds1: xarray dataset for ctrl experiment
    # ds2: perturbed dataset

    diff_ds = ds2 - ds1

    ylim = [0, 975]
    lats = diff_ds.lat
    levels = diff_ds.pfull

    lats_2d, plevel_2d = np.meshgrid(lats, levels)

    plot.close('all')
    fig, axes = plot.subplots(nrows=2, ncols=3, aspect=(1.2, 1), share=1)

    varnms = ['temp', 'sphum', 'rh', 'cf','qcl_rad', 'reff_rad']

    var_labels = ['$\Delta$T (K)', '$\Delta$q (g kg$^{-1}$)', '$\Delta$RH (%)', '$\Delta$CF (%)',
                 '$\Delta q_{cld}$ ($10^{-2}$g kg$^{-1}$)', '$\Delta R_{eff}$ ($\mu$m)']
    cnlevels_arr = [np.arange(180,320,20), np.arange(0,15,2),
                   np.arange(0,101,10), np.arange(0,101,10),
                   np.arange(0,3.1,0.5), np.arange(10,26,1),]
    diff_cnlevels_arr = [np.arange(-20,21,2), np.arange(-20,21,2), # not sure
                       np.arange(-15,15.5,0.5),  np.arange(-10,11,1),
                        np.arange(-2,2.1,0.2), np.arange(-10,10.5,1),]

    colormap_arr = ['RdBu_r'] * len(varnms)
    extend_arr = ['both'] * len(varnms)
    
    coeff_arr = [1.] * len(varnms)
    coeff_arr[1] = 1e3 # q
    coeff_arr[3] = 1e2 # cf
    coeff_arr[4] = 1e5 # qcl
    
    for i, ax in enumerate(axes):
        varnm = varnms[i]
        var_label = '(' + chr(97+i) + ') ' + var_labels[i]
        
        cnlevels = cnlevels_arr[i]
        diff_cnlevels = diff_cnlevels_arr[i]
        colormap = colormap_arr[i]
        extend = extend_arr[i]
        coeff = coeff_arr[i]
        if 'psi' in varnm:
            mass_streamfunction(ds1)
            mass_streamfunction(ds2)
            #dst_dt = (ds2[varnm]-ds1[varnm]).mean('time', skipna=True) * coeff
            dst_dt = (ds2[varnm] - get_final_nyr_mean(ds1[varnm], n=2)).mean('time', skipna=True) * coeff
        else:
            #dst_dt = diff_ds[varnm].mean(('time', 'lon')) * coeff
            dst_dt = (ds2[varnm] - get_final_nyr_mean(ds1[varnm], n=2)).mean(('time', 'lon')) * coeff
    
        cs = ax.contourf(lats_2d, plevel_2d, dst_dt, levels=diff_cnlevels, 
                         cmap=colormap, extend=extend)
        if add_stippling and ds_arr1 is not None and ds_arr2 is not None:
            print('Calc the sign agreement...')
            max_sign = get_agreement_on_change(ds_arr1, ds_arr2, varnm)
            # stippling_ratio
            #print(np.shape(max_sign), max_sign.dims)
            #print(max_sign)
            max_sign_ratio = max_sign >= (len(ds_arr1) * stippling_ratio)
            #print(np.shape(max_sign_ratio), max_sign_ratio.dims)
            #ax.contourf(lats_2d, plevel_2d, max_sign_ratio,
            #         levels=[0,1], colors='none', hatches=['.', None ])
            max_sign_ratio = np.ma.masked_less(max_sign_ratio, 1)
            ax.pcolor(lats_2d, plevel_2d, max_sign_ratio, hatch='.', alpha=0.)

        ax.set_title(var_label)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.colorbar(cs, loc='r')
        
        add_contour_label(ax, ds1, lats_2d, plevel_2d, varnm=varnm,
                 levels=cnlevels, label_interval=2, coeff=coeff)

    axes.format(xlabel='Latitude', ylabel='Pressure (hPa)', 
        xlim=(-90, 90), xlocator=plot.arange(-60, 61, 30),
        xminorlocator=30, xformatter='deglat', xtickminor=False,
        yminorlocator=100, grid=False)

    fig.savefig(figname, transparent=False)

def get_agreement_on_change(ds_arr1, ds_arr2, varnm):
    ds = ds_arr1[0]
    #shp = (len(ds_arr), len(ds.time), len(ds.pfull), len(ds.lat), len(ds.lon))
    shp = (len(ds_arr1), len(ds.pfull), len(ds.lat))
    dt_diff = np.ones(shp, dtype='float32') * np.nan

    for i, (ds1, ds2) in enumerate(zip(ds_arr1, ds_arr2)):
        dt_diff[i,] = (ds2[varnm] - get_final_nyr_mean(ds1[varnm])).mean(('time', 'lon'))

    shp2 = (2, len(ds.pfull), len(ds.lat))
    max_sign = np.ones(shp2, dtype='float32') * np.nan
    max_sign[0,:] = np.sum(dt_diff>0, axis=0)
    max_sign[1,:] = np.sum(dt_diff<=0, axis=0)
    max_sign1 = np.max(max_sign, axis=0)
    dims = ('pfull', 'lat')
    coords = {}
    for d in dims:
        coords[d] = ds[varnm][d]
    max_sign1 = xr.DataArray(max_sign1, dims=dims, coords=coords)

    return max_sign1

def get_ensemble_mean_profile(ds_arr, varnm):
    ds = ds_arr[0]
    shp = (len(ds_arr), len(ds.time), len(ds.pfull), len(ds.lat), len(ds.lon))
    dt_sum = np.ones(shp, dtype='float32') * np.nan

    for i, ds in enumerate(ds_arr):
        dt_sum[i,] = ds[varnm]

    dt_mean = np.nanmean(dt_sum, axis=0)
    dims = ('time', 'pfull', 'lat', 'lon')
    coords = {}
    for d in dims:
        coords[d] = ds[varnm][d]
    dt_mean = xr.DataArray(dt_mean, dims=dims, coords=coords)

    return dt_mean

def estimated_tau_changes(ds1, ds2, figname='test.pdf'):
    '''tau = (3 * LWP) /(2 * rho_w * Reff)'''
    grav = 9.87
    for ds in [ds1, ds2]:
        mixing_ratio = ds.qcl_rad / (1.0 + ds.qcl_rad)
        dims_4d = ds.qcl_rad.dims
        ds['rcl_rad'] = (dims_4d, mixing_ratio)
        #ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        #lwp = ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2
        dst_dt = ds.rcl_rad
        ma_dt = np.ma.MaskedArray(dst_dt, mask=np.isnan(dst_dt))
        try:
            ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        except:
            ds['dpfull'] = (ds.pfull.dims, np.gradient(ds.pfull))
    
        lwc = - ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2
        ds['lwc'] = (dims_4d, lwc)
        cld_tau = 3 * ds.lwc / (2 * 1e3 * ds.reff_rad * 1e-6)
        ds['cld_tau'] = (dims_4d, cld_tau)

    coeff = 1
    lats_2d, plevel_2d = np.meshgrid(ds1.lat, ds1.pfull)
    plot.close('all')
    fig, ax = plot.subplots(nrows=1, ncols=1, aspect=(1.2, 1), share=1)
    
    cnlevels = np.arange(0, 5, 1)
    diff_cnlevels = np.arange(-1.5, 1.6, 0.1)
    dst_dt = (ds2.cld_tau-ds1.cld_tau).mean(('time', 'lon')) * coeff
    cs = ax.contourf(lats_2d, plevel_2d, dst_dt, levels=diff_cnlevels, 
                    cmap='RdBu_r', extend='both')
    ax.set_title('Cloud optical depth')
    ax.set_ylim([0, 1000])
    ax.invert_yaxis()
    ax.colorbar(cs, loc='r', width='1em')
    add_contour_label(ax, ds1, lats_2d, plevel_2d, varnm='cld_tau',
                levels=cnlevels, label_interval=2, coeff=coeff)

    ax.format(xlabel='Latitude', ylabel='Pressure (hPa)', 
        xlim=(-90, 90), xlocator=plot.arange(-60, 61, 30),
        xminorlocator=30, xformatter='deglat', xtickminor=False,
        yminorlocator=100, grid=False)

    fig.savefig(figname, transparent=False)


if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print('Read Isca dataset...')
    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']

    varnms = ['temp', 'sphum', 'rh', 'cf','qcl_rad', 'reff_rad']

    ds_list1 = []
    ds_list2 = []

    for exp_grp in exp_grps:
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        for file_nm in file_nms:
            fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

        # Keep the time coordinates the same
        ds_arr[1]['time'] = ds_arr[0].time
        for ds in ds_arr:
            add_datetime_info(ds)

        ds_list1.append(ds_arr[0])
        ds_list2.append(ds_arr[1])

        # print('zonal mean profile changes..')
        # fig_name = P(fig_dir, 'zonal_mean_profile_change_basic_fields_'+exp_grp+'.pdf')
        # zonal_mean_profile_change_inc_reff(ds_arr[0], ds_arr[1], figname=fig_name)

        # print('zonal mean profile changes..')
        # fig_name = P(fig_dir, 'zonal_mean_profile_change_cld_tau.pdf')
        # estimated_tau_changes(ds_arr[0], ds_arr[1], figname=fig_name)
    
    dst_ds1 = {}
    dst_ds2 = {}
    dims = ('time', 'pfull', 'lat', 'lon')
    for varnm in varnms:
        print(varnm, 'get ensemble mean for ctrl runs...')
        var1 = get_ensemble_mean_profile(ds_list1, varnm)
        dst_ds1[varnm] = (dims, var1)
        print(varnm, 'get ensemble mean for perturbed runs...')
        var2 = get_ensemble_mean_profile(ds_list2, varnm)
        dst_ds2[varnm] = (dims, var2)

    coords = {}
    for d in dims:
        coords[d] = var1[d]
    dst_ds1 = xr.Dataset(dst_ds1, coords=coords)
    dst_ds2 = xr.Dataset(dst_ds2, coords=coords)

    print('zonal mean profile changes..')
    fig_name = P(fig_dir, 'zonal_mean_profile_change_basic_fields_ensemble_mean.pdf')
    zonal_mean_profile_change_inc_reff(dst_ds1, dst_ds2, figname=fig_name)

    # print('zonal mean profile changes..')
    # fig_name = P(fig_dir, 'zonal_mean_profile_change_basic_fields_ensemble_mean_stippling.pdf')
    # zonal_mean_profile_change_inc_reff(dst_ds1, dst_ds2, figname=fig_name,
    #         add_stippling=True, ds_arr1=ds_list1, ds_arr2=ds_list2, stippling_ratio=0.8)
