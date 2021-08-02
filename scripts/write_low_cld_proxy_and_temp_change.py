#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning)
from estimated_inv_strength import estimated_inversion_strength

def get_final_nyr_mean(dt, n=2):
    return dt[-12*n:, :, :].mean('time')

def write_low_cld_proxy_and_temp_change(exp_grp, ds_arr, dt_dir='./data'):
    var_names = ['lts', 'eis', 'ELF', 'temp_2m', 'temp700']
    #('month', 'lat', 'lon')
    diff_ds = {}
    for varnm in var_names:
        if not 'temp700' in varnm:
            dims = ds_arr[0][varnm].dims
            diff_ds[varnm] = (dims, ds_arr[1][varnm] - get_final_nyr_mean(ds_arr[0][varnm]))
        else:
            varnm = 'temp'
            dims = ds_arr[0]['t_surf'].dims
            diff_ds['temp700'] = (dims, ds_arr[1][varnm].sel(pfull=7e2) 
                             - get_final_nyr_mean(ds_arr[0][varnm].sel(pfull=7e2)))
    coords = {}
    for d in dims:
        coords[d] = ds_arr[0][d]
    diff_ds = xr.Dataset(diff_ds, coords=coords)
    fn = P(dt_dir, 'low_cld_proxy_and_temp_changes_'+exp_grp+'.nc')
    diff_ds.to_netcdf(fn, mode='w', format='NETCDF3_CLASSIC')
    print(fn, 'saved.')

if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    
    save_dt_dir = '../data/'
    if not os.path.exists(save_dt_dir):
        os.mkdir(save_dt_dir)

    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')

    exp_tbl = pd.read_csv('isca_qflux_exps.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']

    for exp_grp in exp_grps:
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        
        for file_nm in file_nms:
            fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

        # Keep the time coordinates the same
        ds_arr[1]['time'] = ds_arr[0].time
        exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']

        for ds in ds_arr:
            #add_datetime_info(ds)
            if not 'lts' in ds.variables:
                print('Calc lts and eis...')
                estimated_inversion_strength(ds)
        print('Read data finished.')

        #write_cld_amt_and_proxy(exp_grp, ds_arr, dt_dir=save_dt_dir)
        write_low_cld_proxy_and_temp_change(exp_grp, ds_arr, dt_dir=save_dt_dir)

    print('Done')
