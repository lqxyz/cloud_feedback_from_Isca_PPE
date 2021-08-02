import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import add_datetime_info
from isca_cre_cwp import (calc_toa_cre_for_isca, calc_total_cwp_for_isca, 
                         add_toa_net_flux_to_ds_arr)
#from bin_dataset_nd_alltime_area_weight import bin_isca_exp_nd_data
from bin_dataset_nd_alltime_area_weight_time import bin_isca_exp_nd_data

def cal_total_tendency(ds, var_names):
    dt_sum = ds[var_names[0]]
    dims = dt_sum.dims
    for var_nm in var_names[1:]:
        dt_sum = dt_sum + ds[var_nm]

    if '_tg_' in var_nm:
        sum_varnm = 'dt_tg_sum_cond_diffu_conv'
        ds[sum_varnm] = (dims, dt_sum)
    if '_qg_' in var_nm:
        sum_varnm = 'dt_qg_sum_cond_diffu_conv'
        ds[sum_varnm] = (dims, dt_sum)

    return sum_varnm

if __name__ == '__main__':
    P = os.path.join
    lsm_dir = '../inputs'
    ## ============================ Read Isca data ============================== ##
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    #exp_tbl = pd.read_csv('isca_qflux_exps_for_park_a.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    # exps_arr = list(exp_tbl.iloc[:, 1])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']

    cld_fbk_dt = '../data/'
    ds_dst_arr = []
    for exp_grp in exp_grps:
        print(exp_grp)
        # Read cloud feedbacks
        cld_fbk_fn = P(cld_fbk_dt, 'cld_fbk_decomp_v2_' + exp_grp + '.nc')
        ds_cld = xr.open_dataset(cld_fbk_fn, decode_times=False)
        #ds_dst_arr.append(ds_cld)

        dims = ds_cld.LO680_lw_cld_alt.dims

        fn = P(ppe_dir, file_nms[0].replace('.nc', '_'+exp_grp+'.nc'))
        ds = xr.open_dataset(fn, decode_times=False)
        add_datetime_info(ds)
        if not 'lts' in ds.variables:
            print('Calc lts and eis...')
            estimated_inversion_strength(ds)

        ds_cld = ds_cld.interp(lat=ds.lat, lon=ds.lon)
        for varnm in ['omega500', 'lts', 'eis', 'ELF']:
            if 'omega500' in varnm:
                var_mean = ds['omega'].sel(pfull=500).groupby('month').mean('time')
            else:
                var_mean = ds[varnm].groupby('month').mean('time')
            ds_cld[varnm] = (dims, var_mean)
    
        ds_dst_arr.append(ds_cld)

    print('Read data finished')

    cld_variables = []
    for v in ds_cld.variables:
            if 'cld' in v:
                cld_variables.append(v)

    # Begin to bin data
    land_sea_mask = 'ocean'
    grp_time = None
    s_lat = -30
    n_lat = 30

    three_d_varnames = cld_variables + ['omega500', 'eis', 'ELF', 'lts']
    # bins
    bin_nms = ['omega500', 'lts', 'eis', 'ELF',
                'omega500_percentile', 'lts_percentile', 
                'eis_percentile', 'ELF_percentile']
    bins_arr_old = [np.arange(-100, 101, 10), np.arange(0, 31, 1),
                np.arange(-5, 15, 1), np.arange(0, 1.55, 0.05),
                ] + [np.arange(0, 101, 10)] * 4
    bins_arr_refined = [np.arange(-100, 101, 5), np.arange(0, 31, 1),
                np.arange(-5, 15, 1), np.arange(0, 1.55, 0.05),
                ] + [np.arange(0, 101, 5)] * 4
    # save_dt_dir_arr = ['./data/qflux_area_weight_cld_fbk', './data/qflux_area_weight_refined_cld_fbk']
    # bins_arr_list = [bins_arr_old, bins_arr_refined]
    bins_arr_list = [bins_arr_old, ]
    save_dt_dir_arr = ['../data/qflux_area_weight_cld_fbk',]
    
    for exp_grp, ds in zip(exp_grps, ds_dst_arr):
        print('')
        print('bin', exp_grp)
        #for bins_arr, save_dt_dir in zip(bins_arr_list[0:1], save_dt_dir_arr[0:1]):
        for bins_arr, save_dt_dir in zip(bins_arr_list, save_dt_dir_arr):
            if not os.path.exists(save_dt_dir):
                os.mkdir(save_dt_dir)

            for bin_nm, bins in zip(bin_nms, bins_arr):
                print(bin_nm)
                file_id = exp_grp
                # =========== For 3d data ============== #
                print('  bin 3d data')
                nd = 3
                ds_bin = bin_isca_exp_nd_data(ds, s_lat=s_lat, n_lat=n_lat, 
                        grp_time_var=grp_time, bin_var_nm=bin_nm, bin_var=None,
                        bins=bins, land_sea=land_sea_mask, nd_varnames=three_d_varnames, nd=nd, land_mask_dir=lsm_dir)
                dt_fn = '_'.join(filter(None, ['ds_bin', bin_nm, file_id, grp_time, land_sea_mask, str(nd)+'d']))
                ds_bin.to_netcdf(P(save_dt_dir, dt_fn+'.nc'), mode='w', format='NETCDF3_CLASSIC')
                print(' ' + dt_fn + '.nc saved.')

            print('bin arr end')
    print('Done')
