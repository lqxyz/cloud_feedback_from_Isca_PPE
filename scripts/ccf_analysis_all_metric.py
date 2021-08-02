#!/usr/bin/env python
'''
Myers and Norris, 2016, GRL
Scott et al., 2020
'''
from __future__ import print_function
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from analysis_functions import add_datetime_info, detrend_dim
from isca_cre_cwp import (calc_toa_cre_for_isca, get_gm,
    calc_total_cwp_for_isca, calc_low_mid_high_cwp_for_isca)
import warnings
warnings.simplefilter(action='ignore')
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer # MissingIndicator
from calc_Tadv import calc_Tadv
import copy

def multi_linear_regress(x, y, dst_dt):
    '''
    x shape: (nvar, ntime) --> (nvar, ntime, nlat, nlon)
    y shape: (ntime) --> (ntime, nlat, nlon)
    dst_dt shape: (nvar) -> (nvar, nlat, nlon)
    '''
    # xx should be (ntime, nvar)
    xx = x.transpose() # np.moveaxis(x, 0, -1)
    # https://scikit-learn.org/stable/modules/impute.html
    #imputer = KNNImputer(n_neighbors=2, weights="uniform")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    xx_imputed = imputer.fit_transform(xx)
    #print(f"x: {x.shape} | xx: {xx.shape} | xx_imputed: {xx_imputed.shape} | y: {y.shape}")
    if xx_imputed.shape[-1] == xx.shape[-1] and (not np.any(np.isnan(y))):
        coef = LinearRegression().fit(xx_imputed, y).coef_
    else:
        coef = np.ones(xx.shape[-1], dtype='float32') * np.nan
    dst_dt = coef
    return dst_dt

def CCF_multi_linear_regress(CCF_stand_anomalies_arr, y_anomaly):
    # https://sklearn-xarray.readthedocs.io/en/latest/auto_examples/plot_linear_regression.html

    shp = list(y_anomaly.shape)
    Nx = len(CCF_stand_anomalies_arr)
    shp.insert(0, Nx)
    CCF_stand_anomalies = np.zeros(shp, dtype='float32')
    for i in range(Nx):
        CCF_stand_anomalies[i,] = CCF_stand_anomalies_arr[i]
    
    dims = ('ind', 'time', 'lat', 'lon')
    ind = np.arange(0, Nx, 1)
    coords = {}
    for d in dims:
        if 'ind' in d:
            coords[d] = ind
        else:
            coords[d] = y_anomaly[d]
    CCF_stand_anomalies = xr.DataArray(CCF_stand_anomalies, dims=dims, coords=coords)
    
    lats = y_anomaly.lat
    lons = y_anomaly.lon
    nlat = len(lats)
    nlon = len(lons)
    out_coeff = np.ones((Nx, nlat, nlon), dtype='float32') * np.nan
    out_coeff = xr.DataArray(out_coeff, dims=('new_ind', 'lat', 'lon'), 
                coords={'new_ind':ind, 'lat':lats, 'lon':lons})

    # https://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html
    out_coeff2 = xr.apply_ufunc(
            multi_linear_regress,   # first the function (x, y, dst_dt)
            CCF_stand_anomalies,    # x for func
            y_anomaly,              # y for func
            out_coeff,              # dst_dt for func
            input_core_dims=[["ind", "time"], ["time"], ['new_ind']], # list with one entry per arg
            output_core_dims=[['new_ind']],
            exclude_dims=set(("ind","time")),  # dimensions allowed to change size. Must be a set!
            vectorize=True,  # loop over non-core dims
            dask="parallelized",
            output_dtypes=[y_anomaly.dtype],  # one per output
        ).rename({"new_ind": "ind"})
    return out_coeff2

def sel_from_subsidence_region_tm(ds, slat=-30, nlat=30, ocn=True,
                               land_mask_dir='./data', method='LTS'):
    coeff = 3600. * 24. / 100.
    try:
        w500 = ds.omega.sel(pfull=500).mean('time') * coeff
        w700 = ds.omega.sel(pfull=700).mean('time') * coeff
    except:
        w500 = ds.omega.interp(pfull=500).mean('time') * coeff
        w700 = ds.omega.interp(pfull=700).mean('time') * coeff
    lats = ds.lat
    if 'LTS' in method.upper():
        w0 = 10 # hPa/day
        ind_low = (w500 >= w0) & (w700 >= w0) & (lats >= slat) & (lats <= nlat)
    else:
        w0 = 15 # hPa/day
        ind_low = (w700 >= w0) & (lats >= slat) & (lats <= nlat)

    if ocn:
        # get land_mask
        lsm_fn = os.path.join(land_mask_dir, 'era_land_t42.nc')
        ds_mask = xr.open_dataset(lsm_fn, decode_times=False)
        ds.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)
        # lsm_fn = os.path.join(land_mask_dir, 'land_sea_mask.nc')
        # ds_mask1 = xr.open_dataset(lsm_fn, decode_times=False)
        # ds_mask1 = ds_mask1.rename_dims(dims_dict={'latitude':'lat', 'longitude':'lon'}).mean('time')
        # ds_mask = ds_mask1.interp(lat=ds.lat, lon=ds.lon)
        # ds.coords['mask'] = (('lat', 'lon'), ds_mask.lsm.values)
        ind_low = ind_low & (ds.mask == 0) # (ds.mask < 0.5) #

    return ind_low

def reg_mean_coeff_from_ind(out_coeff, ind_reg):
    reg_coeff = []
    Nx = len(out_coeff.ind)
    for i in range(Nx):
        dt_ind = out_coeff.isel(ind=i).where(ind_reg, drop=True)
        coslat = np.cos(np.deg2rad(dt_ind.lat))
        dt_ind_mean = dt_ind.mean(('lon'), skipna=True)
        dt_mask = np.ma.masked_array(dt_ind_mean, mask=np.isnan(dt_ind_mean))
        dt_ind_mask_mean = np.ma.average(dt_mask, axis=0, weights=coslat)
        reg_coeff.append(dt_ind_mask_mean)
    return np.array(reg_coeff)

def reg_mean_dt_from_ind(dt, ind_reg):
    dt_ind = dt.where(ind_reg, drop=True)
    coslat = np.cos(np.deg2rad(dt_ind.lat))
    dt_ind_mean = dt_ind.mean(('time', 'lon'), skipna=True)
    dt_mask = np.ma.masked_array(dt_ind_mean, mask=np.isnan(dt_ind_mean))
    dt_ind_mask_mean = np.ma.average(dt_mask, axis=0, weights=coslat)

    return dt_ind_mask_mean

def get_final_nyr_mean(dt, n=5):
    return dt[-12*n:, :, :].mean('time')

# def get_dst_dt_from_ind(CCF_arr, cldvar, ind):
#     #Nx = len(CCF_arr)
#     dst_CCF_arr = []
#     #dst_swcre_anomaly
#     for dt in CCF_arr:
#         dt_ind = dt.where(ind, drop=True)
#         # shp = dt_ind.shape
#         # new_shp = (shp[0], shp[1]*shp[2])
#         # dst_CCF_arr.append(dt_ind.values.reshape(new_shp))
#         dst_CCF_arr.append(dt_ind)

#     # ccf_shp = (Nx, shp[0], shp[1]*shp[2])
#     # dst_CCF = np.zeros(ccf_shp, dtype='float32')
#     # for i in range(Nx):
#     #     dst_CCF[i,:,:] = dst_CCF_arr[i]
#     #dst_cldvar = cldvar.where(ind, drop=True).values.reshape(new_shp)
#     dst_cldvar = cldvar.where(ind, drop=True)
#     return dst_CCF_arr, dst_cldvar

def get_stand_anomalies_arr(var_list, var_names, ds_era5):
    stand_anomalies_arr = []
    #anomalies_arr = []
    for var, var_nm in zip(var_list, var_names):
        var_de = detrend_dim(var, 'time', deg=1, skipna=True)
        clim_mean = var_de.groupby("month").mean("time")
        #clim_std = var_de.groupby("month").std("time")
        clim_std = ds_era5[var_nm]
        stand_anomalies = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                        var_de.groupby("month"), clim_mean, clim_std)
        stand_anomalies_arr.append(stand_anomalies)
        # anomalies = var.groupby("month") - clim_mean
        # anomalies_arr.append(anomalies)
    return stand_anomalies_arr


if __name__ == '__main__':
    P = os.path.join

    dt_dir = '../data/CCF_analysis_metric'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)
    dt_nc_dir = '../data/CCF_analysis/reg_coeff_netcdf_metric'
    if not os.path.exists(dt_nc_dir):
        os.makedirs(dt_nc_dir)
    lsm_dir = '../inputs/'

    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data_10yr')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    proxy = sys.argv[1].upper() # ELF, or EIS
    if 'elf' in proxy.lower():
        proxy_var = 'ELF'
    else:
        proxy_var = 'eis'

    if len(sys.argv)>2:
        num_proxy = float(sys.argv[2])
    else:
        num_proxy = 2 # more than one proxies 

    method = sys.argv[3].upper()

    print('proxy, num of proxy, method', proxy, num_proxy, method)

    #var_names = ['SST', 'EIS', 'Tadv', r'RH$_{700}$', r'$\omega_{700}$']
    #var_names = ['SST', proxy, 'Tadv', 'RH700', 'omega700']
    #var_names = ['SST', proxy] #, 'Tadv', 'RH700', 'omega700']
    if num_proxy > 2:
        var_names = ['SST', proxy, 'Tadv', 'RH700', 'omega700']
    else:
        var_names = ['SST', proxy] #, 'Tadv', 'RH700', 'omega700']
    #var_names = ['ELF', 'Tadv', 'RH700', 'omega700']
    
    Nvar = len(var_names)
    Ngrp = len(exp_grps)

    tbl_names = ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                 'low_cld_amt', 
                 'mid_cld_amt', 'high_cld_amt', 'tot_cld_amt',
                 'cwp', 'low_cwp', 'mid_cwp', 'high_cwp',
                 ] #'tauisccp',
    file_nms= ['extracted_data_241_360.nc', 'extracted_data_601_720.nc']

    # Read the stddev of CCF from ERA5
    fn_era5 = P('../data', 'ERA5_monthly_stddev_for_CCFs.nc')
    ds_era5_old = xr.open_dataset(fn_era5, decode_times=False)

    fn = P(ppe_dir, file_nms[0].replace('.nc', '_'+ exp_grps[0]+'.nc'))
    ds1 = xr.open_dataset(fn, decode_times=False)
    ds_era5 = ds_era5_old.interp(lat=ds1.lat, lon=ds1.lon)

    CCF_change_calculated = False
    CCF_chg_tbl = np.zeros((Ngrp, Nvar))
    var_chg_tbl = np.zeros((Ngrp, Nvar))

    for tbl_var_nm in tbl_names:
        print(tbl_var_nm)
        tbl_ctrl = np.zeros((Ngrp, Nvar))
        tbl_perturb = np.zeros((Ngrp, Nvar))
        var_chg_tbl_real = np.zeros(Ngrp)

        #for kk, (exp_grp, exp) in enumerate(zip(exp_grps[0:1], exps_arr[0:1])):
        for kk, exp_grp in enumerate(exp_grps):
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
    
            if 'cre' in tbl_var_nm:
                calc_toa_cre_for_isca(ds_arr)
            if 'cwp' in tbl_var_nm:
                #calc_total_cwp_for_isca(ds)
                calc_low_mid_high_cwp_for_isca(ds_arr, var_nms=[tbl_var_nm])

            print('Get low/ocn index')
            ind_low_ocn_ctrl = sel_from_subsidence_region_tm(ds_arr[0], slat=-35, nlat=35, 
                                    ocn=True, land_mask_dir=lsm_dir)
            ind_low_ocn_perturb = sel_from_subsidence_region_tm(ds_arr[1], slat=-35, nlat=35,
                                    ocn=True, land_mask_dir=lsm_dir)
            
            ind_arr = [ind_low_ocn_ctrl, ind_low_ocn_perturb]
            for nn, (ds, ind_low) in enumerate(zip(ds_arr, ind_arr)):
                Tadv = calc_Tadv(ds.u_10m, ds.v_10m, ds.t_surf)
                var_list = [ds.t_surf, ds[proxy_var], Tadv, 
                            ds.rh.sel(pfull=700), ds.omega.sel(pfull=700)]
                if nn == 0:
                    var_list_ctrl = copy.deepcopy(var_list)
                else:
                    var_list_perturb = copy.deepcopy(var_list)

                stand_anomalies_arr = get_stand_anomalies_arr(var_list, var_names, ds_era5)
                tbl_var = ds[tbl_var_nm]
                if 'cwp' in tbl_var_nm:
                    tbl_var = tbl_var * 1e3
                clim_mean = tbl_var.groupby("month").mean("time")
                tbl_var_anomaly = tbl_var.groupby("month") - clim_mean

                print('Get multivariate linear regression coeff')
                out_coeff = CCF_multi_linear_regress(stand_anomalies_arr, tbl_var_anomaly)

                # Save the coeff to nc file
                nc_fn = P(dt_nc_dir, '_'.join([tbl_var_nm, exp_grp, proxy,  'reg_coeff.nc']) )
                print(nc_fn)
                out_coeff_ds = {}
                out_dims = out_coeff.dims
                coords = {}
                for dim in out_dims:
                    coords[dim] = out_coeff[dim]
                out_coeff_ds['reg_coef'] = (out_dims, out_coeff)
                out_coeff_ds = xr.Dataset(out_coeff_ds, coords=coords)
                out_coeff_ds.to_netcdf(nc_fn, mode='w', format='NETCDF3_CLASSIC')

                reg_coeff = reg_mean_coeff_from_ind(out_coeff, ind_low)
                print(reg_coeff)
                if nn==0:
                    tbl_ctrl[kk,:] = reg_coeff
                else:
                    tbl_perturb[kk,:] = reg_coeff

            #===================== Changes of CCF response to warming =============== #
            if not CCF_change_calculated:
                print('Calculate the CCF changes per K')
                diff_var_list = [ d2-get_final_nyr_mean(d1) for (d1, d2) in zip(var_list_ctrl, var_list_perturb) ]
                diff_ts = ds_arr[1].temp_2m - get_final_nyr_mean(ds_arr[0].temp_2m)
                coslat = np.cos(np.deg2rad(diff_ts.lat))
                diff_ts_gm = np.average(diff_ts.mean(('time', 'lon')), axis=0, weights=coslat)

                # Calculate how many sigma (from ctrl) it has changed
                CCF_chg_perK_arr = []
                for var, dt_old, var_nm in zip(diff_var_list, var_list_ctrl, var_names):
                    clim_std_old = ds_era5[var_nm] #dt_old.groupby("month").std("time")
                    n_sigma = xr.apply_ufunc(lambda x, s: x / s,
                                    var.groupby("month"), clim_std_old)
                    n_sigma_norm = n_sigma / diff_ts_gm
                    n_sigma_norm = xr.where(n_sigma_norm==np.inf, np.nan, n_sigma_norm)
                    n_sigma_norm = xr.where(n_sigma_norm==-np.inf, np.nan, n_sigma_norm)
                    CCF_chg_perK_arr.append(n_sigma_norm)

                # Save the ccf change per K to nc file
                nc_fn = P(dt_nc_dir, '_'.join([tbl_var_nm, exp_grp, proxy,  'CCF_change_perK.nc']) )
                print(nc_fn)
                out_ds = {}
                out_dims = CCF_chg_perK_arr[0].dims
                coords = {}
                for dim in out_dims:
                    coords[dim] = CCF_chg_perK_arr[0][dim]
                for ii, var_nm in enumerate(var_names):
                    out_ds[var_nm] = (out_dims, CCF_chg_perK_arr[ii])
                out_ds = xr.Dataset(out_ds, coords=coords)
                out_ds.to_netcdf(nc_fn, mode='w', format='NETCDF3_CLASSIC')

                # for var in CCF_chg_perK_arr:
                #     print(get_gm(var))
                ind_low_ocn = ind_low_ocn_ctrl & ind_low_ocn_perturb
                CCF_chg_tbl[kk,:] = np.array([reg_mean_dt_from_ind(v, ind_low_ocn)
                                                for v in CCF_chg_perK_arr])
                print(CCF_chg_tbl[kk,:])

            # ==================== Calculate the Var change per K (similar to Feedback) =====================#
            print('Calculate the Var change per K (similar to Feedback)')
            tbl_var_chg_arr = []
            for jj, CCF_chg_perK in enumerate(CCF_chg_perK_arr):
                tbl_var_chg = out_coeff.isel(ind=jj) * CCF_chg_perK
                tbl_var_chg_arr.append(tbl_var_chg)

            ind_low_ocn = ind_low_ocn_ctrl & ind_low_ocn_perturb
            var_chg_tbl[kk,:] = np.array([reg_mean_dt_from_ind(v, ind_low_ocn)
                                                for v in tbl_var_chg_arr])
            print(var_chg_tbl[kk,:])
            delta_tbl_var = (ds_arr[1][tbl_var_nm] - get_final_nyr_mean(ds_arr[0][tbl_var_nm])) / diff_ts_gm
            if 'cwp' in tbl_var_nm:
                delta_tbl_var = delta_tbl_var * 1e3
            var_chg_tbl_real[kk] = reg_mean_dt_from_ind(delta_tbl_var, ind_low_ocn)

            # Save the real change per K to nc file
            nc_fn = P(dt_nc_dir, '_'.join([tbl_var_nm, exp_grp, 'change_perK.nc']) )
            print(nc_fn)
            out_ds = {}
            out_dims = delta_tbl_var.dims
            coords = {}
            for dim in out_dims:
                coords[dim] = delta_tbl_var[dim]
            out_ds[tbl_var_nm] = (out_dims, delta_tbl_var )
            out_ds = xr.Dataset(out_ds, coords=coords)
            out_ds.to_netcdf(nc_fn, mode='w', format='NETCDF3_CLASSIC')

        file_name = P(dt_dir, tbl_var_nm+'_CCF_ctrl_'+proxy+'.csv')
        pdtbl = pd.DataFrame(data=tbl_ctrl, index=exp_grps, columns=var_names)
        pdtbl.to_csv(file_name, header=True, index=True, float_format="%.8f")
        print(file_name, 'saved.')

        file_name = P(dt_dir, tbl_var_nm+'_CCF_perturb_'+proxy+'.csv')
        pdtbl = pd.DataFrame(data=tbl_perturb, index=exp_grps, columns=var_names)
        pdtbl.to_csv(file_name, header=True, index=True, float_format="%.8f")
        print(file_name, 'saved.')

        if not CCF_change_calculated:
            CCF_change_calculated = True
            file_name = P(dt_dir, 'CCF_change_perK_'+proxy+'.csv')
            pdtbl = pd.DataFrame(data=CCF_chg_tbl, index=exp_grps, columns=var_names)
            pdtbl.to_csv(file_name, header=True, index=True, float_format="%.8f")
            print(file_name, 'saved.')

        file_name = P(dt_dir, tbl_var_nm+'_chg_per_K_'+proxy+'.csv')
        pdtbl = pd.DataFrame(data=var_chg_tbl, index=exp_grps, columns=var_names)
        pdtbl.to_csv(file_name, header=True, index=True, float_format="%.8f")
        print(file_name, 'saved.')

        file_name = P(dt_dir, tbl_var_nm+'_chg_per_K_real_'+proxy+'.csv')
        pdtbl = pd.DataFrame(data=var_chg_tbl_real, index=exp_grps, columns=[tbl_var_nm])
        pdtbl.to_csv(file_name, header=True, index=True, float_format="%.8f")
        print(file_name, 'saved.')

    print('Done')
