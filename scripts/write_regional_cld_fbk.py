import os
import sys
sys.path.append('../../scripts')
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca #, calc_total_cwp_for_isca

def calc_ma_gm(dt):
    coslat = np.cos(np.deg2rad(dt.lat))
    dt_tm_zm = calc_ma_tm_zm(dt)
    dt_tm_zm_ma = np.ma.masked_array(dt_tm_zm, mask=np.isnan(dt_tm_zm))
    dt_tm_gm = np.ma.average(dt_tm_zm_ma, axis=0, weights=coslat)

    return dt_tm_gm

def calc_ma_tm_zm(dt):
    dims = dt.dims
    try:
        time_lon_axis = (dims.index('month'), dims.index('lon'))
    except:
        time_lon_axis = dims.index('lon')

    dt_tm_zm = dt.mean(axis=time_lon_axis, skipna=True)
    dt_tm_zm = xr.DataArray(dt_tm_zm, dims=('lat'), coords={'lat':dt.lat})
    return dt_tm_zm

def get_trop_and_extrop_mean_for_ds(ds, var_name):
    lats = ds.lat
    l_lat_tropical = (lats >= -30) & (lats <= 30)
    l_lat_extropical = ((lats >= -60) & (lats < -30)) | ((lats > 30) & (lats <= 60))
    l_lat_high = ((lats >= -90) & (lats < -60)) | ((lats > 60) & (lats <= 90))

    var_tropical = ds[var_name].where(l_lat_tropical, drop=True)
    var_tropical_mean = calc_ma_gm(var_tropical)

    var_extropical = ds[var_name].where(l_lat_extropical, drop=True)
    var_mid_mean = calc_ma_gm(var_extropical)

    var_high = ds[var_name].where(l_lat_high, drop=True)
    var_high_mean = calc_ma_gm(var_high)

    var_gm = calc_ma_gm(ds[var_name])

    return var_tropical_mean, var_mid_mean, var_high_mean, var_gm

def get_land_sea_regional_mean_for_ds(ds, var_name, ds_mask=None):
    ds = ds.interp(lat=ds_mask.lat, lon=ds_mask.lon)
    lats = ds.lat
    l_lat_tropical = (lats >= -30) & (lats <= 30)
    l_lat_extropical = ((lats >= -60) & (lats < -30)) | ((lats > 30) & (lats <= 60))
    l_lat_high = ((lats >= -90) & (lats < -60)) | ((lats > 60) & (lats <= 90))

    var_tropical = ds[var_name].where(l_lat_tropical, drop=True)
    var_tropical_mean = calc_ma_gm(var_tropical)

    lsm = ds_mask.land_mask.where(l_lat_tropical, drop=True)
    var_tropical_sea_mean = calc_ma_gm(var_tropical.where(lsm == 0))
    var_tropical_land_mean = calc_ma_gm(var_tropical.where(lsm == 1))

    var_extropical = ds[var_name].where(l_lat_extropical, drop=True)
    var_mid_mean = calc_ma_gm(var_extropical)
    
    lsm = ds_mask.land_mask.where(l_lat_extropical, drop=True)
    var_mid_sea_mean = calc_ma_gm(var_extropical.where(lsm == 0))
    var_mid_land_mean = calc_ma_gm(var_extropical.where(lsm == 1))

    var_high = ds[var_name].where(l_lat_high, drop=True)
    var_high_mean = calc_ma_gm(var_high)

    lsm = ds_mask.land_mask.where(l_lat_high, drop=True)
    var_high_sea_mean = calc_ma_gm(var_high.where(lsm == 0))
    var_high_land_mean = calc_ma_gm(var_high.where(lsm == 1))

    var_gm = calc_ma_gm(ds[var_name])
    lsm = ds_mask.land_mask
    var_sea_gm = calc_ma_gm(ds[var_name].where(lsm == 0))
    var_land_gm = calc_ma_gm(ds[var_name].where(lsm == 1))

    return (var_tropical_mean, var_tropical_sea_mean, var_tropical_land_mean,
            var_mid_mean, var_mid_sea_mean,  var_mid_land_mean, 
            var_high_mean, var_high_sea_mean, var_high_land_mean,
            var_gm, var_sea_gm, var_land_gm)

def create_cld_fbk_component_table_land_sea(ds, ds_mask=None):
    sections = ['ALL', 'HI680', 'LO680']
    lw_nms = ['lw_cld_tot', 'lw_cld_amt', 'lw_cld_alt', 'lw_cld_tau', 'lw_cld_err']
    sw_nms = ['sw_cld_tot', 'sw_cld_amt', 'sw_cld_alt', 'sw_cld_tau', 'sw_cld_err']
    net_nms = ['net_cld_tot', 'net_cld_amt', 'net_cld_alt', 'net_cld_tau', 'net_cld_err']

    lw_sw_nms = lw_nms + sw_nms + net_nms
    sections_tbl = np.zeros((len(sections)*12, len(lw_sw_nms)))

    for kk, sec in enumerate(sections):
        tr_vals = []
        tr_sea_vals = []
        tr_land_vals = []
        extr_vals = []
        extr_sea_vals = []
        extr_land_vals = []
        high_vals = []
        high_sea_vals = []
        high_land_vals = []
        gm_vals = []
        gm_sea_vals = []
        gm_land_vals = []

        for varnm in lw_sw_nms:
            var_name = '_'.join([sec, varnm])
            #var_tr_mean, var_extr_mean, var_high_mean, var_gm = get_trop_and_extrop_mean_for_ds(ds, var_name)
            (var_tr_mean, var_tr_sea_mean, var_tr_land_mean,
                var_mid_mean, var_mid_sea_mean,  var_mid_land_mean, 
                var_high_mean, var_high_sea_mean, var_high_land_mean,
                var_gm, var_sea_gm, var_land_gm) = get_land_sea_regional_mean_for_ds(ds, var_name, ds_mask)

            tr_vals.append(var_tr_mean)
            extr_vals.append(var_mid_mean)
            high_vals.append(var_high_mean)
            gm_vals.append(var_gm)

            tr_sea_vals.append(var_tr_sea_mean)
            extr_sea_vals.append(var_mid_sea_mean)
            high_sea_vals.append(var_high_sea_mean)
            gm_sea_vals.append(var_sea_gm)

            tr_land_vals.append(var_tr_land_mean)
            extr_land_vals.append(var_mid_land_mean)
            high_land_vals.append(var_high_land_mean)
            gm_land_vals.append(var_land_gm)

        mean_val_arrs = [tr_vals, tr_sea_vals, tr_land_vals,
                    extr_vals, extr_sea_vals, extr_land_vals,
                    high_vals, high_sea_vals, high_land_vals,
                    gm_vals, gm_sea_vals, gm_land_vals]
        for nn, mean_vals in enumerate(mean_val_arrs):
            sections_tbl[nn*len(sections)+kk, :] = np.array(mean_vals)

    index = []
    for reg in ['tropical', 'tropical_sea', 'tropical_land', 
                'mid',  'mid_sea', 'mid_land',
                'high', 'high_sea', 'high_land',
                'global', 'global_sea', 'global_land']:
        for sec in sections:
            index.append('_'.join([reg, sec]))
    
    sections_tbl = pd.DataFrame(data=sections_tbl, index=index, columns=lw_sw_nms)

    return sections_tbl

def create_cld_fbk_component_table(ds, ds_mask=None):
    sections = ['ALL', 'HI680', 'LO680']
    lw_nms = ['lw_cld_tot', 'lw_cld_amt', 'lw_cld_alt', 'lw_cld_tau', 'lw_cld_err']
    sw_nms = ['sw_cld_tot', 'sw_cld_amt', 'sw_cld_alt', 'sw_cld_tau', 'sw_cld_err']
    net_nms = ['net_cld_tot', 'net_cld_amt', 'net_cld_alt', 'net_cld_tau', 'net_cld_err']

    lw_sw_nms = lw_nms + sw_nms + net_nms
    sections_tbl = np.zeros((len(sections)*4, len(lw_sw_nms)))

    for kk, sec in enumerate(sections):
        tr_vals = []
        extr_vals = []
        high_vals = []
        gm_vals = []
        for varnm in lw_sw_nms:
            var_name = '_'.join([sec, varnm])
            var_tr_mean, var_extr_mean, var_high_mean, var_gm = get_trop_and_extrop_mean_for_ds(ds, var_name)
            tr_vals.append(var_tr_mean)
            extr_vals.append(var_extr_mean)
            high_vals.append(var_high_mean)
            gm_vals.append(var_gm)

        sections_tbl[kk, :] = np.array(tr_vals)
        sections_tbl[len(sections)+kk, :] = np.array(extr_vals)
        sections_tbl[2*len(sections)+kk, :] = np.array(high_vals)
        sections_tbl[3*len(sections)+kk, :] = np.array(gm_vals)

    index = []
    for reg in ['tropical', 'mid_lat', 'high_lat', 'global']:
        for sec in sections:
            index.append('_'.join([reg, sec]))

    sections_tbl = pd.DataFrame(data=sections_tbl, index=index, columns=lw_sw_nms)

    return sections_tbl

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Read kernel data
    dt_dir = '../data'
    exp_tbl = pd.read_csv('isca_qflux_exps.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    ds_mask = xr.open_dataset(P('../inputs', 'era_land_t42.nc'))

    for exp_grp in exp_grps:
        print(exp_grp, ': Read dataset...')

        fn = P(dt_dir, 'cld_fbk_decomp_v2_' + exp_grp +'.nc')
        ds = xr.open_dataset(fn, decode_times=False)

        tbl = create_cld_fbk_component_table(ds)

        file_name = P(dt_dir, 'cld_fbk_decomp_v2_regional_mean_' + exp_grp +'.csv')
        tbl.to_csv(file_name, header=True, index=True, float_format="%.5f")
        print('Annual mean csv file saved: ', file_name)

        tbl = create_cld_fbk_component_table_land_sea(ds, ds_mask)

        file_name = P(dt_dir, 'cld_fbk_decomp_v2_regional_mean_land_sea_' + exp_grp +'.csv')
        tbl.to_csv(file_name, header=True, index=True, float_format="%.5f")
        print('Annual mean csv file saved: ', file_name)
