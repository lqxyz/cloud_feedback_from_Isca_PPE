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
#(get_ds_arr_from_exps, #sigma_to_pressure_level, #get_unique_line_labels)
from isca_cre_cwp import calc_toa_cre_for_isca, calc_total_cwp_for_isca, get_gm
import matplotlib.pyplot as plt
import proplot as plot
import warnings
warnings.simplefilter(action='ignore')
import cmaps
import string
import copy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from estimated_inv_strength import estimated_inversion_strength

def plot_multiple_latlon_maps(dt_arr, title_arr, nrows=2, ncols=3,
        units_arr=None, units=None, cmap_arr=None, cmap=None,
        cnlevels_arr=None, cnlevels=None, extend_arr=None, extend=None, 
        axwidth=4, title_add_gm=False, fig_name=None):
    """
    Plot multiple lat/lon maps.
    """
    plot.close()
    fig, axes = plot.subplots(nrows=nrows, ncols=ncols, 
            proj='kav7', proj_kw={'lon_0': 180}, axwidth=axwidth)
    axes.format(coast=True, latlines=30, lonlines=60)

    for i in range(len(dt_arr), nrows*ncols):
        #axes[i].axis("off")
        axes[i].remove()

    for kk, (ax, dt, title) in enumerate(zip(axes, dt_arr, title_arr)):
        # Determine the cmap, cnlevels and extend
        if cmap_arr is not None:
            i_cmap = cmap_arr[kk]
        else:
            if cmap is None:
                print('cmap should not be None if cmap_arr is None.')
            else:
                i_cmap = cmap

        if cnlevels_arr is not None:
            i_cnlevels = cnlevels_arr[kk]
        else:
            i_cnlevels = cnlevels
        if extend_arr is not None:
            i_extend = extend_arr[kk]
        else:
            if extend is None:
                print('extend should not be None if extend_arr is None.')
            else:
                i_extend = extend
        if units_arr is not None:
            i_units = units_arr[kk]
        else:
            if units is None:
                print('units should not be None if units_arr is None.')
            else:
                i_units = units

        # Prepare the title string
        prefix = '('+string.ascii_lowercase[kk]+') ' + title
        if title_add_gm:
            coslat = np.cos(np.deg2rad(dt.lat))
            dt_gm = np.average(dt.mean('lon'), axis=0, weights=coslat)
            val_str = ' (%.2f'%(dt_gm) + i_units + ')'
        else:
            val_str = ''
        i_title = prefix + val_str
        # call func to plot single map
        cs = ax.contourf(dt.lon, dt.lat, dt, cmap=i_cmap, 
                levels= i_cnlevels, extend=i_extend)
        ax.set_title(i_title)
    fig.colorbar(cs, loc='b', shrink=0.8, label=i_units, width='1.2em')

    # save and show figure
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)

def get_regional_mean_for_ds(ds, var_name, s_lat=-90, n_lat=90,
        w_lon=0, e_lon=360, land_sea=None, land_mask_dir='./data'):
    lats = ds.lat
    lons = ds.lon
    l_latlon = (lats >= s_lat) & (lats <= n_lat) & (lons >= w_lon) & (lons <= e_lon)

    if land_sea is not None:
        # get land_mask
        lsm_fn = os.path.join(land_mask_dir, 'era_land_t42.nc')
        ds_mask = xr.open_dataset(lsm_fn, decode_times=False)
        ds.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)
        if 'ocean' in land_sea.lower() or 'ocn' in land_sea.lower():
            l_reg = l_latlon & (ds.mask == 0)
        if 'land' in land_sea.lower():
            l_reg = l_latlon & (ds.mask == 0)
    else:
        l_reg = l_latlon

    var = ds[var_name].where(l_reg, drop=True)

    # calculate regional mean
    coslat = np.cos(np.deg2rad(var.lat))
    lat_dim = var.dims.index('lat')
    lon_dim = var.dims.index('lon')

    var_ma = np.ma.MaskedArray(var, mask=np.isnan(var))
    try:
        time_dim = var.dims.index('time')
        var_zm = np.ma.average(var_ma, axis=(time_dim, lon_dim))
        var_profile = np.average(var_zm, axis=lat_dim-1, weights=coslat)
    except:
        var_zm = np.ma.average(var_ma, axis=lon_dim)
        var_profile = np.average(var_zm, axis=lat_dim, weights=coslat)

    var_profile = xr.DataArray(var_profile, coords={'pfull': ds.pfull},  dims=('pfull'))

    return var_profile

def get_final_nyr_mean(dt, n=2):
    return dt[-12*n:, :, :].mean('time')

if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    dt_dir = '../data/'

    print('Read Isca dataset...')
    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])
    markers = list(exp_tbl.iloc[:, 2])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']

    ds_arr1 = []
    ds_arr2 = []
    #diff_arr = []
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
            #if not 'lts' in ds.variables:
            #    print('Calc lts and eis...')
            #    estimated_inversion_strength(ds)
        # print('Calculate CRE')
        # calc_toa_cre_for_isca(ds_arr)
        # print('Calculate CWP')
        # calc_total_cwp_for_isca(ds_arr)
    
        ds_arr1.append(ds_arr[0])
        ds_arr2.append(ds_arr[1])

        #diff_arr.append(ds_arr[1] - ds_arr[0])

    print('Data read finised.')
    '''
    # Plot surface temperature changes
    title_add_gm = True

    varnm_arr = ['temp_2m', 'toa_sw_cre', 'toa_lw_cre', 'toa_net_cre', 
                'low_cld_amt', 'mid_cld_amt', 'high_cld_amt', 'tot_cld_amt',
                'cwp', 'ctpisccp', 'eis', 'lts',
                 'tauisccp', 'albisccp', 'ELF']
    cmap_arr = ['RdBu_r'] * len(varnm_arr)
    #cmap_arr[0] = 'Oranges' #cmaps.MPL_OrRd  #'Oranges'
    units_arr = ['K', r'Wm$^{-2}$', r'Wm$^{-2}$', r'Wm$^{-2}$',
                '%', '%', '%', '%', r'gm$^{-2}$', 'hPa', 'K', 'K', '', '%', '']
    coeff_arr = [1] * len(varnm_arr)
    coeff_arr[8] = 1e3
    coeff_arr[9] = 1e-2
    coeff_arr[13] = 1e2

    extend_arr = ['both'] * len(varnm_arr)
    extend_arr[0] = 'max'

    cnlevels_arr = [np.arange(0,21,1)] + [np.arange(-40, 42, 2)] * 7 +\
                   [np.arange(-120,121,10), np.arange(-200,201,20)] + \
                   [np.arange(-6,6.2,0.5), np.arange(-6,6.2,0.5),
                   np.arange(-10,10.5,0.5), np.arange(-10,11,2), np.arange(-0.3,0.35,0.05)]

    for varnm, cmap, cnlevels, units, coeff, extend in zip(varnm_arr,
            cmap_arr, cnlevels_arr, units_arr, coeff_arr, extend_arr):
    # for varnm, cmap, cnlevels, units, coeff, extend in zip(varnm_arr[-1:],
    #         cmap_arr[-1:], cnlevels_arr[-1:], units_arr[-1:], coeff_arr[-1:], extend_arr[-1:]):
        print(varnm)
        dt_arr = []
        dt_sum = 0
        #for ds in diff_arr:
        for ds1, ds2 in zip(ds_arr1, ds_arr2):
            var_tm = (ds2[varnm] - get_final_nyr_mean(ds1[varnm])).mean('time') * coeff
            dt_arr.append(var_tm)
            dt_sum = dt_sum + var_tm
        dt_arr.append(dt_sum / len(ds_arr1))
        title_arr = copy.deepcopy(exp_grps)
        title_arr.append('Ensemble mean')

        fig_name = P(fig_dir, varnm + '_change_PPE.pdf')
        plot_multiple_latlon_maps(dt_arr, title_arr, nrows=4, ncols=4,
            units=units, cmap=cmap, cnlevels=cnlevels, extend=extend, 
            axwidth=2, title_add_gm=title_add_gm, fig_name=fig_name)

    # Change normalized by surface temperature change
    cnlevels_arr = [np.arange(0,2.6,0.1)] + [np.arange(-5, 5.5, 0.5)] * 7 +\
                   [np.arange(-15,16,3), np.arange(-25,26,5)] + \
                   [np.arange(-1,1.1,0.1), np.arange(-1,1.1,0.1),
                   np.arange(-1,1.1,0.1), np.arange(-1.5,1.6,0.3), np.arange(-0.3,0.35,0.05)]

    units_arr2 = [ u + r'K$^{-1}$' for u in units_arr]
    for varnm, cmap, cnlevels, units, coeff, extend in zip(varnm_arr,
            cmap_arr, cnlevels_arr, units_arr2, coeff_arr, extend_arr):
    # for varnm, cmap, cnlevels, units, coeff, extend in zip(varnm_arr[-1:],
    #         cmap_arr[-1:], cnlevels_arr[-1:], units_arr2[-1:], coeff_arr[-1:], extend_arr[-1:]):
        print(varnm)
        dt_arr = []
        dt_sum = 0
        for ds1, ds2 in zip(ds_arr1, ds_arr2): #diff_arr:
            var_tm = (ds2[varnm]-get_final_nyr_mean(ds1[varnm])).mean('time') /\
                      get_gm(ds2.temp_2m-get_final_nyr_mean(ds1.temp_2m)) * coeff
            # var_tm_per_K = var_tm / get_gm(ds.temp_2m)
            dt_arr.append(var_tm)
            dt_sum = dt_sum + var_tm
        dt_arr.append(dt_sum / len(ds_arr1))
        title_arr = copy.deepcopy(exp_grps)
        title_arr.append('Ensemble mean')

        fig_name = P(fig_dir, varnm + '_change_per_K_PPE.pdf')
        plot_multiple_latlon_maps(dt_arr, title_arr, nrows=4, ncols=4,
            units=units, cmap=cmap, cnlevels=cnlevels, extend=extend, 
            axwidth=2, title_add_gm=title_add_gm, fig_name=fig_name)
    '''

    # Get global and regional mean profiles
    var_names = ['rh', 'temp', 'cf', 'qcl_rad']
    # var_global_profs = np.zeros((2, len(var_names), len(exp_grps)), dtype='float32')
    # var_tropical_profs = np.zeros((2, len(var_names), len(exp_grps)), dtype='float32')
    # regions: gloabl, global_ocn, global_land
    # tropical, tropical_ocn, tropical_land

    pfulls = ds_arr1[0].pfull
    reg_prof_dict_fn = P(dt_dir, "reg_prof_dict.npy")

    if not os.path.exists(reg_prof_dict_fn):
        reg_prof_dict = {}
        '''
        regions_arr = ['global', 'global_ocn', 'global_land', 
                     'tropical', 'tropical_ocn', 'tropical_land',
                     'NH mid-latitude', 'NH mid-latitude_ocn', 'NH mid-latitude_land',
                     'SH mid-latitude', 'SH mid-latitude_ocn', 'SH mid-latitude_land']
        ranges_arr = [(-90, 90, 0, 360)] * 3 +  [(-30, 30, 0, 360)] * 3 +\
                      [(30, 60, 0, 360)] * 3 +  [(-60, -30, 0, 360)] * 3
        lsm_arr = [None, 'ocean', 'land', None, 'ocean', 'land',
                   None, 'ocean', 'land', None, 'ocean', 'land']
        '''
        regions_arr = ['global']
        ranges_arr = [(-90, 90, 0, 360)]
        lsm_arr = [None]

        for reg, ranges, lsm in zip(regions_arr, ranges_arr, lsm_arr):
            print(reg, ranges, lsm)
            var_prof_dt = np.zeros((2, len(var_names), len(exp_grps), len(pfulls)), dtype='float32')
            s_lat = ranges[0]
            n_lat = ranges[1]
            w_lon = ranges[2]
            e_lon = ranges[3]
            for i in range(2):
                if i == 0:
                    ds_arr = ds_arr1
                if i == 1:
                    ds_arr = ds_arr2
                for j, varnm in enumerate(var_names):
                    for k, exp_grp in enumerate(exp_grps):
                        var_prof_dt[i,j,k,:] = get_regional_mean_for_ds(ds_arr[k], varnm,
                                s_lat=s_lat, n_lat=n_lat, w_lon=w_lon, e_lon=e_lon,
                                land_sea=lsm)
            reg_prof_dict[reg] = var_prof_dt

        np.save(reg_prof_dict_fn, reg_prof_dict) 
    else:
        reg_prof_dict = np.load(reg_prof_dict_fn, allow_pickle='TRUE').item()
    
    '''
    for key, reg_prof_dt in reg_prof_dict.items():
        print(key)
        # # var_names = ['rh', 'theta', 'cf', 'qcl_rad']
        coeffs = [1, 1, 1e2, 1e3]
        xlabels = ['RH (%)', 'Temperature (K)', 'Cloud fraction (%)', r'$r_l$ (g kg$^{-1}$)']
        #xlims1 = [[0, 100], [275, 315], [0, 30], [0, 0.06]] # [275, 310], # [190, 315]
        #xlims2 = [[0, 100], [190, 315], [0, 30], [0, 0.06]]
        #xlims_arr = [xlims1, xlims2]
        xlims = [[0, 100], [190, 315], [0, 30], [0, 0.06]]

        ylim = [3, 950] #ylim = [580, 950] # #
        #ylim_arr = [[580, 950], [50, 950]]
        #for xlims, ylim in zip(xlims_arr, ylim_arr):

        # plot profiles
        plot.close()
        fig, axes = plot.subplots(nrows=1, ncols=4, aspect=(0.4, 1), sharex=False, space=0)
        # var_prof_dt = np.zeros((2, len(var_names), len(exp_grps), len(pfulls)), dtype='float32')
        for ii, (ax, varnm, coeff, xlabel, xlim) in enumerate(zip(axes, var_names, coeffs, xlabels, xlims)):
            for jj, (exp_grp, marker) in enumerate(zip(exp_grps, markers)):
                exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']
                for kk, exp_nm in enumerate(exp_names):
                    ax.plot(reg_prof_dt[kk,ii,jj,:] * coeff, pfulls, ls='-',
                            color='C'+str(kk), marker=marker)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)

        legend_elements = []
        for k, exp_nm in enumerate(['CTRL', '4xCO2']):
            color = 'C'+str(k)
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label=exp_nm))
        for exp_grp, marker in zip(exp_grps, markers):
            legend_elements.append(Line2D([0], [0], linestyle='-', marker=marker,
                    color='k', lw=1, label=exp_grp))
        axes[-1].legend(legend_elements, ncol=1, frame=False)

        axes.format(ylabel='Pressure (hPa)', abc=True, abcstyle='(a)',
                    xtickminor=False, ytickminor=False, grid=False,
                    xspineloc='bottom', yspineloc='left')

        if ylim[0] < 100:
            fig_name = P(fig_dir, key+'_mean_profiles_for_PPE_full.pdf')
        else:
            fig_name = P(fig_dir, key+'_mean_profiles_for_PPE_low.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        #plot.show()
    '''

    for key, reg_prof_dt in reg_prof_dict.items():
        print(key)
        # # var_names = ['rh', 'theta', 'cf', 'qcl_rad']
        coeffs = [1, 1, 1e2, 1e3]
        #xlabels = ['RH (%)', 'Temperature (K)', 'Cloud fraction (%)', r'$r_l$ (g kg$^{-1}$)']
        xlabels = ['Relative humidity (%)', 'Temperature (K)', 'Cloud fraction (%)', r'Cloud water (g kg$^{-1}$)']
        xlims = [[0, 100], [190, 315], [0, 50], [0, 0.06]]
        ylim = [3, 950]
        
        # plot profiles
        plot.close()
        fig, axes = plot.subplots(nrows=1, ncols=4, aspect=(0.5, 1), axwidth=1.3, sharex=False, hspace=0)
        # var_prof_dt = np.zeros((2, len(var_names), len(exp_grps), len(pfulls)), dtype='float32')
        for ii, (ax, varnm, coeff, xlabel, xlim) in enumerate(
                            zip(axes, var_names, coeffs, xlabels, xlims)):
            exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']
            exp_labels = ['CTRL', '4xCO2']
            for kk, exp_nm in enumerate(exp_names):
                #    ax.plot(reg_prof_dt[kk,ii,jj,:] * coeff, pfulls, ls='-',
                #            color='C'+str(kk), marker=marker)
                reg_dt_mean = np.mean(reg_prof_dt[kk,ii,:,:], axis=0) 
                ax.plot(reg_dt_mean * coeff, pfulls, ls='-',
                        color='C'+str(kk), label=exp_labels[kk])
                reg_dt_std = np.std(reg_prof_dt[kk,ii,:,:], axis=0)
                x1 = (reg_dt_mean - reg_dt_std) * coeff #np.min(reg_prof_dt[kk,ii,:,:] * coeff, axis=0)
                x2 = (reg_dt_mean + reg_dt_std) * coeff #np.max(reg_prof_dt[kk,ii,:,:] * coeff, axis=0)
                ax.fill_betweenx(pfulls, x1, x2, facecolor='C'+str(kk), alpha=0.5)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)

        # legend_elements = []
        # for k, exp_nm in enumerate(['CTRL', '4xCO2']):
        #     color = 'C'+str(k)
        #     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=exp_nm))
        # for exp_grp, marker in zip(exp_grps, markers):
        #     legend_elements.append(Line2D([0], [0], linestyle='-', marker=marker,
        #             color='k', lw=1, label=exp_grp))
        axes[-1].legend(ncol=1, frame=False)

        axes.format(ylabel='Pressure (hPa)', abc=True, abcstyle='(a)', #abcloc='ul',
                    xtickminor=False, ytickminor=False, grid=False,
                    xspineloc='bottom', yspineloc='left')

        if ylim[0] < 100:
            fig_name = P(fig_dir, key+'_multi_mean_profiles_for_PPE_full.pdf')
        else:
            fig_name = P(fig_dir, key+'_multi_mean_profiles_for_PPE_low.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        #plot.show()
        print(fig_name, 'saved')
