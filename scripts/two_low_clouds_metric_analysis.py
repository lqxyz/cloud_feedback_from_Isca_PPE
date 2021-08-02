#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning)
import proplot as plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis_functions import (get_ds_arr_from_exps, get_unique_line_labels, add_datetime_info)
from isca_cre_cwp import calc_toa_cre_for_isca, calc_total_cwp_for_isca, add_toa_net_flux_to_ds_arr
from estimated_inv_strength import estimated_inversion_strength

def select_trade_wind_Cu_and_Sc(ds, slat=-35, nlat=35, ocn=True, 
            land_mask_dir='./data', T0=18.5, method='LTS'):
    """
    Medeiros, B., & Stevens, B. (2011). Climate dynamics, 36(1-2), 385-399.
    Revealing differences in GCM representations of low clouds. 
    doi: 10.1007/s00382-009-0694-5

    Criteria:
    1. w500 >= 10 hPa/day & w700 >= 10 hPa/day
    2. Trade wind cumulus: LTS < 18.5; 
       Stratocumulus: LTS >= 18.5
    """
    coeff = 3600. * 24. / 100.
    try:
        w500 = ds.omega.sel(pfull=500) * coeff
        w700 = ds.omega.sel(pfull=700) * coeff
    except:
        w500 = ds.omega.interp(pfull=500) * coeff
        w700 = ds.omega.interp(pfull=700) * coeff
    lts = ds.lts
    eis = ds.eis
    lats = ds.lat

    if 'LTS' in method.upper():
        w0 = 10    # hPa/day
        #T0 = 18.5  # K
        ind_low = (w500 >= w0) & (w700 >= w0) & (lats >= slat) & (lats <= nlat)
        ind_Cu = ind_low & (lts < T0)
        ind_Sc = ind_low & (lts >= T0)

    if 'EIS' in method.upper():
        w0 = 15
        #T0 = 4
        ind_low = (w700 >= w0) & (lats >= slat) & (lats <= nlat)
        ind_Cu = ind_low & (eis < T0)
        ind_Sc = ind_low & (eis >= T0)
        
    if ocn:
        # get land_mask
        ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
        ds.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

        ind_Cu = ind_Cu & (ds.mask == 0)
        ind_Sc = ind_Sc & (ds.mask == 0)

    return ind_Cu, ind_Sc

def get_profiles_from_ind(dt, ind):
    dt_ind = dt.where(ind, drop=True)
    dt_prof = dt_ind.mean(('time', 'lat', 'lon'), skipna=True)
    return dt_prof

def plot_w500_and_lts_threshold(ds_arr, exp_names, figname, land_mask_dir='./data'):
    N = len(ds_arr)
    lons = ds_arr[0].lon
    lats = ds_arr[0].lat
    coeff = 3600. * 24. / 100.

    w500_arr = []
    lts_arr = []

    for ds in ds_arr:
        ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
        ds.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)
        try:
            w500_tm = ds.omega.sel(pfull=500).mean('time') * coeff
        except:
            w500_tm = ds.omega.interp(pfull=500).mean('time') * coeff
        lts_tm = ds.lts.where(ds.mask==0).mean('time')
        w500_arr.append(w500_tm)
        lts_arr.append(lts_tm)

    plot.close()
    proj = plot.Proj('cyl', lon_0=180) 
    #proj = plot.Proj('cyl', lon_0=181, basemap=True)
    fig, axes = plot.subplots(nrows=N, ncols=1, axwidth=5, proj=proj) # 

    cnlevels = np.arange(-80, 85, 5) 
    levels = [18.5]
    for ax, w500, lts, exp_nm in zip(axes, w500_arr, lts_arr, exp_names):
        # plot the annual mean w500
        cs = ax.contourf(lons, lats, w500, levels=cnlevels, cmap='rdbu_r')
        #ax.contour(lons, lats, w500, levels=[10], color='g')
        # add contour of lts
        cs_label = ax.contour(lons, lats, lts, 
                levels=levels, colors='k', linewidths=1)
        ax.clabel(cs_label, cs_label.levels, inline=True) 
        cs_label = ax.contour(lons, lats, lts, 
                levels=np.arange(4,30,2), colors='gray', linewidths=1) 
        # Recast levels to new class 
        fmt = {}  
        for l in cs_label.levels:
            #print(l)
            fmt[l] = str(int(l)) #str(nf(l))
        ax.clabel(cs_label, cs_label.levels[::2], inline=True, fmt=fmt)
        ax.set_title(exp_nm)

    fig.colorbar(cs, loc='b', label='$\omega_{500}$ (hPa day$^{-1}$)')
    axes.format(latlim=(-35,35), land=True, landcolor='gray', labels=True, 
                lonlines=60, latlines=30, gridminor=False,
                 abc=True, abcstyle='(a)')
    #fig.tight_layout()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

def plot_w700_and_eis_threshold(ds_arr, exp_names, figname, land_mask_dir='./data'):
    N = len(ds_arr)
    lons = ds_arr[0].lon
    lats = ds_arr[0].lat
    coeff = 3600. * 24. / 100.

    w_arr = []
    eis_arr = []

    p_thereshold = 700
    for ds in ds_arr:
        ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
        ds.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)
        try:
            w_tm = ds.omega.sel(pfull=p_thereshold).mean('time') * coeff
        except:
            w_tm = ds.omega.interp(pfull=p_thereshold).mean('time') * coeff
        eis_tm = ds.eis.where(ds.mask==0).mean('time')
        w_arr.append(w_tm)
        eis_arr.append(eis_tm)

    plot.close()
    proj = plot.Proj('cyl', lon_0=180) 
    #proj = plot.Proj('cyl', lon_0=181, basemap=True)
    fig, axes = plot.subplots(nrows=N, ncols=1, axwidth=5, proj=proj) # 

    cnlevels = np.arange(-80, 85, 5) 
    levels = [18.5]
    for ax, w_val, eis, exp_nm in zip(axes, w_arr, eis_arr, exp_names):
        # plot the annual mean w_val
        cs = ax.contourf(lons, lats, w_val, levels=cnlevels, cmap='rdbu_r')
        #ax.contour(lons, lats, w_val, levels=[10], color='g')
        # add contour of eis
        cs_label = ax.contour(lons, lats, eis, 
                levels=levels, colors='k', linewidths=1)
        ax.clabel(cs_label, cs_label.levels, inline=True) 
        cs_label = ax.contour(lons, lats, eis, 
                levels=np.arange(0,9,1), colors='gray', linewidths=1) 
        # Recast levels to new class 
        fmt = {}  
        for l in cs_label.levels:
            #print(l)
            fmt[l] = str(int(l)) #str(nf(l))
        ax.clabel(cs_label, cs_label.levels[::2], inline=True, fmt=fmt)
        ax.set_title(exp_nm)

    fig.colorbar(cs, loc='b', label='$\omega_{700}$ (hPa day$^{-1}$)')
    axes.format(latlim=(-35,35), land=True, landcolor='gray', labels=True, 
                lonlines=60, latlines=30, gridminor=False,
                 abc=True, abcstyle='(a)')
    #fig.tight_layout()
    fig.savefig(figname, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()


if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    lsm_dir = '../inputs'

    # ds_arr, exp_names = read_isca_data(n_exp=12)
    # #ds = ds_arr[0]
    # #var_names = ['rh', 'theta', 'cf', 'qcl_rad']

    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')

    #exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0]) 
    # exps_arr = list(exp_tbl.iloc[:, 1])
    markers = list(exp_tbl.iloc[:, 2]) 

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']

    dst_ds_arr = []
    dst_ds_arr_perturb = []
    dst_ds_arr_list = []
    for exp_grp in exp_grps:
        #exp_names = [exp_grp + '_CTRL', exp_grp+'_4xCO2']
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        for file_nm in file_nms:
            fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

        ds_arr[1]['time'] = ds_arr[0].time

        print('Read data finished.')
        for ds in ds_arr:
            add_datetime_info(ds)
        # print('Calculate TOA CRE for dataset...')
        # calc_toa_cre_for_isca(ds_arr)
        # print('Calculate cloud water path for dataset...')
        # calc_total_cwp_for_isca(ds_arr)
        # print('Add Net TOA flux to dataset...')
        # add_toa_net_flux_to_ds_arr(ds_arr)
        
        for ds in ds_arr:
            if not 'lts' in ds.variables:
                print('Calc lts and eis...')
                estimated_inversion_strength(ds)
        dst_ds_arr.append(ds_arr[0])
        dst_ds_arr_perturb.append(ds_arr[1])
        
        dst_ds_arr_list.append(ds_arr)

    plot.rc['cycle'] = 'bmh'
    colors = ['k', 'b'] + ['C'+str(i) for i in range(0,11)]

    # For EIS
    print('EIS')
    #dst_ds_list = dst_ds_arr
    #for ii, dst_ds_list in enumerate([dst_ds_arr, dst_ds_arr_perturb]):
    for T0 in np.arange(4, 4.5, 0.5):
        print('T0', T0)
        var_cu_profs = {}
        var_sc_profs = {}
        ind_Cu_dict = {}
        ind_Sc_dict = {}
        for ds_arr, exp_grp in zip(dst_ds_arr_list, exp_grps):
            exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']
            for ds, exp_nm in zip(ds_arr, exp_names):
                print(' ', exp_nm)
                ind_Cu, ind_Sc = select_trade_wind_Cu_and_Sc(ds, slat=-35, nlat=35, ocn=True, T0=T0, method='EIS', land_mask_dir=lsm_dir)
                ind_Cu_dict[exp_nm] = ind_Cu
                ind_Sc_dict[exp_nm] = ind_Sc

                var_cu_profs[exp_nm] = {}
                var_sc_profs[exp_nm] = {}

                var_names = ['rh', 'temp', 'cf', 'qcl_rad']
                for varnm in var_names:
                    print('      get_profile for', varnm)
                    var_cu_profs[exp_nm][varnm] = get_profiles_from_ind(ds[varnm], ind_Cu)
                    var_sc_profs[exp_nm][varnm] = get_profiles_from_ind(ds[varnm], ind_Sc)

        print('Begin plot')
        ylim_arr = [[580, 950], [50, 950]]
        #ylim = [580, 950]
        for ylim in ylim_arr:
            print(ylim)
            if ylim[0]<100:
                fig_name = P(fig_dir, 'regime_and_profiles_pfull_EIS_'+str(T0)+'.png')
            else:
                fig_name = P(fig_dir, 'regime_and_profiles_EIS_'+str(T0)+'.png')
 
            plot.close()
            # fig, axes = plot.subplots([[1,1,1], [2,3,4], [5,6,7]], share=0, axwidth=5, # hratios=[1.2, 0.6, 0.6],
            #         proj={1:'cyl', 2:None, 3:None, 4:None, 5:None, 6:None, 7:None}, proj_kw={'lon_0':180})
            # #fig, axes = plot.subplots(nrows=3, ncols=3, proj={'cyl'}, proj_kw={'lon_0':180}) 
            fig, axes = plot.subplots([[1,1,1,1], [2,3,4,5], [6,7,8,9]], hratios=[1, 1.5, 1.5], sharex=False, axwidth=5, # ,
                    proj={1:'cyl', 2:None, 3:None, 4:None, 5:None, 6:None, 7:None, 8:None, 9:None},
                    proj_kw={'lon_0':180})
        
            # =========== Plot the map for Sc and Cu partition =========== #
            print('map')
            ax = axes[0]
            ax.format(latlim=(-35, 35), land=True, landcolor='gray', labels=True, 
                    lonlines=60, latlines=30, gridminor=False)

            exp_nm = exp_grps[0] + '_ctrl'
            ind_Cu = ind_Cu_dict[exp_nm]
            ind_Sc = ind_Sc_dict[exp_nm]
            lats = ind_Cu.lat
            lons = ind_Cu.lon
            freq_Cu = ind_Cu.mean('time') * 1e2
            freq_Sc = ind_Sc.mean('time') * 1e2
            ind_both_freq_gt_zero = (freq_Cu > 0) & (freq_Sc > 0)
            freq = -freq_Cu
            freq = xr.where(freq_Sc < freq_Cu, freq, freq_Sc)

            cnlevels = np.arange(-100, 110, 10)    
            cs = ax.contourf(lons, lats, freq, levels=cnlevels, cmap='rdbu_r')
            # The levels and alpha are key parameters..
            # https://stackoverflow.com/questions/55133513/
            # matplotlib-contour-hatching-not-working-if-only-two-levels-was-used
            ax.contourf(lons, lats, ind_both_freq_gt_zero, levels=2, hatches=[None, '..'], alpha=0.05)
            #ax.set_title(exp_nm)
            cbar = ax.colorbar(cs, loc='b', width='1em', shrink=0.9,
                    label='[Trade-wind cumulus]         Frequency of occurrence (%)         [Stratocumulus]') #for (blue) trade-wind cumulus and (red) stratocumulus
            tick_labels = [t.get_text().replace('−', '') for t in cbar.ax.get_xticklabels()]
            #print(tick_labels)
            cbar.ax.set_xticklabels(tick_labels)
            # https://stackoverflow.com/questions/27094747/
            # matplotlib-plots-lose-transparency-when-saving-as-pdf
            ax.set_rasterized(True)

            # Plot profiles for different regimes
            print('profile')
            # # var_names = ['rh', 'theta', 'cf', 'qcl_rad']
            # coeffs = [1, 1, 1e2, 1e3]
            # xlabels = ['RH (%)', 'Temperature (K)', 'Cloud fraction (%)', r'$r_l$ (g kg$^{-1}$)']
            # xlims = [[0, 100], [275, 310], [0, 30], [0, 0.05]]
            # colors = ['k', 'b'] + ['C'+str(i) for i in range(0,11)]

            # pfulls = ds_arr[0].pfull
            # for ax, varnm, coeff, xlabel, xlim in zip(axes[1:], var_names, coeffs, xlabels, xlims):
            #     for kk, exp_nm in enumerate(exp_grps):
            #         ax.plot(var_cu_profs[exp_nm][varnm] * coeff, pfulls, ls='-',
            #                 color=colors[kk], marker='s') #, clip_on=False)
            #         ax.plot(var_sc_profs[exp_nm][varnm] * coeff, pfulls, ls='-', 
            #                 color=colors[kk], marker='+') #, clip_on=False)
            #     ax.set_ylim([590, 1000])
            #     ax.set_xlim(xlim)
            #     ax.invert_yaxis()
            #     ax.set_xlabel(xlabel)
            # legend_elements = []
            # for k, exp_nm in enumerate(exp_grps):
            #     color = colors[k]
            #     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=exp_nm))
            # markers = ['s', '+']
            # labels = ['Trade-wind cumulus (Cu)', 'Stratocumulus (Sc)']
            # for marker, label in zip(markers, labels):
            #     legend_elements.append(Line2D([0], [0], linestyle='-', marker=marker,
            #             color='k', lw=1, label=label))
            # axes[1:].format(ylabel='Pressure (hPa)', #abc=True, abcstyle='(a)',
            #             xtickminor=False, ytickminor=False, grid=False,
            #             xspineloc='bottom', yspineloc='left')
            # fig.legend(legend_elements, loc='b', ncol=4)

            # # var_names = ['rh', 'theta', 'cf', 'qcl_rad']
            coeffs = [1, 1, 1e2, 1e3]
            xlabels = [r'$\Delta$RH (%)', r'$\Delta$Temperature (K)', 
                    r'$\Delta$Cloud fraction (%)', r'$\Delta$$r_l$ (g kg$^{-1}$)']
            xlims = [[-10, 22], [0, 30], [-8, 10], [-0.015, 0.02]]

            pfulls = ds_arr[0].pfull
            xlims1 = [[-10, 22], [0, 30], [-8, 10], [-0.015, 0.02]]
            xlims2 = [[-10, 22], [0, 30], [-8, 10], [-0.015, 0.02]]
            xlims_arr = [xlims1, xlims2]
            pfulls = ds_arr[0].pfull
            # #ylim_arr = [[580, 950], [50, 950]]
            # ylim = [580, 950]
            xlims = xlims1
            #for xlims, ylim in zip(xlims_arr, ylim_arr):
                # plot profiles
            
            lines = []
            for ax, varnm, coeff, xlabel, xlim in zip(axes[1:5], var_names, coeffs, xlabels, xlims):
                for i, (exp_grp, marker) in enumerate(zip(exp_grps, markers)):
                    exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']
                    diff_dt = var_cu_profs[exp_names[1]][varnm] - var_cu_profs[exp_names[0]][varnm]
                    l = ax.plot(diff_dt * coeff, pfulls, ls='-', color=colors[i],
                                label=exp_grp, marker=marker, ms=4)
                    lines.extend(l)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)
                ax.invert_yaxis()
                ax.set_xlabel('')

            for ax, varnm, coeff, xlabel, xlim in zip(axes[5:], var_names, coeffs, xlabels, xlims):
                for i, (exp_grp, marker) in enumerate(zip(exp_grps, markers)):
                    exp_names = [exp_grp + '_ctrl', exp_grp + '_perturb']
                    diff_dt = var_sc_profs[exp_names[1]][varnm] - var_sc_profs[exp_names[0]][varnm]
                    ax.plot(diff_dt * coeff, pfulls, ls='-', color=colors[i], marker=marker, ms=4)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)
                ax.invert_yaxis()
                ax.set_xlabel(xlabel)
            new_lines, new_labels = get_unique_line_labels(lines)
            fig.legend(new_lines, new_labels, loc='b', ncol=4)

            axes[1:].format(ylabel='Pressure (hPa)', #abc=True, abcstyle='(a)', abcloc='ul',
                        xtickminor=False, ytickminor=False, grid=False,
                        xspineloc='bottom', yspineloc='left')
            
            axes.format(abc=True, abcstyle='(a)', abcloc='ul')
            fig.tight_layout()
            fig.savefig(fig_name, tight=False, pad_inches=0.1, transparent=False)
            #plot.show()

    """
    # For LTS method
    print('LTS')
    for ii, dst_ds_list in enumerate([dst_ds_arr, dst_ds_arr_perturb]):
        for T0 in [18.5, 19, 20, 20.5, 21, 22]:
            print('T0', T0)
            ind_Cu_dict = {}
            ind_Sc_dict = {}
            # var_cu_profs = {}
            # var_sc_profs = {}

            for ds, exp_nm in zip(dst_ds_list, exp_grps):
                ind_Cu, ind_Sc = select_trade_wind_Cu_and_Sc(ds, slat=-35, nlat=35, ocn=True, T0=T0)
                ind_Cu_dict[exp_nm] = ind_Cu
                ind_Sc_dict[exp_nm] = ind_Sc

                # var_cu_profs[exp_nm] = {}
                # var_sc_profs[exp_nm] = {}
                # for varnm in var_names:
                #     var_cu_profs[exp_nm][varnm] = get_profiles_from_ind(ds[varnm], ind_Cu)
                #     var_sc_profs[exp_nm][varnm] = get_profiles_from_ind(ds[varnm], ind_Sc)
            if ii==0:
                exp_label = 'ctrl'
            else:
                exp_label = '4xco2'
            fig_name = P(fig_dir, 'trade_wind_Cu_and_Sc_regimes_ctrl_then_4xCO2_LTS_all_'+exp_label+'_T0_'+str(T0)+'.png')
            # exp_nm = exp_names[0]
            # plot_trade_wind_cu_and_sc_regime_frequency(ind_Cu_dict[exp_nm], ind_Sc_dict[exp_nm], figname=fig_name)

            plot.close()
            fig, axes = plot.subplots(nrows=4, ncols=3, proj='cyl', proj_kw={'lon_0':180}) 
            axes.format(latlim=(-35, 35), land=True, landcolor='gray', labels=True, 
                    lonlines=60, latlines=30, gridminor=False)

            for ax, exp_nm in zip(axes, exp_grps):
                ind_Cu = ind_Cu_dict[exp_nm]
                ind_Sc = ind_Sc_dict[exp_nm]

                lats = ind_Cu.lat
                lons = ind_Cu.lon

                freq_Cu = ind_Cu.mean('time') * 1e2
                freq_Sc = ind_Sc.mean('time') * 1e2

                ind_both_freq_gt_zero = (freq_Cu > 0) & (freq_Sc > 0)

                freq = -freq_Cu
                freq = xr.where(freq_Sc < freq_Cu, freq, freq_Sc)

                cnlevels = np.arange(-100, 110, 10)    
                cs = ax.contourf(lons, lats, freq, levels=cnlevels, cmap='rdbu_r')
                # The levels and alpha are key parameters..
                # https://stackoverflow.com/questions/55133513/matplotlib-contour-hatching-not-working-if-only-two-levels-was-used
                ax.contourf(lons, lats, ind_both_freq_gt_zero, levels=2, hatches=[None, '..'], alpha=0.05)
                ax.set_title(exp_nm)
                '''
                # following https://matplotlib.org/2.0.2/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes
                # we need to set axes_class=plt.Axes, else it attempts to create
                # a GeoAxes as colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('bottom', size='10%', pad=0.2, axes_class=plt.Axes)
                # fig size constantly changes...
                #cax = fig.add_axes([0.2, 0.3, 0.8, 0.1])
                # Note that shrink='0.7' not work for cax
                cb = fig.colorbar(cs, cax=cax, orientation='horizontal')
                ticks = cb.get_ticks()
                labels = [abs(int(v)) for v in ticks]
                cb.set_ticklabels(labels)
                cb.set_label('Frequency of occurrence [%] for (blue) trade-wind cumulus and (red) stratocumulus')
                '''
            # https://stackoverflow.com/questions/45508036/get-tick-values-of-colorbar-in-matplotlib
            cbar = fig.colorbar(cs, loc='b', label='Frequency of occurrence [%] for (blue) trade-wind cumulus and (red) stratocumulus')
            tick_labels = [t.get_text().replace('−', '') for t in cbar.ax.get_xticklabels()]
            #print(tick_labels)
            cbar.ax.set_xticklabels(tick_labels)
            fig.tight_layout()

            # https://stackoverflow.com/questions/27094747/matplotlib-plots-lose-transparency-when-saving-as-pdf
            ax.set_rasterized(True)
            fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
            #plot.show()
    """
