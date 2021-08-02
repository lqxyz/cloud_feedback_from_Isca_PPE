from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
from scipy import stats
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca, add_toa_net_flux_to_ds_arr
from calc_toa_cld_induced_flux_from_zelinka_kernel import get_global_annual_mean

def add_fitted_line_info(ax, result, x=0, y=0, color='k',
                        fmt='%.2f', nsigma=1, prefix=''):
    slope_str = fmt % result.slope + '$\pm$' + fmt % (nsigma * result.stderr)
                #' (' + str(nsigma) + '$\sigma$)'
    intercept_str = fmt % result.intercept + '$\pm$' + fmt % \
                (nsigma * result.intercept_stderr) + ' (' + str(nsigma) + '$\sigma$)'
    ax.text(x, y, prefix + 'slope: ' + slope_str + '; intercept: ' + intercept_str,
            color=color, fontweight='bold') #, fontsize=fontsize)

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dt_dir = '../data'

    base_dir = '../inputs/'
    file_nm_1xco2 = 'extracted_flux_data_30yr_1xco2.nc'
    file_nm_4xco2 = 'extracted_flux_data_30yr_4xco2.nc'

    print('Read dataset...')
    ds_arr = []
    for file_nm in [file_nm_1xco2, file_nm_4xco2]:
        fn = P(base_dir, 'cld_fbk_cmp', file_nm)
        ds = xr.open_dataset(fn, decode_times=False)
        ds_arr.append(ds)
    
    #ds_arr[1]['time'] = ds_arr[0].time

    for ds in ds_arr:
        add_datetime_info(ds)

    print('Calculate TOA CRE for dataset...')
    calc_toa_cre_for_isca(ds_arr)
    print('Add Net TOA flux to dataset...')
    add_toa_net_flux_to_ds_arr(ds_arr)

    print('Calculate diff flux...')
    '''
    diff_ds = ds_arr[1] - ds_arr[0]
    tsurf = get_global_annual_mean(diff_ds.t_surf)
    toa_net_sw = get_global_annual_mean(diff_ds.soc_toa_sw)
    olr = get_global_annual_mean(diff_ds.soc_olr)
    toa_net_flux = get_global_annual_mean(diff_ds.soc_toa_net_flux)

    toa_sw_cre = get_global_annual_mean(diff_ds.toa_sw_cre)
    toa_lw_cre = get_global_annual_mean(diff_ds.toa_lw_cre)
    toa_net_cre = get_global_annual_mean(diff_ds.toa_net_cre)
    '''

    def get_final_yr_mean(dt):
        return dt[-12:, :, :].mean('time')
    
    tsurf = get_global_annual_mean(ds_arr[1].t_surf - get_final_yr_mean(ds_arr[0].t_surf))
    toa_net_sw = get_global_annual_mean(ds_arr[1].soc_toa_sw - get_final_yr_mean(ds_arr[0].soc_toa_sw))
    olr = get_global_annual_mean(ds_arr[1].soc_olr - get_final_yr_mean(ds_arr[0].soc_olr))
    toa_net_flux = get_global_annual_mean(ds_arr[1].soc_toa_net_flux -
                                          get_final_yr_mean(ds_arr[0].soc_toa_net_flux))
    toa_sw_cre = get_global_annual_mean(ds_arr[1].toa_sw_cre - get_final_yr_mean(ds_arr[0].toa_sw_cre))
    toa_lw_cre = get_global_annual_mean(ds_arr[1].toa_lw_cre - get_final_yr_mean(ds_arr[0].toa_lw_cre))
    toa_net_cre = get_global_annual_mean(ds_arr[1].toa_net_cre - get_final_yr_mean(ds_arr[0].toa_net_cre))

    delta_cres = [toa_lw_cre, toa_sw_cre]

    # Read data estimated from kernel methods
    print('Read kernel derived flux data...')
    ds_flux = xr.open_dataset(P(dt_dir, 'toa_cld_flux_v2.nc'), decode_times=False)
    cld_flux_arr = [ds_flux.toa_lw_cld_flux, ds_flux.toa_sw_cld_flux] #,  ds_flux.toa_net_cld_flux]

    '''
    # First 20 years
    nyr = 20
    tsurf = tsurf[0:nyr]
    delta_cres_nyr = [v[0:nyr] for v in delta_cres]
    cld_flux_arr_nyr = [v[0:nyr] for v in cld_flux_arr]

    print('Plot...')
    titles = ['Cloud-induced LW Flux Anomalies', 'Cloud-induced SW Flux Anomalies']
    ylims =[[-5, 5], [-2, 7]]
    xlim = [0, 10]

    plot.close()
    fig, axes = plot.subplots(nrows=2, ncols=1,
                aspect=(2, 1), sharey=False, axwidth=4)
    for ax, delta_cre, cld_flux, title, ylim in zip(axes,
                delta_cres_nyr, cld_flux_arr_nyr, titles, ylims):
        # ========= For delta CRE =========== #
        color = 'gray'
        ax.plot(tsurf, delta_cre, '.', color=color, markersize=8)
        # slope, intercept, r_value, p_value, stderr, intercept_stderr
        result = stats.linregress(tsurf, delta_cre)
        x2 = np.linspace(min(xlim), max(xlim), 100)
        y2 = result.slope * x2 + result.intercept
        ax.plot(x2, y2, '-', color=color, linewidth=1)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(title)
        add_fitted_line_info(ax, result, x=0.1, y=max(ylim)-1.5,
                color=color, nsigma=2, prefix='$\Delta$CRE ')

        # ========= For cld induced flux =========== #
        color = 'k'
        ax.plot(tsurf, cld_flux, '.', color=color, markersize=8)
        result = stats.linregress(tsurf, cld_flux)
        x2 = np.linspace(min(xlim), max(xlim), 100)
        y2 = result.slope * x2 + result.intercept
        ax.plot(x2, y2, '-', color=color, linewidth=1)
        add_fitted_line_info(ax, result, x=0.1, y=max(ylim)-0.8,
                color=color, nsigma=2, prefix='$\Delta$$R_c$    ')

    axes.format(xlabel='$\Delta T_s (K)$', ylabel='Wm$^{-2}$',
                ytickminor=False, abc=True, abcstyle='(a)')

    fig_name = P(fig_dir, 'gregory_plot_cld_fbk_v2.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()
    '''
    # nyr = 20
    for nyr in [20, 30]: 
        tsurf2 = tsurf[0:nyr]
        delta_cres_nyr = [v[0:nyr] for v in delta_cres]
        cld_flux_arr_nyr = [v[0:nyr] for v in cld_flux_arr]
        delta_cres = [toa_lw_cre, toa_sw_cre, toa_net_cre]
        cld_flux_arr = [ds_flux.toa_lw_cld_flux, ds_flux.toa_sw_cld_flux, ds_flux.toa_net_cld_flux]

        delta_cres_nyr = [v[0:nyr] for v in delta_cres]
        cld_flux_arr_nyr = [v[0:nyr] for v in cld_flux_arr]

        # Read data estimated from kernel methods
        print('Plot...')
        titles = ['Cloud-induced LW Flux Anomalies', 'Cloud-induced SW Flux Anomalies',
                 'Cloud-induced Net Flux Anomalies']
        ylims =[[-5, 6], [-2, 9], [-3, 10]]
        xlim = [0, 10]

        gregory_delta_cre = []
        gregory_kernel = []

        plot.close()
        fig, axes = plot.subplots(nrows=3, ncols=1,
                    aspect=(2, 1), sharey=False, axwidth=4)
        for ax, delta_cre, cld_flux, title, ylim in zip(axes,
                    delta_cres_nyr, cld_flux_arr_nyr, titles, ylims):
            # ========= For delta CRE =========== #
            color = 'gray'
            ax.plot(tsurf2, delta_cre, '.', color=color, markersize=8)
            # slope, intercept, r_value, p_value, stderr, intercept_stderr
            result = stats.linregress(tsurf2, delta_cre)
            x2 = np.linspace(min(xlim), max(xlim), 100)
            y2 = result.slope * x2 + result.intercept
            ax.plot(x2, y2, '-', color=color, linewidth=1)
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_title(title)
            add_fitted_line_info(ax, result, x=0.1, y=max(ylim)-1.5,
                    color=color, nsigma=2, prefix='$\Delta$CRE ')

            gregory_delta_cre.append(result.slope)

            # ========= For cld induced flux =========== #
            color = 'k'
            ax.plot(tsurf2, cld_flux, '.', color=color, markersize=8)
            result = stats.linregress(tsurf2, cld_flux)
            x2 = np.linspace(min(xlim), max(xlim), 100)
            y2 = result.slope * x2 + result.intercept
            ax.plot(x2, y2, '-', color=color, linewidth=1)
            add_fitted_line_info(ax, result, x=0.1, y=max(ylim)-0.8,
                    color=color, nsigma=2, prefix='$\Delta$$R_c$    ')
            gregory_kernel.append(result.slope)

        axes.format(xlabel='$\Delta T_s$ (K)', ylabel='Wm$^{-2}$',
                    ytickminor=False, abc=True, abcstyle='(a)')

        fig_name = P(fig_dir, 'gregory_plot_all_cld_fbk_v2_nyr_'+str(nyr)+'.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        #plot.show()
        print(fig_name, 'saved')

        ## Save slope info to tables
        table = np.zeros((2,3))
        table[0,:] = gregory_delta_cre
        table[1,:] = gregory_kernel
        row_names = ['Delta CRE (Slope)', 'Delta R (Slope)']
        col_names = ['LW', 'SW', 'Net']
        tbl = pd.DataFrame(data=table, index=row_names, columns=col_names)
        file_name = P(dt_dir, 'slope_of_delta_CRE_and_R_'+str(nyr)+'yr.csv')
        tbl.to_csv(file_name, header=True, index=True, float_format="%.3f")
        print(file_name, 'saved')

