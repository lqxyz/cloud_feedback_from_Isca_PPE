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
from analysis_functions import get_unique_line_labels, add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca, add_toa_net_flux_to_ds_arr

def get_global_annual_mean(dt):
    dims = dt.dims
    try:
        lat_ind = dims.index('latitude')
        lat_nm = 'latitude'
        lon_nm = 'longitude'
    except:
        lat_ind = dims.index('lat')
        lat_nm = 'lat'
        lon_nm = 'lon'

    coslat = np.cos(np.deg2rad(dt[lat_nm]))
    dt_gm = np.average(dt.mean(lon_nm), axis=lat_ind, weights=coslat)
    dims1 = [d for d in dims if d!=lat_nm and d!=lon_nm]
    coords = [dt[d] for d in dims1]
    dt_gm = xr.DataArray(dt_gm, coords=coords, dims=dims1)
    add_datetime_info(dt_gm)
    dt_gm_annual = dt_gm.groupby('year').mean()

    return dt_gm_annual

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
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dt_dir = '../data'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    #exp_tbl = pd.read_csv('isca_qflux_exps.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')
    ppe_toa_flux_dir = P(base_dir, 'qflux_extracted_data_toa_flux_30yr')

    #for nyr in [20]: #[20, 30]
    for nyr in [30]: #[20, 30]
        if nyr == 30:
            file_nms = ['extracted_data_301_360.nc', 'extracted_toa_flux_data_361_720.nc']
        if nyr == 20:
            file_nms = ['extracted_data_301_360.nc', 'extracted_flux_data_361_600.nc']

        # ds_arr1 = []
        # ds_arr2 = []
        # diff_arr = []
        
        for xlim in [[0,12], [0,20]]:
            plot.close()
            ncols = 3
            nrows = len(exp_grps) // ncols 
            if np.mod(len(exp_grps), ncols) != 0:
                nrows += 1

            fig, axes = plot.subplots(nrows=nrows, ncols=ncols, aspect=(2, 1), space=0)

            #  slope, intercept, r_value, p_value, stderr, intercept_stderr, ERF_2x, ECS
            # ERF_2x = intercept/ 2
            # ECS = ERF_2x / slope
            tbl = np.zeros((len(exp_grps), 8))
            #for ax in axes[len(exp_grps):]:
            #    ax.remove()

            for kk, (ax, exp_grp, exp) in enumerate(zip(axes[0:len(exp_grps)], exp_grps, exps_arr)):
                print(exp_grp, ': Read dataset...')
                ds_arr = []
                for i_fn, file_nm in enumerate(file_nms):
                    #fn = P(base_dir, exp, file_nm)
                    if nyr == 30 and i_fn == 1:
                        fn = P(ppe_toa_flux_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
                    else:
                        fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
                    ds = xr.open_dataset(fn, decode_times=False)
                    ds_arr.append(ds)

                # Keep the time coordinates the same
                # ds_arr[1]['time'] = ds_arr[0].time
                for ds in ds_arr:
                    add_datetime_info(ds)
            
                # ds_arr1.append(ds_arr[0])
                # ds_arr2.append(ds_arr[1])

                # diff_arr.append(ds_arr[1] - ds_arr[0])

                print('Data read finised.')

                # print('Calculate TOA CRE for dataset...')
                # calc_toa_cre_for_isca(ds_arr)
                print('Add Net TOA flux to dataset...')
                add_toa_net_flux_to_ds_arr(ds_arr)
            
                
                def get_final_nyr_mean(dt, n=5):
                    return dt[-12*n:, :, :].mean('time')

                print('Calculate diff flux...')
                tsurf = get_global_annual_mean(ds_arr[1].t_surf - get_final_nyr_mean(ds_arr[0].t_surf))
                #toa_net_sw = get_global_annual_mean(ds_arr[1].soc_toa_sw - ds_arr[0].soc_toa_sw)
                #olr = get_global_annual_mean(ds_arr[1].soc_olr - ds_arr[0].soc_olr)
                toa_net_flux = get_global_annual_mean(ds_arr[1].soc_toa_net_flux -
                                                    get_final_nyr_mean(ds_arr[0].soc_toa_net_flux))
                # upward, so add - to make it downward positive
                #olr_clr = get_global_annual_mean(ds_arr[1].soc_olr_clr - ds_arr[0].soc_olr_clr)
                #toa_sw_cre = get_global_annual_mean(ds_arr[1].toa_sw_cre - ds_arr[0].toa_sw_cre)
                #toa_lw_cre = get_global_annual_mean(ds_arr[1].toa_lw_cre - ds_arr[0].toa_lw_cre)
                #toa_net_cre = get_global_annual_mean(ds_arr[1].toa_net_cre - ds_arr[0].toa_net_cre)

                #delta_cres = [toa_lw_cre, toa_sw_cre]
                #delta_flux_clr = [-olr_clr, toa_net_sw - toa_sw_cre]
                # delta_net_flux = [-olr_clr + toa_net_sw - toa_sw_cre, toa_net_cre, toa_net_flux]

                tsurf_nyr = tsurf[0:nyr]
                #delta_cres_nyr = [v[0:nyr] for v in delta_cres]
                #delta_flux_clr_nyr = [v[0:nyr] for v in delta_flux_clr]
                #delta_net_flux_nyr = [v[0:nyr] for v in delta_net_flux]
                toa_net_flux_nyr = toa_net_flux[0:nyr]

                
                #xlim = [0, 20] #[0, 20]
                ax.plot(tsurf_nyr, toa_net_flux_nyr, ls='None', marker='o', color='k',  markersize=4)
                ax.set_title('('+chr(kk+97)+') '+exp_grp)
                ax.set_xlim(xlim)
                #ax.set_ylim([-0.25, 10])
                ax.set_ylim(top=10)

                result = stats.linregress(tsurf_nyr, toa_net_flux_nyr)
                x2 = np.linspace(min(xlim), max(xlim), 100)
                y2 = result.slope * x2 + result.intercept
                ax.plot(x2, y2, '-', color='C1', linewidth=1)

                # Add fitted info
                fmt = '%.2f'
                if xlim[1] == 20:
                    xloc = 11.5
                else:
                    xloc = 6.5
                yloc = 6
                ERF_str = fmt % (result.intercept / 2) + r' Wm$^{-2}$'
                slope_str = fmt % result.slope + r' Wm$^{-2}$K$^{-1}$'
                ECS_str = fmt % (-result.intercept / 2 / result.slope) + ' K'
                ax.text(xloc, yloc, r'ERF$_{2x}$=' + ERF_str + 
                                    '\n$\lambda$=' + slope_str +
                                    '\nECS=' + ECS_str, color='k')

                # slope, intercept, r_value, p_value, stderr, intercept_stderr
                tbl[kk, 0:6] = [result.slope, result.intercept, result.rvalue, 
                            result.pvalue, result.stderr, result.intercept_stderr]
                tbl[kk, 6] = result.intercept / 2
                tbl[kk, 7] = -result.intercept / 2 / result.slope

            if xlim[1] == 20:
                xticks = [0, 4, 8, 12, 16]
            else:
                xticks = [0, 2, 4, 6, 8, 10]
            axes.format(titleloc='ul', xlabel='Surface temperature anomaly (K)',
                        ylabel=r'Top of atmosphere net radiation flux anomaly (Wm$^{-2}$)',
                        ytickminor=False, xticks=xticks, yticks=[0, 2, 4, 6, 8])
            # Print mean and stddev info
            print('Climate feedback, mean, stddev:', np.mean(tbl[:,0]), np.std(tbl[:,0]))
            print('Effective forcing, mean, stddev:', np.mean(tbl[:,6]), np.std(tbl[:,6]))
            print('ECS, mean, stddev:', np.mean(tbl[:,7]), np.std(tbl[:,7]))
            '''
            Climate feedback, mean, stddev: -0.6996627200373201 0.08329717145623543
            Effective forcing, mean, stddev: 2.8433813722754184 0.20833213446104873
            ECS, mean, stddev: 4.118714382110627 0.5351870699913492
            '''

            if xlim[1] == 20:
                fig_name = P(fig_dir, 'forcing_slope_ECS_for_qflux_PPE_nyr_' + str(nyr) + '_12_2.pdf')
            else:
                fig_name = P(fig_dir, 'forcing_slope_ECS_for_qflux_PPE_nyr_' + str(nyr) + '_12.pdf')
            fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
            #plot.show()

            column_nms = ['slope', 'intercept', 'rvalue', 'pvalue', 
                        'stderr', 'intercept_stderr', 'ERF_2x', 'ECS']
            tbl = pd.DataFrame(data=tbl, index=exp_grps, columns=column_nms)
            file_name = P(dt_dir, 'forcing_slope_ECS_qflux_PPE_nyr_' + str(nyr) + '_12.csv')
            tbl.to_csv(file_name, header=True, index=True, float_format="%.10f")
