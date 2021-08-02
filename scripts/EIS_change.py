import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import get_unique_line_labels, add_datetime_info
#import seaborn as sns
from sklearn.linear_model import LinearRegression
from isca_cre_cwp import get_gm
import cmaps

def regional_mean(dt):
    return np.average(dt.mean('lon'), axis=dt.dims.index('lat'), weights=np.cos(np.deg2rad(dt.lat)))

def seasonal_regional_mean(dt):
    add_datetime_info(dt)
    dt_season = dt.groupby(dt.season).mean('time')
    dt_season_area_mean = []
    for i in range(len(dt_season.season)):
        dt_season_area_mean.append(regional_mean(dt_season[i,...]))
    return np.array(dt_season_area_mean)

def get_final_nyr_mean(dt, n=5):
    return dt[-12*n:, :, :].mean('time')

def get_ensemble_mean_for_singl_var(ds_arr1, ds_arr2, varnm, norm_per_K=True):
    dt_sum = 0
    if 'temp700' in varnm:
        dims = ds_arr1[0].temp_2m.dims
        for ds1, ds2 in zip(ds_arr1, ds_arr2):
            ds1['temp700'] = (dims, ds1.temp.sel(pfull=700))
            ds2['temp700'] = (dims, ds2.temp.sel(pfull=700))

    for ds1, ds2 in zip(ds_arr1, ds_arr2):
        if norm_per_K:
            var_tm = (ds2[varnm] - get_final_nyr_mean(ds1[varnm])).mean('time') /\
                    get_gm(ds2.temp_2m - get_final_nyr_mean(ds1.temp_2m))
        else:
            var_tm = (ds2[varnm] - get_final_nyr_mean(ds1[varnm])).mean('time')
        dt_sum = dt_sum + var_tm
    var_ens_mean = dt_sum / len(ds_arr1)
    return var_ens_mean

def get_ctrl_perturb_ds_arr(exp_grps, file_nms, ppe_dir):
    ds_arr1 = []
    ds_arr2 = []

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

        ds_arr1.append(ds_arr[0])
        ds_arr2.append(ds_arr[1])
        #diff_arr.append(ds_arr[1] - ds_arr[0])
    return ds_arr1, ds_arr2

if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs/check_EIS_change'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dt_dir = '../data/'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)
    lsm_dir = '../data/'

    # Read data
    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    #exps_arr = list(exp_tbl.iloc[:, 1])

    regions_dict = {'Peru':           {'lat':(-20.0, -10.0), 'lon': (360.0-90.0, 360.0-80.0)},
                    'Namibia':        {'lat':(-20.0, -10.0), 'lon': (0.0, 10.0)}, 
                    'California':     {'lat':(20.0, 30.0),   'lon': (360.0-130.0, 360.0-120.0)},
                    'Australia':      {'lat':(-35.0, -25.0), 'lon': (95.0, 105.0)},
                    'Canary':         {'lat':(15.0, 25.0),   'lon': (360.0-45.0, 360.0-35.0)},
                    #'North Pacific':  {'lat':(40.0, 50.0),   'lon': (170.0, 180.0)},
                    #'North Atlantic': {'lat':(50.0, 60.0),   'lon': (360.0-45.0, 360.0-35.0)},
                    #'China':          {'lat':(20.0, 30.0),   'lon': (105.0, 120)},
                    }
    eis_arr =[]
    temp0_arr = []
    temp700_arr = []

    for exp_grp in exp_grps:
        #exp_grp = exp_grps[0]
        print(exp_grp)
        fn = P(dt_dir, 'low_cld_proxy_and_temp_changes_' + exp_grp + '.nc')
        ds = xr.open_dataset(fn, decode_times=False)

        #print('Calculate regional mean...')
        for region, range_dict in regions_dict.items():
            l_lat = np.logical_and(ds.lat >= range_dict['lat'][0], ds.lat <= range_dict['lat'][1])
            l_lon = np.logical_and(ds.lon >= range_dict['lon'][0], ds.lon <= range_dict['lon'][1])
            var_names = ['lts', 'eis', 'ELF', 'temp_2m', 'temp700']
            for varnm in var_names:
                regions_dict[region][varnm] = ds[varnm].where(l_lat & l_lon, drop=True)

            eis = seasonal_regional_mean(range_dict['eis'])
            temp0 = seasonal_regional_mean(range_dict['temp_2m'])
            temp700 = seasonal_regional_mean(range_dict['temp700'])

            eis_arr.extend(eis)
            temp0_arr.extend(temp0)
            temp700_arr.extend(temp700)

    # Read Isca data
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')
    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']
    ds_arr1_qflux, ds_arr2_qflux = get_ctrl_perturb_ds_arr(exp_grps, file_nms, ppe_dir)

    X = np.transpose(np.array([temp700_arr, temp0_arr]))
    y = np.array(eis_arr)
    reg = LinearRegression().fit(X, y)
    print(reg.coef_, reg.intercept_) # [ 0.74885833 -1.06505558] # [ 0.77151483 -1.01402952]

    R = np.abs(reg.coef_[1] / reg.coef_[0])
    print('Ratio is', R)

    region_shapes = ['.', '^', '+', 's', 'x', 'v', 'p', '*']

    # ===================== PLOT ========================= #
    plot.rc['cycle'] = 'bmh'     #'default' #'ggplot'
    plot.close()
    # fig, axes = plot.subplots(nrows=3, ncols=1, aspect=(2.4, 1), 
    #         axwidth=3, sharex=True, sharey=False)

    # fig, axes_all = plot.subplots([[1,2,3], [1,2,4], [5,6,7], [5,6,8]], axwidth=5, sharex=True, # ,
    #                 proj={1:'kav7', 2:'kav7', 3:None, 4:None, 5:'kav7', 6:None, 7:None, 8:None},
    #                 proj_kw={'lon_0':180})

    # fig, axes_all = plot.subplots([[1,2,3], [4,5,5], [6,5,5], [7,5,5]],
    #                 hratios=[1, 0.7, 0.7, 0.7], axwidth=3, share=0, # ,
    #                 proj={1:'kav7', 2:'kav7', 3:'kav7', 4:None, 5:None, 6:None, 7:None},
    #                 proj_kw={'lon_0':180})
    fig, axes_all = plot.subplots([[1,2], [3,4], [5,6], [5,7]],
                    hratios=[0.4, 0.4, 0.3, 0.3],
                    axwidth=2.5, share=0,
                    proj={1:'kav7', 2:'kav7', 3:'kav7', 4:None, 5:None, 6:None, 7:None},
                    proj_kw={'lon_0':180})

    ##### ================= Plot maps ============= #####
    norm_per_K = True
    delta_T0 = get_ensemble_mean_for_singl_var(ds_arr1_qflux, ds_arr2_qflux, 'temp_2m', norm_per_K=norm_per_K)
    delta_T700 = get_ensemble_mean_for_singl_var(ds_arr1_qflux, ds_arr2_qflux, 'temp700', norm_per_K=norm_per_K)
    delta_EIS = get_ensemble_mean_for_singl_var(ds_arr1_qflux, ds_arr2_qflux, 'eis', norm_per_K=norm_per_K)

    map_dt_arr = [delta_T0, delta_T700, delta_EIS]
    titles = [r'$\Delta T_0$', r'$\Delta T_{700}$', r'$\Delta$EIS']
    cnlevels_arr = [np.arange(0, 2.1, 0.2)] * 2 + [np.arange(-1, 1.1, 0.1)]

    axes_all[0:3].format(coast=True, latlines=30, lonlines=60)
    for jj, (ax, dt, title) in enumerate(zip(axes_all[0:3], map_dt_arr, titles)):
        print(' ', jj)
        if jj < 2:
            cmap = cmaps.MPL_YlOrBr #'Oranges'
            extend = 'max'
        else:
            cmap = 'RdBu_r'
            extend = 'both'
        cs = ax.contourf(dt.lon, dt.lat, dt, cmap=cmap, extend=extend, levels=cnlevels_arr[jj])
        ax.set_title(title)
        cbar = ax.colorbar(cs, loc='r', shrink=0.8, width='1em') # label='%'
        cbar.ax.set_title(r'K K$^{-1}$')

    # ================== Plot ratio ================= #
    axes = axes_all[3:4] + axes_all[5:]
    lines = []
    for kk, exp_grp in enumerate(exp_grps):
        #exp_grp = exp_grps[0]
        print(exp_grp)
        fn = P(dt_dir, 'low_cld_proxy_and_temp_changes_' + exp_grp + '.nc')
        ds = xr.open_dataset(fn, decode_times=False)

        eis_arr =[]
        temp0_arr = []
        temp700_arr = []
        region_arr = []

        #print('Calculate regional mean...')
        for region, range_dict in regions_dict.items():
            l_lat = np.logical_and(ds.lat >= range_dict['lat'][0], ds.lat <= range_dict['lat'][1])
            l_lon = np.logical_and(ds.lon >= range_dict['lon'][0], ds.lon <= range_dict['lon'][1])
            var_names = ['lts', 'eis', 'ELF', 'temp_2m', 'temp700']
            for varnm in var_names:
                regions_dict[region][varnm] = ds[varnm].where(l_lat & l_lon, drop=True)

            eis = np.mean(regional_mean(range_dict['eis'])) #.mean('time')
            temp0 = np.mean(regional_mean(range_dict['temp_2m'])) #.mean('time')
            temp700 = np.mean(regional_mean(range_dict['temp700'])) #.mean('time')
            
            region_arr.append(region)
            eis_arr.append(eis)
            temp0_arr.append(temp0)
            temp700_arr.append(temp700)
    
        for i, (eis, temp0, temp700, region) in enumerate(zip(eis_arr, temp0_arr, temp700_arr, region_arr)):
            l = axes[0].plot(kk, eis, region_shapes[i], color='C'+str(i), label=region)
            axes[1].plot(kk, np.mean(temp700) / np.mean(temp0), region_shapes[i], color='C'+str(i), label=region)
            axes[2].plot(kk, temp0, region_shapes[i], color='C'+str(i), label=region)
            lines.extend(l)
    axes[1].set_ylim(ymin=0)
    xlim = [-1, len(exp_grps)]
    xticks = np.arange(0, len(exp_grps), 1)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(xx+1)) for xx in xticks])
    axes[2].set_xlabel('Isca simulations')

    axes[1].plot(xlim, [R, R], 'k--')
    axes[0].plot(xlim, [0, 0], 'k--')

    axes[0].set_ylabel(r'${\Delta}$EIS (K)')
    axes[1].set_ylabel(r'${\Delta T_{700}}/{\Delta T_0}$')
    axes[2].set_ylabel(r'${\Delta T_0}$ (K)')

    #ax.set_ylim(ymin=0)
    #new_lines, new_labels = get_unique_line_labels(lines)
    #axes[-1].legend(new_lines, new_labels, loc='b', ncol=3)

    # ========================== Plot regression ================ #
    ax = axes_all[4]

    lines = []
    for i, (marker, (region, local_dt)) in enumerate(zip(region_shapes, regions_dict.items())):
        eis = seasonal_regional_mean(local_dt['eis'])
        temp0 = seasonal_regional_mean(local_dt['temp_2m'])
        temp700 = seasonal_regional_mean(local_dt['temp700'])
        estimated_eis = reg.coef_[0] * temp700 + reg.coef_[1] * temp0
        #print(eis, estimated_eis)
        line = ax.scatter(eis, estimated_eis, marker=marker, s=30, label=region)
        lines.append(line)
    ax.set_xlabel(r'%.2f$\Delta T_{700}%.2f\Delta T_0$ (K)'%(reg.coef_[0], reg.coef_[1]))
    ax.set_ylabel(r'$\Delta$EIS (K)')
    xlim = [-6, 6]
    ax.plot(xlim, xlim, 'k--')
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    new_lines, new_labels = get_unique_line_labels(lines)
    ax.legend(new_lines, new_labels, loc='lr', ncol=1)

    axes_all[3:].format(grid=True, xtickminor=False, ytickminor=False)
    axes_all.format(abc=True, abcstyle='(a)', abcloc='ul')

    fig_name = P(fig_dir, 'EIS_change_pattern_and_estimation.png')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=200)
    #plot.show()
    print(fig_name, 'saved')
