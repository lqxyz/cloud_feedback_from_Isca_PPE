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

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dt_dir = '../data'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)
    
    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    #exp_tbl2 = pd.read_csv('isca_qflux_exps.csv', header=0)
    exp_tbl2 = pd.read_csv('isca_qflux_exps2.csv', header=0)
    exp_grps2 = list(exp_tbl2.iloc[:, 0])
    markers2 = list(exp_tbl2.iloc[:, 2])
    exp_grps2_dst = [x for x in exp_grps2 if x!='Tmin_m20' and x!='qcl_T_0.17']

    # ================= For basic state ===================#
    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir  = P(base_dir, 'qflux_extracted_data_first_20yr')
    ppe_dir2 = P(base_dir, 'qflux_extracted_data_10yr')

    file_nms = ['extracted_flux_data_1_240.nc', 'extracted_flux_data_361_600.nc']
    file_nms2 = ['extracted_data_241_360.nc', 'extracted_data_601_720.nc']
    file_nms22 = ['extracted_ts_data_241_360.nc', 'extracted_ts_data_601_720.nc']

    # ============= Prepare colors and markers ============== #
    plot.rc['cycle'] = 'bmh'
    markers_dict = {}
    for exp_grp, m in zip(exp_grps2, markers2):
        markers_dict[exp_grp] = m
    colors_dict = {}
    colors = ['k', 'b'] + ['C'+str(i) for i in range(0,11)]
    colors2 = ['lightgray', 'silver', 'gray', 'slategray']
    for exp_grp, c in zip(exp_grps, colors):
         colors_dict[exp_grp] = c
    jj = -1
    for exp_grp in exp_grps2:
        if exp_grp not in exp_grps:
            jj += 1
            colors_dict[exp_grp] = colors2[jj]

    lines = []
    plot.close()
    fig, ax = plot.subplots(nrows=1, ncols=1, aspect=(1.5, 1), axwidth=4)

    # ========= For basic state ============= #
    for kk, exp_grp in enumerate(exp_grps2):
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        for file_nm in file_nms:
            fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

        ds_arr2 = []
        if 'qcl_H' in exp_grp:
            for file_nm in file_nms22:
                fn = P(ppe_dir2, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
                ds = xr.open_dataset(fn, decode_times=False)
                ds_arr2.append(ds)
        else:
            for file_nm in file_nms2:
                fn = P(ppe_dir2, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
                ds = xr.open_dataset(fn, decode_times=False)
                ds_arr2.append(ds)

        # ds_arr[1]['time'] = ds_arr[0].time
        for ds in ds_arr:
            add_datetime_info(ds)
        for ds in ds_arr2:
            add_datetime_info(ds)

        tsurf11 = get_global_annual_mean(ds_arr[0].t_surf)
        tsurf12 = get_global_annual_mean(ds_arr2[0].t_surf)
        tsurf21 = get_global_annual_mean(ds_arr[1].t_surf)
        tsurf22 = get_global_annual_mean(ds_arr2[1].t_surf)
    
        marker = markers_dict[exp_grp]
        ms = 4
        l = ax.plot(tsurf11.year, tsurf11, '-'+marker, markersize=ms, color=colors_dict[exp_grp], label=exp_grp)
        ax.plot(tsurf12.year, tsurf12, '-'+marker, markersize=ms, color=colors_dict[exp_grp])
        ax.plot(tsurf21.year, tsurf21, '--'+marker, markersize=ms, color=colors_dict[exp_grp]) #, label=exp_grp)
        ax.plot(tsurf22.year, tsurf22, '--'+marker, markersize=ms, color=colors_dict[exp_grp])
        ## for legend purpose
        #l = ax.plot(tsurf11.year, tsurf11, linestyle='None', marker=marker, 
        #        markersize=ms, color=colors_dict[exp_grp], label=exp_grp)
        lines.extend(l)
        ax.set_xlabel('Year')
        ax.set_ylabel('Surface temperature (K)')

    ax.format(xtickminor=False, ytickminor=False)

    new_lines, new_labels = get_unique_line_labels(lines)
    fig.legend(new_lines, new_labels, loc='b', ncols=4)

    fig_name = P(fig_dir, 'basic_state_tsurf_in_all_PPE_runs.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    print(fig_name, 'saved.')
