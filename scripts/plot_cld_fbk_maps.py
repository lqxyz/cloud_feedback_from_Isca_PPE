import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import get_unique_line_labels

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
    # dt_ma = np.ma.masked_array(dt, mask=np.isnan(dt))
    # dt_tm_zm = np.ma.average(dt_ma, axis=time_lon_axis)
    dt_tm_zm = dt.mean(axis=time_lon_axis, skipna=True)
    dt_tm_zm = xr.DataArray(dt_tm_zm, dims=('lat'), coords={'lat':dt.lat})
    return dt_tm_zm

def get_ensemble_mean(ds_arr, varnm):
    ds = ds_arr[0]
    shp = (len(ds_arr), len(ds.lat), len(ds.lon))
    dt_sum = np.ones(shp, dtype='float32') * np.nan

    for i, ds in enumerate(ds_arr):
        # dt_ma = np.ma.masked_array(ds[varnm], mask=np.isnan(ds[varnm]))
        # dt_tm = np.ma.average(dt_ma, axis=0)
        # dt_sum[i,] = dt_tm
        dt_sum[i,] = ds[varnm].mean(axis=0, skipna=True)

    dt_mean = np.nanmean(dt_sum, axis=0)
    dims = ('lat', 'lon')
    coords = {}
    for d in dims:
        coords[d] = ds[varnm][d]
    dt_mean = xr.DataArray(dt_mean, dims=dims, coords=coords)
    
    return dt_mean

def get_ensemble_mean_std(ds_arr, varnm):
    ds = ds_arr[0]
    shp = (len(ds_arr), len(ds.lat), len(ds.lon))
    dt_sum = np.ones(shp, dtype='float32') * np.nan

    for i, ds in enumerate(ds_arr):
        # dt_ma = np.ma.masked_array(ds[varnm], mask=np.isnan(ds[varnm]))
        # dt_tm = np.ma.average(dt_ma, axis=0)
        # dt_sum[i,] = dt_tm
        dt_sum[i,] = ds[varnm].mean(axis=0, skipna=True)

    dt_mean = np.nanmean(dt_sum, axis=0)
    dims = ('lat', 'lon')
    coords = {}
    for d in dims:
        coords[d] = ds[varnm][d]
    dt_mean = xr.DataArray(dt_mean, dims=dims, coords=coords)

    dt_std = np.nanstd(dt_sum, axis=0)
    dims = ('lat', 'lon')
    coords = {}
    for d in dims:
        coords[d] = ds[varnm][d]
    dt_std = xr.DataArray(dt_std, dims=dims, coords=coords)
    
    return dt_mean, dt_std

'''
def plot_zonal_mean_cldfbk_components2(ds_arr, fig_name):
    secs = ['ALL', 'HI680', 'LO680']
    var_nms = ['net_cld_tot', 'net_cld_amt', 'net_cld_alt', 'net_cld_tau']
    N = len(ds_arr)
    nlat = len(ds_arr[0].lat)

    dt_zm_dict = {}
    for sec in secs:
        for var_nm in var_nms:
            var_name = sec + '_' + var_nm
            fbk_arr = np.ones((N, nlat), dtype='float32') * np.nan
            # Record each member data
            for i, ds in enumerate(ds_arr):
                fbk_arr[i,] = calc_ma_tm_zm(ds[var_name])
            dt_zm_dict[var_name] = fbk_arr

    dt_ensemble_mean_dict = {}
    for sec in secs:
        for var_nm in var_nms:
            var_name = sec + '_' + var_nm
            dt_ensemble = get_ensemble_mean(ds_arr, var_name)
            dt_ensemble_mean_dict[var_name] = calc_ma_tm_zm(dt_ensemble)
 
    # Plot annual and zonal mean of cloud feedback components
    plot.close()
    fig, axes = plot.subplots(nrows=3, ncols=1, aspect=(2,1)) #, axwidth=5)
    for ax, sec in zip(axes, secs):
        for i, var_nm in enumerate(var_nms):
            var_name = sec + '_' + var_nm
            dt = dt_ensemble_mean_dict[var_name]
            if 'tot' in var_name.lower():
                color = 'k'
            else:
                color = 'C'+str(i-1)
            ax.plot(dt.lat, dt, color=color)
        ax.set_title(sec)
        ax.set_ylim([-2,1.5])
    axes.format(ylabel='Wm$^{-2}$K$^{-1}$', xlabel='Latitude', 
                xlocator=plot.arange(-90, 90, 30),
                xformatter='deglat', abc=True, abcstyle='(a)')
    #plot.show()
'''

def plot_zonal_mean_cldfbk_components_with_std(ds_arr, fig_name):
    secs = ['ALL', 'HI680', 'LO680']
    sec_title = {'ALL': 'All', 'HI680':'Non-low', 'LO680':'Low'}

    # Plot annual and zonal mean of cloud feedback components
    plot.close()
    fig, axes = plot.subplots(nrows=4, ncols=3, aspect=(2,1), sharey=False) #, axwidth=5)
    # Plot first row
    ylabel = 'Wm$^{-2}$K$^{-1}$'
    grps = ['lw', 'sw', 'net']
    suptitles = ['Longwave', 'Shortwave', 'Net']
    var_names = []
    lines1 = []
    for ax, grp, suptitle in zip(axes[0:3], grps, suptitles):
        colors = ['k', 'r', 'g']
        for sec, color in zip(secs, colors):
            var_name = sec + '_' + grp + '_' + 'cld_tot'
            var_mean, var_std = get_ensemble_mean_std(ds_arr, var_name)
            dt_mean = calc_ma_tm_zm(var_mean)
            l = ax.plot(dt_mean.lat, dt_mean, '-', color=color, label=sec_title[sec])
            lines1.extend(l)
            dt_std = calc_ma_tm_zm(var_std)
            ax.fill_between(dt_std.lat, dt_mean-dt_std, dt_mean+dt_std, color=color, alpha=0.2)
        ax.set_title(suptitle, fontweight='bold')
        if 'lw' in grp.lower():
            ax.set_ylabel(ylabel)
    new_lines1, new_labels1 = get_unique_line_labels(lines1)
    axes[0].legend(new_lines1, new_labels1, frame=False)

    var_nms = ['cld_tot', 'cld_amt', 'cld_alt', 'cld_tau'] #, 'cld_err']
    var_labels = ['Total', 'Amount', 'Altitude', 'Optical depth'] #, 'Residual']
    colors = ['k', 'C1', 'purple', 'C2'] #, 'C3']
    for k, grp in enumerate(grps):
        lines2 = []
        for ax, sec in zip(axes[1:,k], secs):
            for var_nm, var_label, color in zip(var_nms, var_labels, colors):
                var_name = sec + '_' + grp + '_' + var_nm
                # dt = calc_ma_tm_zm(get_ensemble_mean(ds_arr, var_name))
                # l = ax.plot(dt.lat, dt, '-', color=color, label=var_label)
                var_mean, var_std = get_ensemble_mean_std(ds_arr, var_name)
                dt_mean = calc_ma_tm_zm(var_mean)
                l = ax.plot(dt_mean.lat, dt_mean, '-', color=color, label=var_label)
                lines1.extend(l)
                dt_std = calc_ma_tm_zm(var_std)
                ax.fill_between(dt_std.lat, dt_mean-dt_std, dt_mean+dt_std, color=color, alpha=0.2)
                lines2.extend(l)
            if k==0:
                ax.set_ylabel(sec_title[sec] + ' (' + ylabel + ')')

        new_lines2, new_labels2 = get_unique_line_labels(lines2)
        axes[3].legend(new_lines2, new_labels2, ncols=2, frame=False)

    for ax in axes:
        ax.plot([-90, 90], [0, 0], '--', color='gray')

    axes.format(xlabel='Latitude', xlocator=plot.arange(-90, 90, 30),
                xlim=[-90,90], ylim=[-2,2], grid=False,
                xminorticks=False, yminorticks=None, #yminorlocator=plot.arange(-2,2,0.5),
                xformatter='deglat', abc=True, abcstyle='(a)', abcloc='ul')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

def plot_zonal_mean_cldfbk_components(ds_arr, fig_name):
    secs = ['ALL', 'HI680', 'LO680']
    sec_title = {'ALL': 'All', 'HI680':'Non-low', 'LO680':'Low'}

    # Plot annual and zonal mean of cloud feedback components
    plot.close()
    fig, axes = plot.subplots(nrows=4, ncols=3, aspect=(2,1), sharey=False) #, axwidth=5)
    # Plot first row
    ylabel = 'Wm$^{-2}$K$^{-1}$'
    grps = ['lw', 'sw', 'net']
    suptitles = ['Longwave', 'Shortwave', 'Net']
    var_names = []
    lines1 = []
    for ax, grp, suptitle in zip(axes[0:3], grps, suptitles):
        colors = ['k', 'r', 'g']
        for sec, color in zip(secs, colors):
            var_name = sec + '_' + grp + '_' + 'cld_tot'
            dt = calc_ma_tm_zm(get_ensemble_mean(ds_arr, var_name))
            l = ax.plot(dt.lat, dt, '-', color=color, label=sec_title[sec])
            lines1.extend(l)
        ax.set_title(suptitle, fontweight='bold')
        if 'lw' in grp.lower():
            ax.set_ylabel(ylabel)
    new_lines1, new_labels1 = get_unique_line_labels(lines1)
    axes[0].legend(new_lines1, new_labels1, frame=False)

    var_nms = ['cld_tot', 'cld_amt', 'cld_alt', 'cld_tau'] #, 'cld_err']
    var_labels = ['Total', 'Amount', 'Altitude', 'Optical depth'] #, 'Residual']
    colors = ['k', 'C1', 'purple', 'C2'] #, 'C3']
    for k, grp in enumerate(grps):
        lines2 = []
        for ax, sec in zip(axes[1:,k], secs):
            for var_nm, var_label, color in zip(var_nms, var_labels, colors):
                var_name = sec + '_' + grp + '_' + var_nm
                dt = calc_ma_tm_zm(get_ensemble_mean(ds_arr, var_name))
                l = ax.plot(dt.lat, dt, '-', color=color, label=var_label)
                lines2.extend(l)
            if k==0:
                ax.set_ylabel(sec_title[sec] + ' (' + ylabel + ')')

        new_lines2, new_labels2 = get_unique_line_labels(lines2)
        axes[3].legend(new_lines2, new_labels2, ncols=2, frame=False)

    for ax in axes:
        ax.plot([-90, 90], [0, 0], '--', color='gray')

    axes.format(xlabel='Latitude', xlocator=plot.arange(-90, 90, 30),
                xlim=[-90,90], ylim=[-2,2], 
                xminorticks=False, yminorticks=None, #yminorlocator=plot.arange(-2,2,0.5),
                xformatter='deglat', abc=True, abcstyle='(a)', abcloc='ul')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

def plot_cld_fbk_components_map(ds_arr, fbk='net', fig_dir='./figs'):
    secs = ['ALL', 'HI680', 'LO680']
    sec_title = {'ALL': 'All', 'HI680':'Non-low', 'LO680':'Low'}

    var_nms = ['cld_tot', 'cld_amt', 'cld_alt', 'cld_tau'] #, 'cld_err']
    var_labels = ['Total', 'Amount', 'Altitude', 'Optical depth'] #, 'Residual']

    var_names = []
    var_titles = []
    for var_nm, var_label in zip(var_nms, var_labels):
        for sec in secs:
            var_names.append(sec + '_' + fbk + '_' + var_nm)
            var_titles.append(sec_title[sec] + ' ' + var_label.lower())
    dt_ensemble_mean = []
    dt_gm_arr = []
    for varnm in var_names:
        dt = get_ensemble_mean(ds_arr, varnm)
        dt_ensemble_mean.append(dt)
        dt_gm_arr.append(calc_ma_gm(dt))

    lons = ds_arr[0].lon
    lats = ds_arr[0].lat

    fbk_nm = {'net': 'net', 'lw': 'longwave', 'sw':'shortwave'}
    cnlevels = np.arange(-3, 3.2, 0.2)
    cmap = 'rdbu_r'
    plot.close()
    fig, axes = plot.subplots(nrows=len(var_nms), ncols=len(secs), proj='kav7', 
                    proj_kw={'lon_0': 180}, figsize=(6,6))
    axes.format(coast=True, latlines=30, lonlines=60,
        suptitle='Ensemble mean ' + fbk_nm[fbk] + ' cloud feedback') 
        #, abc=True, abcstyle='(a)')
    for kk, (ax, dt, dt_gm, varnm, var_title) in enumerate(zip(axes, 
                dt_ensemble_mean, dt_gm_arr, var_names, var_titles)):
        cs = ax.contourf(lons, lats, dt, cmap=cmap, extend='both', levels=cnlevels)
        ax.set_title('('+chr(97+kk)+') ' + var_title + ' [' + str(np.round(dt_gm, 2)) + ']')
    fig.colorbar(cs, loc='b', shrink=0.7, label='Wm$^{-2}$K$^{-1}$')
    fig_name = P(fig_dir, 'annual_mean_' + fbk + '_cld_fbk_map.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

def plot_cld_fbk_components_map_stddev(ds_arr, fbk='net', fig_dir='./figs'):
    secs = ['ALL', 'HI680', 'LO680']
    sec_title = {'ALL': 'All', 'HI680':'Non-low', 'LO680':'Low'}

    var_nms = ['cld_tot', 'cld_amt', 'cld_alt', 'cld_tau'] #, 'cld_err']
    var_labels = ['Total', 'Amount', 'Altitude', 'Optical depth'] #, 'Residual']

    var_names = []
    var_titles = []
    for var_nm, var_label in zip(var_nms, var_labels):
        for sec in secs:
            var_names.append(sec + '_' + fbk + '_' + var_nm)
            var_titles.append(sec_title[sec] + ' ' + var_label.lower())
    dt_ensemble_stddev = []
    dt_gm_arr = []
    for varnm in var_names:
        dt_mean, dt_stddev = get_ensemble_mean_std(ds_arr, varnm)
        dt_ensemble_stddev.append(dt_stddev)
        dt_gm_arr.append(calc_ma_gm(dt_stddev))

    lons = ds_arr[0].lon
    lats = ds_arr[0].lat

    fbk_nm = {'net': 'net', 'lw': 'longwave', 'sw':'shortwave'}
    cnlevels = np.arange(0, 1.1, 0.1)
    cmap = 'Oranges' #'rdbu_r'
    plot.close()
    fig, axes = plot.subplots(nrows=len(var_nms), ncols=len(secs), proj='kav7', 
                    proj_kw={'lon_0': 180}, figsize=(6,6))
    axes.format(coast=True, latlines=30, lonlines=60,
        suptitle='Ensemble stddev ' + fbk_nm[fbk] + ' cloud feedback') 
        #, abc=True, abcstyle='(a)')
    for kk, (ax, dt, dt_gm, varnm, var_title) in enumerate(zip(axes, 
                dt_ensemble_stddev, dt_gm_arr, var_names, var_titles)):
        cs = ax.contourf(lons, lats, dt, cmap=cmap, extend='both', levels=cnlevels)
        ax.set_title('('+chr(97+kk)+') ' + var_title + ' [' + str(np.round(dt_gm, 2)) + ']')
    fig.colorbar(cs, loc='b', shrink=0.7, label='Wm$^{-2}$K$^{-1}$')
    fig_name = P(fig_dir, 'annual_mean_' + fbk + '_cld_fbk_map_stddev.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

def plot_cld_fbk_components_map_for_each_run(ds, exp_nm, fbk='net', fig_dir='./figs'):
    secs = ['ALL', 'HI680', 'LO680']
    sec_title = {'ALL': 'All', 'HI680':'Non-low', 'LO680':'Low'}

    var_nms = ['cld_tot', 'cld_amt', 'cld_alt', 'cld_tau'] #, 'cld_err']
    var_labels = ['Total', 'Amount', 'Altitude', 'Optical depth'] #, 'Residual']

    var_names = []
    var_titles = []
    for var_nm, var_label in zip(var_nms, var_labels):
        for sec in secs:
            var_names.append(sec + '_' + fbk + '_' + var_nm)
            var_titles.append(sec_title[sec] + ' ' + var_label.lower())
    #dt_ensemble_mean = []
    dst_dt_arr = []
    dt_gm_arr = []
    for varnm in var_names:
        #dt = get_ensemble_mean(ds_arr, varnm)
        dt = ds[varnm]
        #dt_ensemble_mean.append(dt)
        dst_dt_arr.append(dt.mean('month'))
        dt_gm_arr.append(calc_ma_gm(dt))

    lons = ds_arr[0].lon
    lats = ds_arr[0].lat

    fbk_nm = {'net': 'net', 'lw': 'longwave', 'sw':'shortwave'}
    cnlevels = np.arange(-3, 3.2, 0.2)
    cmap = 'rdbu_r'
    plot.close()
    fig, axes = plot.subplots(nrows=len(var_nms), ncols=len(secs), proj='kav7', 
                    proj_kw={'lon_0': 180}, figsize=(6,6))
    axes.format(coast=True, latlines=30, lonlines=60,
        suptitle=fbk_nm[fbk] + ' cloud feedback') 
        #, abc=True, abcstyle='(a)')
    for kk, (ax, dt, dt_gm, varnm, var_title) in enumerate(zip(axes, 
                dst_dt_arr, dt_gm_arr, var_names, var_titles)):
                #dt_ensemble_mean, dt_gm_arr, var_names, var_titles)):
        cs = ax.contourf(lons, lats, dt, cmap=cmap, extend='both', levels=cnlevels)
        ax.set_title('('+chr(97+kk)+') ' + var_title + ' [' + str(np.round(dt_gm, 2)) + ']')
    fig.colorbar(cs, loc='b', shrink=0.7, label='Wm$^{-2}$K$^{-1}$')
    fig_name = P(fig_dir, 'annual_mean_' + fbk + '_cld_fbk_map_'+exp_nm+'.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

if __name__ == '__main__':
    P = os.path.join
    dt_dir = '../data'
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # fig_dir2 = '../figs/fbk_maps_individual'
    # if not os.path.exists(fig_dir2):
    #     os.mkdir(fig_dir2)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    # Read dataset
    ds_arr = []
    for exp_nm in exp_grps:
        #print(exp_nm)
        fn = P(dt_dir, 'cld_fbk_decomp_v2_' + exp_nm + '.nc')
        ds = xr.open_dataset(fn, decode_times=False)
        ds_arr.append(ds)

    # for ds in ds_arr:
    #     print(calc_ma_gm(ds.ALL_net_cld_tot))
    '''
    secs = ['ALL', 'HI680', 'LO680']
    var_nms = ['lw_cld_tot', 'sw_cld_tot', 'net_cld_tot',]

    var_names = []
    for sec in secs:
        for var_nm in var_nms:
            var_names.append(sec+'_'+var_nm)
    #print(var_names)

    dt_ensemble_mean = []
    for varnm in var_names:
        dt = get_ensemble_mean(ds_arr, varnm)
        dt_ensemble_mean.append(dt)

    lons = ds_arr[0].lon
    lats = ds_arr[0].lat

    print('Plot maps...')
    cnlevels = np.arange(-4, 4.1, 0.2)
    cmap = 'rdbu_r'
    plot.close()
    fig, axes = plot.subplots(nrows=3, ncols=3, proj='kav7', 
                    proj_kw={'lon_0': 180}, figsize=(6,5))
    axes.format(coast=True, latlines=30, lonlines=60, abc=True, abcstyle='(a)')
    for ax, dt, varnm in zip(axes, dt_ensemble_mean, var_names):
        cs = ax.contourf(lons, lats, dt, cmap=cmap, extend='both', levels=cnlevels)
        ax.set_title(varnm)
    fig.colorbar(cs, loc='b', shrink=0.8, label='Wm$^{-2}$K$^{-1}$')

    fig_name = P(fig_dir, 'annual_mean_tot_cld_fbk_map.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()
    '''
    for fbk in ['net', 'lw', 'sw']:
        print('Plot maps for ', fbk)
        plot_cld_fbk_components_map(ds_arr, fbk=fbk, fig_dir=fig_dir)

    # for fbk in ['net', 'lw', 'sw']:
    #     print('Plot maps for ', fbk)
    #     plot_cld_fbk_components_map_stddev(ds_arr, fbk=fbk, fig_dir=fig_dir)

    # for ds, exp_grp in zip(ds_arr,exp_grps):
    #     for fbk in ['net', 'lw', 'sw']:
    #         print('Plot maps for ', exp_grp, fbk)
    #         plot_cld_fbk_components_map_for_each_run(ds, exp_grp, fbk=fbk, fig_dir=fig_dir2)

    print('Plot zonal mean...')
    fig_name = P(fig_dir, 'annual_and_zonal_mean_cld_fbk.pdf')
    plot_zonal_mean_cldfbk_components(ds_arr, fig_name)
    
    print('Plot zonal mean with std...')
    fig_name = P(fig_dir, 'annual_and_zonal_mean_cld_fbk_std.pdf')
    plot_zonal_mean_cldfbk_components_with_std(ds_arr, fig_name)
