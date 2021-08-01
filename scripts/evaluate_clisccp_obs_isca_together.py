import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import add_datetime_info
import copy
import proplot as plot
import matplotlib
import cmaps
from bin_clisccp import (construct_clisccp_tau_pres, bin_obs_exp_clisccp_data, 
                    get_percentile, evaluate_thin_intermediate_thick_clouds)

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # save_dt_dir = '../data/vertical_bins'
    # if not os.path.exists(save_dt_dir):
    #     os.mkdir(save_dt_dir)

    # Read land_mask data
    dt_dir = '../inputs'

    # ============ Read Observed ISCCP simulator data =========# 
    obs_dt_dir = '../inputs'
    old_obs_fn = P(obs_dt_dir, 'clisccp_198307-200806.nc')
    # # CDO not work as the clisscp is missed in the output netcdf file
    # start_year = 1998
    # end_year = 2007
    # new_obs_fn = P(obs_dt_dir, 'clisccp_'+str(start_year)+'_'+str(end_year)+'.nc')
    # if not os.path.exists(new_obs_fn):
    #     os.system(' '.join(['cdo', 'selyear,'+str(start_year)+','+str(end_year),
    #                          old_obs_fn, new_obs_fn]))
    ds_obs = xr.open_dataset(old_obs_fn, decode_times=False)
    try:
        ds_obs = ds_obs.rename({'latitude':'lat', 'longitude':'lon', 'plev7':'pres7'})
        
    except:
        print('No need to rename lat/lon.')

    #add_datetime_info(ds_obs)
    start_year = 1998
    end_year = 2007
    start_ind = (start_year - 1984) * 12 + (12 - 7) + 1
    end_ind = start_ind + (end_year - start_year + 1) * 12
    print(end_ind-start_ind)
    clisccp = ds_obs.clisccp[start_ind:end_ind]
    clisccp['pres7'] = ds_obs.pres7 / 1e2
    
    # ============ Read ERAI omega data ============ #
    print('Vertical velocity omega from ERAI...')
    obs_base_dir = '../inputs/'
    old_fnm_omega = P(obs_base_dir, 'ecmwf_omega_1979_2017_t42.nc')
    new_fnm_omega = P(obs_base_dir, 'ecmwf_omega_'+str(start_year)+'_'+str(end_year)+'_t42.nc')

    if not os.path.exists(new_fnm_omega):
        print('select year from', start_year, 'to', end_year)
        os.system(' '.join(['cdo', 'selyear,'+str(start_year)+'/'+str(end_year),
                    old_fnm_omega, new_fnm_omega]))
    ds_w = xr.open_dataset(new_fnm_omega, decode_times=False)
    add_datetime_info(ds_w)
    omega = ds_w.w
    omega_coeff = 3600. * 24. / 100.
    omega500 = omega.sel(level=500).interp(lat=ds_obs.lat, lon=ds_obs.lon) * omega_coeff
    
    ds_clisccp_obs = {}
    ds_clisccp_obs['clisccp'] = (clisccp.dims, clisccp.values)
    ds_clisccp_obs['omega500'] = (omega500.dims, omega500.values)
    coords = {}
    for dim in clisccp.dims:
        if 'time' in dim:
            coords[dim] = ds_w.time
        else:
            coords[dim] = clisccp[dim]
    ds_clisccp_obs = xr.Dataset(ds_clisccp_obs, coords=coords)
    add_datetime_info(ds_clisccp_obs)
    # delete temporary file
    os.remove(new_fnm_omega)

    # =================== Read Isca dataset ================== #
    base_dir = '../inputs/'
    ppe_dir = P(base_dir, 'qflux_extracted_data')
    ppe_clisccp_dir = P(base_dir, 'qflux_clisccp_data')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']
    file_clisccp_nms = ['extracted_clisccp_data_301_360.nc', 
                        'extracted_clisccp_data_661_720.nc']

    for exp_grp in exp_grps[0:1]:
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        ds_clisccp_arr = []
        
        for file_nm, file_clisccp_nm in zip(file_nms, file_clisccp_nms):
            fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

            fn = P(ppe_clisccp_dir, file_clisccp_nm.replace('.nc', '_'+exp_grp+'.nc'))
            ds = xr.open_dataset(fn, decode_times=False)
            ds_clisccp_arr.append(ds)

        # Keep the time coordinates the same
        ds_arr[1]['time'] = ds_arr[0].time

        print('Construct clisccp')
        for ds, ds_clisccp in zip(ds_arr, ds_clisccp_arr):
            add_datetime_info(ds)
            #construct_clisccp(ds)
            #construct_clisccp2(ds_clisccp, ds)
            construct_clisccp_tau_pres(ds_clisccp, ds)

        #print('Begin kernel calculation...')
        #nyr = 2
        ds1 = ds_arr[0] #.where(ds_arr[0].year>30-nyr, drop=True)
        ds2 = ds_arr[1]

        bin_nm = 'omega500'
        land_sea_mask = 'ocean'
        grp_time = 'year' #None
        s_lat = -30
        n_lat = 30
        bins = np.arange(-100, 101, 10)

        # ds_bin = bin_isca_exp_clisccp_data(ds1, s_lat=s_lat, n_lat=n_lat, bin_var_nm=bin_nm, bin_var=None,
        #         grp_time_var=grp_time, bins=bins, land_sea=land_sea_mask, land_mask_dir=dt_dir)
        ds_bin = bin_obs_exp_clisccp_data(ds1, s_lat=s_lat, n_lat=n_lat, bin_var_nm=bin_nm, bin_var=None,
                grp_time_var=grp_time, bins=bins, land_sea=land_sea_mask, land_mask_dir=dt_dir)

        # dt_fn = '_'.join(filter(None, ['ds_bin', bin_nm, 'clisccp', grp_time, land_sea_mask]))
        # ds_bin.to_netcdf(P(save_dt_dir, dt_fn+'.nc'), mode='w', format='NETCDF3_CLASSIC')
        # print(dt_fn + '.nc saved.')
        # fig_name = P(fig_dir, 'clisccp_cloud_thickness_category_example.pdf')
        # evaluate_thin_intermediate_thick_clouds(ds_bin, fig_name=fig_name)

        ds_obs_bin = bin_obs_exp_clisccp_data(ds_clisccp_obs, s_lat=s_lat, n_lat=n_lat, bin_var_nm=bin_nm, bin_var=None,
                grp_time_var=grp_time, bins=bins, land_sea=land_sea_mask, land_mask_dir=dt_dir)

        # dt_fn = '_'.join(filter(None, ['ds_bin', bin_nm, 'clisccp_obs', grp_time, land_sea_mask]))
        # ds_bin.to_netcdf(P(save_dt_dir, dt_fn+'.nc'), mode='w', format='NETCDF3_CLASSIC')
        # print(dt_fn + '.nc saved.')
        # fig_name = P(fig_dir, 'clisccp_cloud_thickness_category_obs.pdf')
        # evaluate_thin_intermediate_thick_clouds(ds_obs_bin, fig_name=fig_name)

        # ============ Plot obs and isca together ==============# 
        print('Begin plot clisccp')
        tau = [0, 0.3, 1.3, 3.6, 9.4, 23., 60, 380]
        ctp = [1000, 800, 680, 560, 440, 310, 180, 50]

        #matplotlib.rcParams['text.usetex'] = True

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=4, axwidth=1.9)

        #cmap = 'Blues' # cmaps.MPL_PuOr_r
        dt_arr = []
        titles_arr = []

        for kk, ds in enumerate([ds_obs_bin, ds_bin]):
            tau_ind1 = [(1,3), (3,5), (5,7), (0,7)]
            if 'tau7' in ds.dims:
                tau_ind = tau_ind1
                tau_dim = 'tau7'
            else:
                tau_ind = [(x-1, y-1) for (x,y) in tau_ind1]
                tau_ind[-1] = (0, 6) # correct the last index
                tau_dim = 'tau'
                # array([ 90.5, 245. , 375. , 500. , 620. , 740. , 950. ] hPa, reverse them
                # pcolor (1000 at bottom)
                ds['pres7'] = np.arange(6, -1, -1)

            for ind in tau_ind:
                dt = ds.clisccp[:,ind[0]:ind[1],:,:].mean('year').sum(tau_dim)
                dt_arr.append(dt)
            i_titles_arr = [ r'Thin (%.1f $\leq \tau<$ %.1f, obs)'% (tau[1], tau[3]),
                        r'Intermediate (%.1f $\leq \tau<$ %.1f, obs)'% (tau[3], tau[5]),
                        r'Thick ($\tau\geq$ %.1f, obs)'% (tau[5]),  'All (obs)']
            if kk == 0:
                titles_arr.extend(i_titles_arr)
            else:
                titles_arr.extend([t.replace('obs', 'isca') for t in i_titles_arr])

        cs_arr = []
        for kk, (ax, dt, title) in enumerate(zip(axes, dt_arr, titles_arr)):
            if 'all' in title.lower():
                vmax = 30
            else:
                vmax = 10
            if kk==3 or kk==7:
                cmap = 'blue8' #cmaps.MPL_pink_r #Greens #'Blues'
            else:
                cmap = 'Blues'
            cs = ax.pcolor(dt, cmap=cmap, vmin=0, vmax=vmax, levels=20, extend='max')
            cs_arr.append(cs)
            ax.set_title(title)
            # if kk//4 == 1:
            #     cbar = ax.colorbar(cs, loc='b', label='\%') # label='%'
            #     #cbar.ax.set_title('(\%)')
        fig.colorbar(cs_arr[0], loc='b', col=1, label='%')
        fig.colorbar(cs_arr[-1], loc='b', col=4, label='%')
        axes.format(suptitle='', abc=True, abcstyle='(a)', abcloc='ur',
                    xlim=[-50,50], xminorlocator=10,
                    ylim=[-0.5, len(ctp)-1.5], ylocator='index', 
                    yformatter=[str(v) for v in ctp],
                    xlabel=r'Vetical velocity ($\omega_{500}$, hPa day$^{-1}$)',
                    ylabel='Cloud top pressure (hPa)')
        #axes.set_xlabel(r'Vetical velocity ($\omega_{500}$, hPa day$^{-1}$)', fontsize=18)
        fig_name = P(fig_dir, 'clisccp_cloud_thickness_category_obs_isca_linear_sc.pdf')
        fig.savefig(fig_name, bbox_inches='tight', transparent=False)
        #plot.show()
        print(fig_name, 'saved')

