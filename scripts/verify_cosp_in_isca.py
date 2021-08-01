import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')
import proplot as plot
import copy
from analysis_functions import add_datetime_info
from bin_clisccp import construct_clisccp_tau_pres

def calc_ma_gm(dt):
    lat_axis = dt.dims.index('lat')
    coslat = np.cos(np.deg2rad(dt.lat))
    dt_tm_zm = calc_ma_tm_zm(dt)
    dt_tm_zm_ma = np.ma.masked_array(dt_tm_zm, mask=np.isnan(dt_tm_zm))
    dt_tm_gm = np.ma.average(dt_tm_zm_ma, axis=lat_axis, weights=coslat)

    return dt_tm_gm

def calc_ma_tm_zm(dt):
    dims = dt.dims
    # print(dims)
    out_dims = list(copy.deepcopy(dims))
    try:
        time_lon_axis = (dims.index('time'), dims.index('lon'))
        out_dims.remove('time').remove('lon')
    except:
        time_lon_axis = dims.index('lon')
        out_dims.remove('lon')
    coords = {}
    for d in out_dims:
        coords[d] = dt[d]
    # dt_ma = np.ma.masked_array(dt, mask=np.isnan(dt))
    # dt_tm_zm = np.ma.average(dt_ma, axis=time_lon_axis)
    dt_tm_zm = dt.mean(axis=time_lon_axis, skipna=True)
    dt_tm_zm = xr.DataArray(dt_tm_zm, dims=out_dims, coords=coords)
    return dt_tm_zm

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # =================== Read Isca dataset ================== #
    base_dir = '../inputs'
    ppe_dir = P(base_dir, 'qflux_extracted_data')
    ppe_clisccp_dir = P(base_dir, 'qflux_clisccp_data')

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    file_nm = 'extracted_data_301_360.nc'
    file_clisccp_nm = 'extracted_clisccp_data_301_360.nc'

    for exp_grp in exp_grps[0:1]:
        print(exp_grp, ': Read dataset...')

        fn = P(ppe_dir, file_nm.replace('.nc', '_'+exp_grp+'.nc'))
        ds = xr.open_dataset(fn, decode_times=False)

        fn = P(ppe_clisccp_dir, file_clisccp_nm.replace('.nc', '_'+exp_grp+'.nc'))
        ds_clisccp = xr.open_dataset(fn, decode_times=False)

        add_datetime_info(ds)
        construct_clisccp_tau_pres(ds_clisccp, ds)

        sum_clisccp = ds.clisccp.sum(('tau7', 'pres7')).mean('time')
        tot_cld_amt = ds.tot_cld_amt.mean('time').interp(lat=sum_clisccp.lat, lon=sum_clisccp.lon)
        dt_arr = [calc_ma_gm(ds.clisccp.mean('time').transpose('pres7', 'tau7',...)), sum_clisccp,
                  tot_cld_amt,  tot_cld_amt-sum_clisccp]
        var_titles = ['Joint histogram of ISCCP', r'Sum of histogram bins of ISCCP', 
                     'Total cloud amount from Isca', 'Isca - ISCCP']
        # dt_gm_arr = [ calc_ma_gm(dt) for dt in dt_arr]

        tau = [0, 0.3, 1.3, 3.6, 9.4, 23., 60, 380]
        ctp = [1000, 800, 680, 560, 440, 310, 180, '']

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=2, proj={1:None, 2:'kav7', 3:'kav7', 4:'kav7'}, 
                        proj_kw={'lon_0': 180}, share=False) #, axwidth=3) #, figsize=(6,4))
        axes[1:].format(coast=True, latlines=30, lonlines=60)
        for kk, (ax, dt, var_title) in enumerate(zip(axes, dt_arr, var_titles)):
            if kk == 0:
                vmax = 10
                cs = ax.pcolor(dt, cmap='Oranges', vmin=0, vmax=vmax, levels=20, extend='max')
                ax.set_title('('+chr(97+kk)+') ' + var_title)
                cbar = ax.colorbar(cs, loc='r', shrink=0.9, width='1em') # label='%'
                cbar.ax.set_title('%')
                ax.format(suptitle='', #abc=True, abcstyle='(a)',
                        xlim=[-0.5, len(tau)-1.5], xlocator='index', xformatter=[str(v) for v in tau],
                        ylim=[-0.5, len(ctp)-1.5], ylocator='index', yformatter=[str(v) for v in ctp],
                        xlabel=r'Optical depth', ylabel='Cloud top pressure (hPa)')
            else:
                if kk < 3:
                    cmap = 'Blues_r'
                    cnlevels = np.arange(0, 101, 10)
                    extend = 'neither'
                else:
                    cmap = 'RdBu_r'
                    cnlevels = np.arange(-10, 11, 1)
                    extend = 'both'
                dt_gm =  calc_ma_gm(dt)
                cs = ax.contourf(ds.lon, ds.lat, dt, cmap=cmap, extend=extend, levels=cnlevels)
                ax.set_title('('+chr(97+kk)+') ' + var_title + ' (' + str(np.round(dt_gm, 1)) + ' %)')
                cbar = ax.colorbar(cs, loc='r', shrink=0.6, width='1em') # label='%'
                cbar.ax.set_title('%')
    
        #fig.colorbar(cs1, loc='b', col=1, shrink=0.7, label='%', width='1em')
        #fig.colorbar(cs2, loc='b', col=2, shrink=0.7, label='%', width='1em')
        fig_name = P(fig_dir, 'cmp_isca_cosp_tot_cld_amt_with_diff.png')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=300)
        #plot.show()


        # =========================================================================== #
        dt_arr = [calc_ma_gm(ds.clisccp.mean('time').transpose('pres7', 'tau7',...)), 
                  sum_clisccp, tot_cld_amt,]
        var_titles = ['Joint histogram of ISCCP', r'Sum of histogram bins of ISCCP', 
                     'Total cloud amount from Isca']
        # dt_gm_arr = [ calc_ma_gm(dt) for dt in dt_arr]

        tau = [0, 0.3, 1.3, 3.6, 9.4, 23., 60, 380]
        ctp = [1000, 800, 680, 560, 440, 310, 180, '']

        plot.close()
        fig, axes = plot.subplots([[1,2],[1,3]], proj={1:None, 2:'kav7', 3:'kav7'}, 
                        proj_kw={'lon_0': 180}, share=False, axwidth=3.5) #, axwidth=3) #, figsize=(6,4))
        axes[1:].format(coast=True, latlines=30, lonlines=60)
        for kk, (ax, dt, var_title) in enumerate(zip(axes, dt_arr, var_titles)):
            if kk == 0:
                vmax = 10
                cs = ax.pcolor(dt, cmap='Oranges', vmin=0, vmax=vmax, levels=20, extend='max')
                ax.set_title('('+chr(97+kk)+') ' + var_title)
                cbar = ax.colorbar(cs, loc='r', shrink=0.9, width='1em') # label='%'
                cbar.ax.set_title('%')
                ax.format(suptitle='', #abc=True, abcstyle='(a)',
                        xlim=[-0.5, len(tau)-1.5], xlocator='index', xformatter=[str(v) for v in tau],
                        ylim=[-0.5, len(ctp)-1.5], ylocator='index', yformatter=[str(v) for v in ctp],
                        xlabel=r'Optical depth', ylabel='Cloud top pressure (hPa)')
            else:
                cmap = 'Blues_r'
                cnlevels = np.arange(0, 101, 10)
                extend = 'neither'

                dt_gm =  calc_ma_gm(dt)
                cs = ax.contourf(ds.lon, ds.lat, dt, cmap=cmap, extend=extend, levels=cnlevels)
                ax.set_title('('+chr(97+kk)+') ' + var_title + ' (' + str(np.round(dt_gm, 1)) + '%)')
                # cbar = ax.colorbar(cs, loc='r', shrink=0.6, width='1em') # label='%'
                # cbar.ax.set_title('%')
    
        #fig.colorbar(cs1, loc='b', col=1, shrink=0.7, label='%', width='1em')
        cbar = fig.colorbar(cs, loc='r', row=(1,2), shrink=0.9, width='1em')
        cbar.ax.set_title('%')
        fig_name = P(fig_dir, 'cmp_isca_cosp_tot_cld_amt.png')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=300)
        # plot.show()
        print(fig_name, 'saved')

        # =========================================================================== #
        # Add observed ISCCP histogram

        # ============ Read Observed ISCCP simulator data =========# 
        obs_dt_dir = '../inputs/'
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

        dt_arr = [calc_ma_gm(clisccp.mean('time').transpose('pres7', 'tau',...)), 
                  calc_ma_gm(ds.clisccp.mean('time').transpose('pres7', 'tau7',...)),
                 sum_clisccp, tot_cld_amt]
        var_titles = ['Observed ISCCP joint histogram', 'Joint ISCCP histogram from Isca',
                     r'Sum of histogram bins of ISCCP', 'Total cloud amount from Isca']
        # dt_gm_arr = [ calc_ma_gm(dt) for dt in dt_arr]

        tau = [0, 0.3, 1.3, 3.6, 9.4, 23., 60, 380]
        ctp = [1000, 800, 680, 560, 440, 310, 180, '']

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=2, proj={1:None, 2:None, 3:'kav7', 4:'kav7'}, 
                        proj_kw={'lon_0': 180}, share=False, axwidth=3.5) #, axwidth=3) #, figsize=(6,4))
        axes[2:].format(coast=True, latlines=30, lonlines=60)
        for kk, (ax, dt, var_title) in enumerate(zip(axes, dt_arr, var_titles)):
            if kk < 2:
                vmax = 10
                if kk==0:
                    dt = dt[::-1,:]
                cs1 = ax.pcolor(dt, cmap='Oranges', vmin=0, vmax=vmax, levels=20, extend='max')
                ax.set_title('('+chr(97+kk)+') ' + var_title)
                # cbar = ax.colorbar(cs1, loc='r', shrink=0.9, width='1em') # label='%'
                # cbar.ax.set_title('%')
                if kk==0:
                    ### ax.invert_yaxis() ### 1000 hPa at bottom (index order...)
                    ax.format(suptitle='', #abc=True, abcstyle='(a)',
                        xlim=[-0.5, len(tau)-2.5], xlocator='index', xformatter=[str(v) for v in tau[1:]],
                        ylim=[-0.5, len(ctp)-1.5], ylocator='index', yformatter=[str(v) for v in ctp],
                        xlabel=r'Optical depth', ylabel='Cloud top pressure (hPa)')
                else:
                    ax.format(suptitle='', #abc=True, abcstyle='(a)',
                            xlim=[-0.5, len(tau)-1.5], xlocator='index', xformatter=[str(v) for v in tau],
                            ylim=[-0.5, len(ctp)-1.5], ylocator='index', yformatter=[str(v) for v in ctp],
                            xlabel=r'Optical depth', ylabel='Cloud top pressure (hPa)')
            else:
                cmap = 'Blues_r'
                cnlevels = np.arange(0, 101, 10)
                extend = 'neither'

                dt_gm =  calc_ma_gm(dt)
                cs2 = ax.contourf(ds.lon, ds.lat, dt, cmap=cmap, extend=extend, levels=cnlevels)
                ax.set_title('('+chr(97+kk)+') ' + var_title + ' (' + str(np.round(dt_gm, 1)) + '%)')
                # cbar = ax.colorbar(cs2, loc='r', shrink=0.6, width='1em') # label='%'
                # cbar.ax.set_title('%')
    
        #fig.colorbar(cs1, loc='b', col=1, shrink=0.7, label='%', width='1em')
        cbar = fig.colorbar(cs1, loc='r', row=1, shrink=0.9, width='1em')
        cbar.ax.set_title('%')
        cbar = fig.colorbar(cs2, loc='r', row=2, shrink=0.5, width='1em')
        cbar.ax.set_title('%')
        fig_name = P(fig_dir, 'cmp_obs_isca_cosp_tot_cld_amt.png')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=300)
        # plot.show()
        print(fig_name, 'saved')

