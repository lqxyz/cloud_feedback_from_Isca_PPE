import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import numpy as np
from scipy import stats
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')
import copy
import proplot as plot
import cmaps

def get_percentile(dt):
    dims = dt.dims
    coords = {}
    for d in dims:
        coords[d] = dt.coords[d]
    dt_percentile = stats.rankdata(dt) / np.size(dt) * 1e2  # Units: %
    dt_percentile = np.reshape(dt_percentile, dt.shape)
    dt_percentile = xr.DataArray(dt_percentile, dims=dims, coords=coords)
    return dt_percentile

def construct_clisccp_tau_pres(ds, ds_no_clisccp):
    # get demisions size
    ntime = len(ds.time)
    ntau = len(ds.tau7)
    try:
        npres = len(ds.pres7)
    except:
        npres = 7
    nlat = len(ds.lat)
    nlon = len(ds.lon)

    clisccp = np.zeros((ntime, npres, ntau, nlat, nlon), dtype="float32")
    for i in range(npres):
        clisccp[:,i,:,:,:] = ds['clisccp_'+str(i+1)]

    dims = ('time', 'pres7', 'tau7', 'lat', 'lon')
    ds_no_clisccp['clisccp'] = (dims, clisccp)
    dim = 'tau7'
    ds_no_clisccp[dim] = ((dim), ds[dim])

    new_dims = ('time', 'tau7', 'pres7', 'lat', 'lon')
    clisccp_t = ds_no_clisccp['clisccp'].transpose('time', 'tau7', 'pres7', 'lat', 'lon')
    ds_no_clisccp['clisccp'] = (new_dims, clisccp_t)

def get_final_nyr_mean(dt, n=2):
    return dt[-12*n:, :, :].mean('time')

def select_clisccp_data(ds_m, bin_data_dict, ds_mask, bins,
      bin_var_nm='omega500', land_sea='ocean', grp_time_var='year',
      var_names=['clisccp']):
    for vn in var_names:
        bin_data_dict[vn] = ds_m[vn]
    coords = {}
    dims = ds_m[var_names[0]].dims #['time', 'tau7', 'pres7', 'lat', 'lon']
    try:
        tau_dim = 'tau7'
        tau7 = ds_m[tau_dim]
        pres7 = ds_m.pres7
    except:
        tau_dim = 'tau'
        tau7 = ds_m[tau_dim]
        pres7 = ds_m.pres7

    for dim in dims:
        coords[dim] = ds_m[dim]
    ds_bin_m = xr.Dataset(bin_data_dict, coords=coords)

    ds_bin_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    for varname, da in ds_bin_m.data_vars.items():
        if land_sea == 'ocean': # or 'ocn' in land_sea:
            ds_bin_m[varname] = da.where(ds_bin_m.mask==0)
        if land_sea == 'land':
            ds_bin_m[varname] = da.where(ds_bin_m.mask==1)
    if 'lts' in bin_var_nm.lower():
        for varname, da in ds_bin_m.data_vars.items():
            ds_bin_m[varname] = da.where(ds_bin_m.lts>min(bins))
    
    pdf_m, ds_bin_mean_m = bin_clisccp_data(ds_bin_m, bins, 
            grp_time_var=grp_time_var, bin_var_nm=bin_var_nm)

    # print('========================')
    # print(ds_bin_mean_m.keys())
    # for key, val in ds_bin_mean_m.items():
    #     print(key, np.shape(val))
    # print('========================')

    # Write data in xarray dataset format
    bins_coord = (bins[0:-1] + bins[1:]) / 2.0

    if grp_time_var is not None:
        ds_grp = ds_m.groupby(grp_time_var).mean('time')
        coords = {grp_time_var: ds_grp[grp_time_var], tau_dim:tau7, 
                  'pres7':pres7, 'bin':bins_coord}
        dims = (grp_time_var, tau_dim, 'pres7', 'bin')
    else:
        coords = {tau_dim:tau7, 'pres7':pres7, 'bin':bins_coord}
        dims = (tau_dim, 'pres7', 'bin')
    return pdf_m, ds_bin_mean_m, dims, coords

def bin_clisccp_data(ds, bins, grp_time_var='year', bin_var_nm='omega500'):
    """
    ds: a xarray dataset containing several variables
    """
    if grp_time_var is not None:
        ds_grp = ds.groupby(grp_time_var).mean('time')
        ntime = len(ds_grp[grp_time_var])

        nbins = len(bins)
        npres7 = 7
        try:
            tau_dim = 'tau7'
            ntau7 = len(ds[tau_dim])
        except:
            tau_dim = 'tau'
            ntau7 = len(ds[tau_dim])
        pdf = np.zeros((ntime, nbins-1))
        ds_bin_mean = {}
        for var in ds.variables:
            if len(ds[var].shape) == 5:
                ds_bin_mean[var] = np.zeros((ntime, ntau7, npres7, nbins-1))

        for i in range(ntime):
            #print('bin i=', i)
            if grp_time_var is not None:
                ds_i = ds_grp.isel({grp_time_var: i})
            else:
                ds_i = ds.isel({'time':i})
            pdf[i,:] = np.histogram(ds_i[bin_var_nm], bins=bins, density=True)[0]
            grouped = ds_i.groupby_bins(bin_var_nm, bins).mean()
            # print(grouped)
            grouped_t = grouped.transpose(tau_dim, 'pres7', bin_var_nm+'_bins')
            # print('transpose:', grouped_t)
            for var in ds.variables:
                if len(ds[var].shape) == 5:
                    ds_bin_mean[var][i,:,:,:] = grouped_t.variables.get(var)
    else:
        ds_bin_mean = {}
        pdf = np.histogram(ds[bin_var_nm], bins=bins, density=True)[0]
        grouped = ds.groupby_bins(bin_var_nm, bins).mean(skipna=True)
        try:
            tau_dim = 'tau7'
            ntau7 = len(ds[tau_dim]) # Just test
        except:
            tau_dim = 'tau'
            ntau7 = len(ds[tau_dim])
        grouped_t = grouped.transpose(tau_dim, 'pres7', bin_var_nm+'_bins')
        for var in ds.variables:
            if len(ds[var].shape) == 5:
                ds_bin_mean[var] = grouped_t.variables.get(var)

    return pdf, ds_bin_mean

def bin_obs_exp_clisccp_data(ds, s_lat=-30, n_lat=30, bin_var_nm='omega500', 
        bin_var=None, grp_time_var='year', bins=np.arange(0,1.1,0.1), land_sea='global',
        land_mask_dir='./data', var_names_to_bin=['clisccp']):
    """ Return binned data for obs and isca dataset (clisccp type) based on certain variable
        such as vertical pressure velocity at 500hPa (omega500)...
    """
    ds_m = ds.where(np.logical_and(ds.lat>=s_lat, ds.lat<=n_lat), drop=True)

    ds_mask = xr.open_dataset(os.path.join(land_mask_dir, 'era_land_t42.nc'), decode_times=False)
    if len(ds.lat) != len(ds_mask.lat):
        ds_mask = ds_mask.interp(lat=ds.lat, lon=ds.lon)
    ds_mask = ds_mask.where(np.logical_and(ds_mask.lat>=s_lat,ds_mask.lat<=n_lat), drop=True)
    #ds_m.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    try: # For obs
        omega500_m = ds_m.omega500
    except: # For isca
        omega_coeff = 3600. * 24. / 100.
        omega_m = ds_m.omega * omega_coeff
        try:
            omega500_m = omega_m.sel(pfull=500)
        except:
            omega500_m = omega_m.interp(pfull=500)

    bin_data_dict = {}
    if bin_var is None:
        bin_data_dict['omega500'] = omega500_m
    else:
        omega500_obs_t = np.zeros_like(omega500_m)
        omega500_obs_lat_range = bin_var.where(np.logical_and(bin_var.lat>=s_lat, bin_var.lat<=n_lat), drop=True)
        for t in range(len(ds_m.time)):
            omega500_obs_t[t,:,:] = omega500_obs_lat_range
        omega500_obs_t = xr.DataArray(omega500_obs_t, coords=[ds_m.time, ds_m.lat, ds_m.lon],
                dims=['time', 'lat', 'lon'])
        bin_data_dict['omega500'] = omega500_obs_t

    # Add percentile for each variable
    bin_data_dict_tmp = copy.deepcopy(bin_data_dict)
    for key, val in bin_data_dict_tmp.items():
        val_percentile = get_percentile(val)
        bin_data_dict[key + '_percentile'] = val_percentile

    vars_dict = {}
    ## clisccp variables
    bin_data_dict2 = copy.deepcopy(bin_data_dict)
    pdf_m, ds_bin_mean_m, dims, coords2 = select_clisccp_data(ds_m, bin_data_dict2, ds_mask,
                    bins, bin_var_nm=bin_var_nm, land_sea=land_sea, grp_time_var=grp_time_var,
                    var_names=var_names_to_bin)
    for key, val in ds_bin_mean_m.items():
        vars_dict[key] = (dims, val)

    dims2 = tuple([d for d in dims if (not 'tau' in d) and (d != 'pres7')])
    vars_dict['pdf'] = (dims2, pdf_m)
    ds_bin_mean_m_array = xr.Dataset(vars_dict, coords=coords2)

    return ds_bin_mean_m_array

def evaluate_thin_intermediate_thick_clouds(ds, fig_name='test.pdf'):
    tau = [0, 0.3, 1.3, 3.6, 9.4, 23., 60, 380]
    ctp = [1000, 800, 680, 560, 440, 310, 180, 50]

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

    dt_arr = []
    title_arr = []
    for ind in tau_ind:
        dt = ds.clisccp[:,ind[0]:ind[1],:,:].mean('year').sum(tau_dim)
        dt_arr.append(dt)
    titles_arr = [ r'Thin (%.1f $\le \tau<$ %.1f)'% (tau[1], tau[3]),
                r'Intermediate (%.1f $\le \tau<$ %.1f)'% (tau[3], tau[5]),
                r'Thick ($\tau\ge$ %.1f)'% (tau[5]),  'All']

    cmap = 'Blues' # cmaps.MPL_PuOr_r
    plot.close()
    mpl.rcParams['text.usetex'] = True
    fig, axes = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True) #, figsize=(14,10)
    for ax, dt, title in zip(axes, dt_arr, titles_arr):
        if 'all' in title.lower():
            vmax = 20
        else:
            vmax = 10
        cs = ax.pcolor(dt, cmap=cmap, vmin=0, vmax=vmax, levels=20, extend='max')
        ax.set_title(title)
        cbar = ax.colorbar(cs, loc='r') # label='%'
        cbar.ax.set_title('(\%)')

    axes.format(suptitle='', abc=True, abcstyle='(a)',
                xlim=[-50,50], xminorlocator=10,
                ylim=[-0.5, len(ctp)-1.5], ylocator='index', yformatter=[str(v) for v in ctp],
                xlabel=r'$\omega_{500}$ (hPa day$^{-1}$)', ylabel='Cloud top pressure (hPa)')

    fig.savefig(fig_name, bbox_inches='tight', transparent=False)
    #plot.show()
