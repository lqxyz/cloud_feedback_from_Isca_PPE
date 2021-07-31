from __future__ import print_function, division
import numpy as np
import xarray as xr
import os
import matplotlib.ticker as ticker
import netCDF4
from isca.util import interpolate_output


def open_experiment(exp_folder_name, start_file, end_file, file_name='atmos_monthly.nc', nbits=4, base_dir=None, decode_times=False):
    """
    Modified from 
    """
    if base_dir is None:
        base_dir = os.environ['GFDL_DATA']
    folder_list = ['run%0{0}d'.format(nbits) % m for m in range(start_file, end_file+1)]
    files = [os.path.join(base_dir, exp_folder_name, folder_list[i], file_name) for i in range(len(folder_list))]
    files_exist=[os.path.isfile(s) for s in files]

    if not(all(files_exist)):
        raise EOFError('EXITING BECAUSE OF MISSING FILES', [files[elem] for elem in range(len(files_exist)) if not files_exist[elem]])

    ds = xr.open_mfdataset(files, decode_times=decode_times) #, autoclose=True) # not used in py3

    return ds

def get_ds_arr_from_exps(exp_folder_names, start_file, end_file, 
        file_name='atmos_monthly.nc', nbits=4, base_dir=None, decode_times=False):
    """ exp_folder_names is a string array, start_file and end_file
    are the integer indicating the starting/ending months """
    ds_arr = []
    for exp_folder_nm in exp_folder_names:
        ds = open_experiment(exp_folder_nm, start_file, end_file, 
            file_name=file_name,nbits=nbits, base_dir=base_dir, decode_times=decode_times)
        ds_arr.append(ds)
    return ds_arr

def get_var_arr_from_ds(ds_arr, var_name):
    var_arr = []
    for ds in ds_arr:
        var_arr.append(ds[var_name])
    return var_arr


def close_extra_axes(axes, n_figs_keeped):
    try:
        axes1 = axes.flatten()
    except:
        axes1 = axes
    # for i, ax in enumerate(axes1):
    #     if i >= n_figs_keeped:
    #         ax.remove()
    for ax in axes1[n_figs_keeped:]:
        ax.remove()


def get_unique_line_labels(lines):
    """ Parameters: lines should be the returned value from ax.plot(x, y, ...)"""
    #labls_tmp = set([l.get_label() for l in lines])
    labls_tmp = []
    for l in lines:
        if l.get_label() not in labls_tmp:
            labls_tmp.append(l.get_label())

    new_lines = []
    new_labels = []
    for labl in labls_tmp:
        for l in lines:
            if labl == l.get_label():
                new_lines.append(l)
                new_labels.append(labl)
                break
    return new_lines, new_labels


def set_ax_tick_format(ax, xbase=None, ybase=None, ):
    """Remove the extra zeros in the tick labels;
    If possible, set tick locators at specified interval"""

    if xbase is not None:
        loc = ticker.MultipleLocator(base=xbase)
        ax.xaxis.set_major_locator(loc)
    if ybase is not None:
        loc = ticker.MultipleLocator(base=ybase)
        ax.yaxis.set_major_locator(loc)

    formatter = ticker.FuncFormatter(lambda x, pos: ('%f'% x).rstrip('0').rstrip('.'))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def set_axis_ratio(ax, ratio):
    [xmin, xmax] = ax.get_xlim()
    [ymin, ymax] = ax.get_ylim()
    #ax.set_aspect(float(np.abs(xmax - xmin) / np.abs(ymax - ymin) * ratio), 'box-forced')
    ax.set_aspect(float(np.abs(xmax - xmin) / np.abs(ymax - ymin) * ratio), adjustable='box')


def z_to_p(z):
    """Change altitude (m) to pressure (hPa)
    Refer to: https://en.wikipedia.org/wiki/Atmospheric_pressure"""
    p0 = 1.01325e5  # Pa
    T0 = 288.15     # K
    g = 9.80665     # m/s^2
    M = 0.0289644   # kg/mol
    R0 = 8.31447    # J/mol/K
    p = p0 * np.exp(-g * z * M / T0 / R0) / 1e2
    return p

def add_datetime_info(data):
    time = data.time
    try:
        dates = netCDF4.num2date(time, time.units, time.calendar)
    except:
        # print('No calendar attribute in time.')
        dates = netCDF4.num2date(time, time.units)

    years = []
    months = []
    seasons = []
    days = []
    hours = []
    for now in dates:
        years.append(now.year)
        seasons.append((now.month%12 + 3)//3)
        months.append(now.month)
        days.append(now.day)
        hours.append(now.hour)
    data.coords['month'] = ('time', months)
    data.coords['year'] = ('time', years)
    data.coords['season'] = ('time', seasons)
    data.coords['day'] = ('time', days)
    data.coords['hour'] = ('time', hours)


def global_average_lat_lon(ds_in, var_name):
    coslat = np.cos(np.deg2rad(ds_in.lat))
    var = ds_in[var_name]
    var_in_dims = var.dims
    lat_ind = var_in_dims.index('lat')
    var_gm = np.average(var.mean('lon'), axis=lat_ind, weights=coslat)
    var_out_dims = tuple(x for x in var_in_dims if x!='lat' and x!='lon')
    ds_in[var_name+'_gm'] = (var_out_dims, var_gm)


def sigma_to_pressure_level(exp_folder_name, start_file, end_file, in_file_name='atmos_monthly.nc',
         out_file_name='atmos_monthly_plev.nc', nbits=4, base_dir=None, var_names=['slp', 'height'],
         p_levs='input', all_fields=True):

    if base_dir is None:
        base_dir = os.environ['GFDL_DATA']

    folder_list = ['run%0{0}d'.format(nbits) % m for m in range(start_file, end_file+1)]
    files = [os.path.join(base_dir, exp_folder_name, folder_list[i], in_file_name) for i in range(len(folder_list))]
    files_exist=[os.path.isfile(s) for s in files]

    if not(all(files_exist)):
        raise EOFError('EXITING BECAUSE OF MISSING FILES', [files[elem] for elem in range(len(files_exist)) if not files_exist[elem]])
    
    out_files = [f.replace(in_file_name, out_file_name) for f in files]

    for infile, outfile in zip(files, out_files):
        interpolate_output(infile, outfile, all_fields=all_fields, p_levs=p_levs, var_names=var_names)

def get_global_mean(dt):
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
    return dt_gm

def detrend_dim(da, dim, deg=1, skipna=True):
    # https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg, skipna=skipna)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1, skipna=True):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg, skipna=skipna)
    return da_detrended
