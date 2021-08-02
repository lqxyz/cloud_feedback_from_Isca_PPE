import os
import sys
import numpy as np
import xarray as xr
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

def map_SWkern_to_lon(Ksw, albcsmap):
    from scipy.interpolate import interp1d
    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,kernel_lats,3
    # albcsmap is size A,kernel_lats,kernel_lons
    
    albcs = np.arange(0.0, 1.5, 0.5) 
    A = albcsmap.shape[0]
    TT = Ksw.shape[1]
    PP = Ksw.shape[2]
    nlat = Ksw.shape[3]
    nlon = albcsmap.shape[2]
    SWkernel_map = np.ones((A, TT, PP, nlat, nlon)) * np.nan

    for M in range(A):
        MM = M
        while MM > 11:
            MM = MM - 12
        for i_lat in range(nlat):
            alon = albcsmap[M, i_lat, :] 
            # interp1d can't handle mask but it can deal with NaN (?)
            #try:
            #    alon2 = np.ma.masked_array(alon, np.isnan(alon)) 
            #except:
            #    alon2 = alon
            alon2 = alon
            if np.ma.count(alon2) > 1: # at least 1 unmasked value
                if np.nansum(Ksw[MM,:,:,i_lat,:] > 0) == 0:
                    SWkernel_map[M,:,:,i_lat,:] = 0
                else:
                    f = interp1d(albcs, Ksw[MM,:,:,i_lat,:], axis=2)
                    ynew = f(alon2)
                    #ynew= np.ma.masked_array(alon2, np.isnan(alon2))
                    SWkernel_map[M,:,:,i_lat,:] = ynew
            else:
                continue

    return SWkernel_map

def construct_clisccp(ds):
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
    ds['clisccp'] = (dims, clisccp)

def get_rsuscs_rsdscs_from_soc(ds):
    sw_down_sfc_clr = ds.soc_surf_flux_sw_down_clr
    sw_net_sfc_clr = ds.soc_surf_flux_sw_clr
    sw_up_sfc_clr = - (sw_net_sfc_clr - sw_down_sfc_clr)
    dims = ds.soc_surf_flux_sw_down_clr.dims
    # Surface Upwelling Clear-Sky Shortwave Radiation
    ds['rsuscs'] = (dims, sw_up_sfc_clr)
    # Surface Downwelling Clear-Sky Shortwave Radiation
    ds['rsdscs'] = (dims, sw_down_sfc_clr)

def calc_annual_mean_cld_flux_with_cld_kernel(ds_ctrl, ds_perturb, kernel_lats, kernel_lons):
    coslat = np.cos(np.deg2rad(kernel_lats))

    # Load in clisccp from two models
    clisccp1 = ds_ctrl.clisccp.transpose("time", "tau7", "pres7", "lat", "lon")
    clisccp2 = ds_perturb.clisccp.transpose("time", "tau7", "pres7", "lat", "lon")

    # Make sure clisccp is in percent  
    sumclisccp1 = np.nansum(clisccp1, axis=(1,2))
    sumclisccp2 = np.nansum(clisccp2, axis=(1,2))   
    if np.max(sumclisccp1) <= 1.:
        clisccp1 = clisccp1 * 1e2        
    if np.max(sumclisccp2) <= 1.:
        clisccp2 = clisccp2 * 1e2

    # Compute climatological annual cycle:
    try:
        avgclisccp1 = clisccp1.groupby('time.month').mean('time') #(12, TAU, CTP, LAT, LON)
        avgclisccp2 = clisccp2.groupby('time.month').mean('time') #(12, TAU, CTP, LAT, LON)
    except:
        avgclisccp1 = clisccp1.groupby('month').mean('time') #(12, TAU, CTP, LAT, LON)
        avgclisccp2 = clisccp2.groupby('month').mean('time') #(12, TAU, CTP, LAT, LON)

    # Compute clisccp anomalies
    anomclisccp = avgclisccp2 - avgclisccp1
    # Compute clear-sky surface albedo
    get_rsuscs_rsdscs_from_soc(ds_ctrl)

    rsuscs1 = ds_ctrl.rsuscs
    rsdscs1 = ds_ctrl.rsdscs

    albcs1 = rsuscs1 / rsdscs1
    try:
        avgalbcs1 = albcs1.groupby('time.month').mean('time') #(12, 90, 144)
    except:
        avgalbcs1 = albcs1.groupby('month').mean('time') #(12, 90, 144)
    
    # where(condition, x, y) is x where condition is true, y otherwise
    avgalbcs1 = xr.where(avgalbcs1>1., 1, avgalbcs1) 
    avgalbcs1 = xr.where(avgalbcs1<0., 0, avgalbcs1)

    # Regrid everything to the kernel grid:
    avganomclisccp = anomclisccp

    avgalbcs1_grd = avgalbcs1.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avgclisccp1_grd = avgclisccp1.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avgclisccp2_grd = avgclisccp2.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avganomclisccp_grd = avganomclisccp.interp(lat=kernel_lats, lon=kernel_lons, method="linear")

    # Use control albcs to map SW kernel to appropriate longitudes
    SWkernel_map = map_SWkern_to_lon(SWkernel, avgalbcs1_grd)

    # Compute clisccp anomalies normalized by global mean delta tas
    #anomclisccp = avganomclisccp_grd / avgdtas
    anomclisccp = avganomclisccp_grd

    # Compute feedbacks: Multiply clisccp anomalies by kernels
    SW0 = SWkernel_map * anomclisccp
    LW_cld_delta_flux = LWkernel_map * anomclisccp

    # Set the SW cloud feedbacks to zero in the polar night
    # The sun is down if every bin of the SW kernel is zero:
    sundown = np.nansum(SWkernel_map, axis=(1,2))  #12,90,144
    repsundown = np.tile(np.tile(sundown, (1,1,1,1,1)), (7,7,1,1,1))
    repsundown = np.transpose(repsundown, axes=[2,1,0,3,4])

    SW1 = np.where(repsundown==0, 0, SW0)
    SW_cld_delta_flux = np.where(np.isnan(repsundown), 0, SW1)

    sumLW = np.average(np.nansum(LW_cld_delta_flux, axis=(1,2)), axis=0)
    coslat = np.cos(np.deg2rad(kernel_lats))
    avgLW_cld_delta_flux = np.average(np.mean(sumLW, axis=1), axis=0, weights=coslat)
    sumSW = np.average(np.nansum(SW_cld_delta_flux, axis=(1,2)), axis=0)
    avgSW_cld_delta_flux = np.average(np.mean(sumSW, axis=1), axis=0, weights=coslat)
    cld_flux_arr = [avgLW_cld_delta_flux, avgSW_cld_delta_flux, 
                    avgLW_cld_delta_flux + avgSW_cld_delta_flux]
    return cld_flux_arr

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    dt_dir = '../data'
    # Read kernel data
    kernel_dt_dir = '../inputs'
    # Load in the Zelinka et al 2012 kernels:
    #kernel_nm = P(kernel_dt_dir, 'obs_cloud_kernels3.nc')
    kernel_nm = P(kernel_dt_dir, 'cloud_kernels2.nc')
    f = xr.open_dataset(kernel_nm) #, decode_times=False)

    LWkernel = f.LWkernel
    SWkernel = f.SWkernel

    LWkernel = xr.where(np.isnan(LWkernel), np.nan, LWkernel)
    SWkernel = xr.where(np.isnan(SWkernel), np.nan, SWkernel)
    print('LWkernel', LWkernel.dims, LWkernel.shape)

    albcs = np.arange(0.0, 1.5, 0.5) # the clear-sky albedos over which the kernel is computed

    # LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
    if 'kernels3' in kernel_nm:
        LWkernel_map = np.tile(np.tile(LWkernel[:,:,:,:], (1,1,1,1,1)), (144,1,1,1,1))
    else: #if 'kernels2' in kernel_nm:
        LWkernel_map = np.tile(np.tile(LWkernel[:,:,:,:,0], (1,1,1,1,1)), (144,1,1,1,1))
    print('LWkernel_map', np.shape(LWkernel_map))
    LWkernel_map = np.transpose(LWkernel_map, axes=[1,2,3,4,0])
    print('LWkernel_map', np.shape(LWkernel_map))

    # Define the cloud kernel axis attributes
    kernel_lats = LWkernel.lat
    kernel_lons = np.arange(1.25, 360, 2.5)

    # =================== Read Isca dataset ================== #
    base_dir = '../inputs/'
    file_nm_1xco2 = 'extracted_flux_data_30yr_1xco2.nc'
    file_nm_4xco2 = 'extracted_flux_data_30yr_4xco2.nc'

    print('Read dataset...')
    ds_arr = []
    for file_nm in [file_nm_1xco2, file_nm_4xco2]:
        fn = P(base_dir, 'cld_fbk_cmp', file_nm)
        ds = xr.open_dataset(fn, decode_times=False)
        ds_arr.append(ds)
    
    ds_arr[1]['time'] = ds_arr[0].time

    print('Construct clisccp')
    for ds in ds_arr:
        add_datetime_info(ds)
        construct_clisccp(ds)

    print('Begin kernel calculation...')
    nyr = 30
    lw_flux = np.zeros(nyr)
    sw_flux = np.zeros(nyr)
    net_flux = np.zeros(nyr)
    for nn in range(nyr):
        print('year: ', nn)
        # ========== Note that ds1 is the last 10 year data of the 30year data=====#
        ds1 = ds_arr[0].isel(time=np.arange(20*12, 30*12, 1))
        ds2 = ds_arr[1].isel(time=np.arange(nn*12, (nn+1)*12, 1))
        print(ds2.time.values)
        flux = calc_annual_mean_cld_flux_with_cld_kernel(ds1, ds2, kernel_lats, kernel_lons)
        lw_flux[nn] = flux[0]
        sw_flux[nn] = flux[1]
        net_flux[nn] = flux[2]
        print(' flux is ', flux)

    print('Saving data...')
    try:
        tsurf = ds_arr[1].t_surf.groupby('year').mean()
    except:
        tsurf = ds_arr[1].t_surf.groupby('time.year').mean()
    flux_dict = {}
    dims = 'year'
    years = tsurf.year
    flux_dict['toa_lw_cld_flux'] = (dims, lw_flux)
    flux_dict['toa_sw_cld_flux'] = (dims, sw_flux)
    flux_dict['toa_net_cld_flux'] = (dims, net_flux)

    ds_save = xr.Dataset(flux_dict, coords={dims:years})
    if 'kernels3' in kernel_nm:
        fn_save = P(dt_dir, 'toa_cld_flux.nc')
    else: # kernels2
        fn_save = P(dt_dir, 'toa_cld_flux_v2.nc')
    ds_save.to_netcdf(fn_save, mode='w', format='NETCDF3_CLASSIC')

    print(fn_save + ' saved.')
