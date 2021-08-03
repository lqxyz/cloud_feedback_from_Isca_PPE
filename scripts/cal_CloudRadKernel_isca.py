#=============================================
# Performs the cloud feedback and cloud error
# metric calculations in preparation for comparing
# to expert-assessed values from Sherwood et al (2020)
#=============================================
import numpy as np
import xarray as xr
import pandas as pd
from datetime import date
from zelinka_analysis import map_SWkern_to_lon, KT_decomposition_general
from pathlib import Path
import sys
from analysis_functions import add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca
from bin_clisccp_area_weights_time import (construct_clisccp_tau_pres, bin_obs_exp_clisccp_data, get_final_nyr_mean)
import organize_jsons as OJ
import json

datadir = Path('../inputs/zelinka_data')

# Define a python dictionary containing the sections of the histogram to consider
# These are the same as in Zelinka et al, GRL, 2016
sections = ['ALL', 'HI680', 'LO680']
Psections=[slice(0,7), slice(2,7), slice(0,2)]
sec_dic = dict(zip(sections, Psections))

# TR = cdutil.region.domain(latitude=(-30.,30.))
# # 10 hPa/dy wide bins:
# width = 10
# binedges=np.arange(-100,100,width)
# x1=np.arange(binedges[0]-width,binedges[-1]+2*width,width)
# binmids = x1+width/2.
# cutoff = np.int(len(binmids)/2) # [:cutoff] = ascent; [cutoff:-1] = descent; [-1] = land

land_mask_dir = '../inputs/'
bin_nm = 'omega500'
land_sea_mask = 'ocean'
grp_time = 'month' #None # 'year'
s_lat = -30 
n_lat = 30
bins = np.arange(-110, 111, 10) 
cutoff = np.argmax(bins>0)

# Load in the Zelinka et al 2012 kernels:
kernel_version = 'v2'
if kernel_version == 'v2':
    kernel_nm = datadir / 'cloud_kernels2.nc'
else: # v3
    kernel_nm = datadir / 'obs_cloud_kernels3.nc'
f = xr.open_dataset(kernel_nm, decode_times=False)
LWkernel = f.LWkernel
SWkernel = f.SWkernel
LWkernel = xr.where(np.isnan(LWkernel), np.nan, LWkernel)
SWkernel = xr.where(np.isnan(SWkernel), np.nan, SWkernel)
# print('LWkernel', LWkernel.dims, LWkernel.shape)

# the clear-sky albedos over which the kernel is computed
albcs = np.arange(0.0, 1.5, 0.5)
# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
if 'kernels3' in str(kernel_nm):
    LWkernel_map = np.tile(np.tile(LWkernel[:,:,:,:], (1,1,1,1,1)), (144,1,1,1,1))
else: #if 'kernels2' in str(kernel_nm):
    LWkernel_map = np.tile(np.tile(LWkernel[:,:,:,:,0], (1,1,1,1,1)), (144,1,1,1,1))
#print('LWkernel_map', np.shape(LWkernel_map))
LWkernel_map = np.transpose(LWkernel_map, axes=[1,2,3,4,0])
#print('LWkernel_map', np.shape(LWkernel_map))

# Define the cloud kernel axis attributes
kernel_lats = LWkernel.lat
kernel_lons = np.arange(1.25, 360, 2.5)
kernel_lons = xr.DataArray(kernel_lons, dims=('lon'), coords={'lon': kernel_lons})

dims = list(LWkernel.dims)
dims[-1] = 'lon'
coords = {}
for d in dims:
    if d != 'lon':
        coords[d] = LWkernel[d]
    else:
        coords[d] = kernel_lons
LWkernel_map = xr.DataArray(LWkernel_map, dims=dims, coords=coords)
LWkernel_map = LWkernel_map.rename({'mo':'time', 'tau_midpt':'tau7', 'p_midpt':'pres7'})
LWkernel_map.coords['month'] = ('time', np.arange(1, 13, 1))

##########################################################
##### Load in ISCCP HGG clisccp climo annual cycle  ######
##########################################################
f = xr.open_dataset(datadir / 'AC_clisccp_ISCCP_HGG_198301-200812.nc', decode_times=False)
#f.history='Written by /work/zelinka1/scripts/load_ISCCP_HGG.py on feedback.llnl.gov'
#f.comment='Monthly-resolved climatological annual cycle over 198301-200812'
obs_clisccp_AC = f.AC_clisccp

f = xr.open_dataset(datadir / 'AC_clisccp_wap_ISCCP_HGG_198301-200812.nc', decode_times=False)
#f.history='Written by /work/zelinka1/scripts/load_ISCCP_HGG.py on feedback.llnl.gov'
#f.comment='Monthly-resolved climatological annual cycle over 198301-200812, in omega500 space'
obs_clisccp_AC_wap = f.AC_clisccp_wap
obs_N_AC_wap = f.AC_N_wap

def get_rsuscs_rsdscs_from_soc(ds):
    sw_down_sfc_clr = ds.soc_surf_flux_sw_down_clr
    sw_net_sfc_clr = ds.soc_surf_flux_sw_clr
    sw_up_sfc_clr = - (sw_net_sfc_clr - sw_down_sfc_clr)
    dims = ds.soc_surf_flux_sw_down_clr.dims
    # Surface Upwelling Clear-Sky Shortwave Radiation
    ds['rsuscs'] = (dims, sw_up_sfc_clr)
    # Surface Downwelling Clear-Sky Shortwave Radiation
    ds['rsdscs'] = (dims, sw_down_sfc_clr)

def get_monthly_clim(dt):
    try:
        dt_clim = dt.groupby('time.month').mean('time')
    except:
        dt_clim = dt.groupby('month').mean('time')
    return dt_clim

def get_area_wts(dt):
    # Create map of area weights
    lats = dt.lat
    lons = dt.lon
    ntime, nlat, nlon = dt.shape # (ntime, nlat, nlon)
    coslat = np.cos(np.deg2rad(lats))
    coslat2 = coslat / np.nansum(coslat) / nlon
    # summing this over lat and lon = 1
    area_wts = np.moveaxis(np.tile(coslat2, [ntime, nlon, 1]), 1, 2)

    dims = dt.dims
    coords = {}
    for d in dims:
        coords[d] = dt[d]
    area_wts = xr.DataArray(area_wts, dims=dims, coords=coords)

    return area_wts

def get_CRK_data_from_isca(ds1, ds2):
    # Read in data, regrid and map kernels to lat/lon
    
    # Load in regridded monthly mean climatologies from control and perturbed simulation
    print('    control run')
    get_rsuscs_rsdscs_from_soc(ds1)
    ctl_tas = get_monthly_clim(ds1.temp_2m)
    ctl_rsdscs = get_monthly_clim(ds1.rsdscs)
    ctl_rsuscs = get_monthly_clim(ds1.rsuscs)
    try:
        ctl_wap = get_monthly_clim(ds1.omega.sel(pfull=500))
    except:
        ctrl_wap = get_monthly_clim(ds1.omeaga.interp(pfull=500))
    ctl_clisccp = get_monthly_clim(ds1.clisccp.interp(lat=kernel_lats, lon=kernel_lons))

    print('    perturbed run')
    get_rsuscs_rsdscs_from_soc(ds2)
    fut_tas = get_monthly_clim(ds2.temp_2m)
    fut_rsdscs = get_monthly_clim(ds2.rsdscs)
    fut_rsuscs = get_monthly_clim(ds2.rsuscs)
    try:
        fut_wap = get_monthly_clim(ds2.omega.sel(pfull=500))
    except:
        fut_wap = get_monthly_clim(ds2.omeaga.interp(pfull=500))
    fut_clisccp = get_monthly_clim(ds2.clisccp.interp(lat=kernel_lats, lon=kernel_lons))

    # Make sure wap is in hPa/day
    wap_coeff =  3600. * 24. / 100.
    ctl_wap = wap_coeff * ctl_wap   # Pa/s --> hPa/day
    fut_wap = wap_coeff * fut_wap   # Pa/s --> hPa/day

    # Make sure clisccp is in percent
    sumclisccp1 = np.nansum(ctl_clisccp, axis=(1,2))
    sumclisccp2 = np.nansum(fut_clisccp, axis=(1,2))   
    if np.max(sumclisccp1) <= 1.:
        ctl_clisccp = ctl_clisccp * 1e2        
    if np.max(sumclisccp2) <= 1.:
        fut_clisccp = fut_clisccp * 1e2

    # Compute clear-sky surface albedo
    ctl_albcs = ctl_rsuscs / ctl_rsdscs # (12, 90, 144)
    # where(condition, x, y) is x where condition is true, y otherwise
    ctl_albcs = xr.where(ctl_albcs>1., 1.0, ctl_albcs) 
    ctl_albcs = xr.where(ctl_albcs<0., 0.0, ctl_albcs)

    ctl_albcs_grd = ctl_albcs.interp(lat=kernel_lats, lon=kernel_lons) #, method="linear")
    # Use control albcs to map SW kernel to appropriate longitudes
    SWkernel_map = map_SWkern_to_lon(SWkernel, ctl_albcs_grd)

    dims = list(LWkernel_map.dims)
    coords = {}
    for d in dims:
        coords[d] = LWkernel_map[d]
    SWkernel_map = xr.DataArray(SWkernel_map, dims=dims, coords=coords)
    SWkernel_map.coords['month'] = ('time', np.arange(1, 13, 1))

    ds_kernel_map = {}
    ds_kernel_map['SWkernel_map'] = (dims, SWkernel_map)
    ds_kernel_map['LWkernel_map'] = (dims, LWkernel_map)
    wap_for_kernel = ctl_wap.interp(lat=kernel_lats, lon=kernel_lons)
    #print(wap_for_kernel.shape)
    ds_kernel_map['omega500'] = (('time', 'lat', 'lon'), wap_for_kernel)
    ds_kernel_map = xr.Dataset(ds_kernel_map, coords=coords)
    #print(ds_kernel_map)

    # Compute global annual mean tas anomalies
    anom_tas = fut_tas - ctl_tas
    coslat1 = np.cos(np.deg2rad(anom_tas.lat))
    avgdtas = np.average(anom_tas.mean(('month', 'lon')), axis=0, weights=coslat1) # (scalar)
    print('avgdtas =', avgdtas)

    # ======================== For omega500 bins ===================== #
    print('Sort into omega500 bins')
    # TR = cdutil.region.domain(latitude=(-30.,30.))
    # ctl_wap_ocean,ctl_wap_land = apply_land_mask_v2(TR.select(ctl_wap))
    # fut_wap_ocean,fut_wap_land = apply_land_mask_v2(TR.select(fut_wap))
    # ctl_OKwaps = BA.bony_sorting_part1(ctl_wap_ocean,binedges)
    # fut_OKwaps = BA.bony_sorting_part1(fut_wap_ocean,binedges)

    area_wts = get_area_wts(ctl_tas) # summing this over lat and lon = 1

    print('Bin ctrl clisccp')
    ds_bin_ctl = bin_obs_exp_clisccp_data(ds1, s_lat=s_lat, n_lat=n_lat, 
            bin_var_nm=bin_nm, bin_var=None, grp_time_var=grp_time, bins=bins, 
            land_sea=land_sea_mask, land_mask_dir=land_mask_dir,
            var_names_to_bin=['clisccp'])
    ctl_clisccp_wap = ds_bin_ctl.clisccp
    # print(ds_bin_ctl)

    print('Bin fut clisccp')
    ds_bin_fut = bin_obs_exp_clisccp_data(ds2, s_lat=s_lat, n_lat=n_lat, 
            bin_var_nm=bin_nm, bin_var=None, grp_time_var=grp_time, bins=bins, 
            land_sea=land_sea_mask, land_mask_dir=land_mask_dir,
            var_names_to_bin=['clisccp'])
    fut_clisccp_wap = ds_bin_fut.clisccp

    print('For kernel map')
    ds_bin_k = bin_obs_exp_clisccp_data(ds_kernel_map, s_lat=s_lat, n_lat=n_lat, 
            bin_var_nm=bin_nm, bin_var=None, grp_time_var=grp_time, bins=bins, 
            land_sea=land_sea_mask, land_mask_dir=land_mask_dir,
            var_names_to_bin=['SWkernel_map', 'LWkernel_map'])
    LWK_wap = ds_bin_k.LWkernel_map
    SWK_wap = ds_bin_k.SWkernel_map
    
    ctl_N = ds_bin_ctl.area_sum
    fut_N = ds_bin_fut.area_sum
    print('ctl_N:', ctl_N)
    print('fut_N:', fut_N)

    return (ctl_clisccp, fut_clisccp, LWkernel_map, SWkernel_map, avgdtas, 
            ctl_clisccp_wap, fut_clisccp_wap, LWK_wap, SWK_wap, ctl_N, fut_N)

def select_regions(field, region):
    lats = field.lat
    if region == 'eq90':
        inds=np.where((np.abs(lats[:])<90))
    elif region == 'eq60':
        inds=np.where((np.abs(lats[:])<60))
    elif region == 'eq30':
        inds=np.where((np.abs(lats[:])<30))
    elif region == '30-60':
        inds=np.where((np.abs(lats[:])<60) & (np.abs(lats[:])>30))
    elif region == '30-80':
        inds=np.where((np.abs(lats[:])<80) & (np.abs(lats[:])>30))
    elif region == '40-70':
        inds=np.where((np.abs(lats[:])<70) & (np.abs(lats[:])>40))
    elif region == 'Arctic':
        inds=np.where((lats[:]>70))
    field_dom = np.take(field, inds[0], axis=-2)
    return(field_dom)

def klein_metrics(obs_clisccp, gcm_clisccp, LWkern, SWkern, WTS):
    ########################################################
    ######### Compute Klein et al (2013) metrics ###########
    ########################################################

    # Remove the thinnest optical depth bin from models/kernels so as to compare properly with obs:
    # print('    gcm_clisccp shp:', gcm_clisccp.shape, 'obs_clisccp shp:', obs_clisccp.shape)
    gcm_clisccp = gcm_clisccp[:,1:,:,:]
    LWkern = LWkern[:,1:,:,:]
    SWkern = SWkern[:,1:,:,:]

    ## Compute Cloud Fraction Histogram Anomalies w.r.t. observations
    clisccp_bias = np.array(gcm_clisccp) - np.array(obs_clisccp)

    ## Multiply Anomalies by Kernels
    SW = SWkern * clisccp_bias
    LW = LWkern * clisccp_bias
    # print('SW', SW.shape, SWkern.shape, clisccp_bias.shape)
    NET = SW + LW

    ########################################################
    # E_TCA (TOTAL CLOUD AMOUNT METRIC)
    ########################################################
    # take only clouds with tau>1.3
    WTS_dom = WTS / 12
    WTS_dom = WTS_dom / np.nansum(WTS_dom) # np.nansum(WTS_dom) = 1, so weighted sums give area-weighted avg, NOT scaled by fraction of planet
    obs_clisccp_dom = obs_clisccp[:,1:,:,]
    gcm_clisccp_dom = gcm_clisccp[:,1:,:,]

    # sum over CTP and TAU:
    gcm_cltisccp_dom = np.nansum(gcm_clisccp_dom, axis=(1,2)) #MV.sum(MV.sum(gcm_clisccp_dom,1),1) # (time, lat, lon)
    obs_cltisccp_dom = np.nansum(obs_clisccp_dom, axis=(1,2)) #MV.sum(MV.sum(obs_clisccp_dom,1),1) # (time, lat, lon)

    # 1) Denominator (Eq. 3 in Klein et al. (2013))
    avg = np.nansum(obs_cltisccp_dom * WTS_dom) # (scalar)
    anom1 = obs_cltisccp_dom - avg # anomaly of obs from its spatio-temporal mean
    # 2) Numerator -- Model minus ISCCP
    anom2 = gcm_cltisccp_dom - obs_cltisccp_dom  # (time, lat, lon)

    E_TCA_denom = np.sqrt(np.nansum(WTS_dom * anom1**2)) # (scalar)
    E_TCA_numer2 = np.sqrt(np.nansum(WTS_dom * anom2**2)) # (scalar)

    E_TCA = E_TCA_numer2 / E_TCA_denom

    ########################################################
    # CLOUD PROPERTY METRICS
    ########################################################
    # take only clouds with tau>3.6
    clisccp_bias_dom = clisccp_bias[:,2:,:]
    obs_clisccp_dom = obs_clisccp[:,2:,:]
    gcm_clisccp_dom = gcm_clisccp[:,2:,:]
    LWkernel_dom = LWkern[:,2:,:]
    SWkernel_dom = SWkern[:,2:,:]
    NETkernel_dom = SWkernel_dom + LWkernel_dom

    # Compute anomaly of obs histogram from its spatio-temporal mean
    #print(obs_clisccp_dom.shape)
    # try:
    #     this = np.transpose(obs_clisccp_dom, axes=[1,2,0,3,4]) #np.moveaxis(obs_clisccp_dom, 0, 2) # [TAU,CTP,month,space]
    # except:
    #     this = np.transpose(obs_clisccp_dom, axes=[1,2,0,3])
    this = np.moveaxis(np.array(obs_clisccp_dom), 0, 2)
    #print(this.shape)
    if np.ndim(WTS_dom)==2: # working in wap space
        avg_obs_clisccp_dom = np.nansum(np.nansum(np.array(this)*np.array(WTS_dom),-1),-1) # (TAU,CTP)
    else: # working in lat/lon space
        #print(this.shape, WTS_dom.shape)
        #np.nansum(np.nansum(np.nansum(this*WTS_dom,-1),-1),-1) # (TAU,CTP)
        #avg_obs_clisccp_dom = np.nansum(np.array(this)*np.array(WTS_dom), axis=(2,3,4))
        avg_obs_clisccp_dom = np.nansum(np.nansum(np.nansum(np.array(this)*np.array(WTS_dom),-1),-1),-1)  # (TAU,CTP)
    this = np.moveaxis(np.moveaxis(np.array(obs_clisccp_dom),1,-1),1,-1) - avg_obs_clisccp_dom
    #this =  np.transpose(np.array(obs_clisccp_dom), axes=(0,3,4,1,2)) - avg_obs_clisccp_dom #  1,-1),1,-1)
    #anom_obs_clisccp_dom = np.transpose(this, axes=(0,3,4,1,2)) #np.moveaxis(np.moveaxis(this,-1,1),-1,1)
    anom_obs_clisccp_dom = np.moveaxis(np.moveaxis(this,-1,1), -1, 1)

    ## Compute radiative impacts of cloud fraction anomalies
    gcm_NET_bias = NET[:,2:,:]
    obs_NET_bias = anom_obs_clisccp_dom * NETkernel_dom
    gcm_SW_bias = SW[:,2:,:]
    obs_SW_bias = anom_obs_clisccp_dom * SWkernel_dom
    gcm_LW_bias = LW[:,2:,:]
    obs_LW_bias = anom_obs_clisccp_dom * LWkernel_dom

    ## Aggregate high, mid, and low clouds over medium and thick ISCCP ranges
    CTPmids = obs_clisccp[obs_clisccp.dims[2]]#.getAxis(2)[:]
    Psec_name = ['LO', 'MID', 'HI']
    Psec_bnds = ((1100,680),(680,440),(440,10))
    Psec_dic=dict(zip(Psec_name,Psec_bnds))
    Tsec_name = ['MED', 'THICK']
    Tsections = [slice(0,2), slice(2,4)]
    Tsec_dic = dict(zip(Tsec_name, Tsections))

    agg_obs_NET_bias = np.zeros(gcm_SW_bias.shape)
    #print(' agg_obs_NET_bias:',  agg_obs_NET_bias.shape, gcm_SW_bias.shape)
    agg_gcm_NET_bias = np.zeros(gcm_SW_bias.shape)
    agg_obs_SW_bias = np.zeros(gcm_SW_bias.shape)
    agg_gcm_SW_bias = np.zeros(gcm_SW_bias.shape)
    agg_obs_LW_bias = np.zeros(gcm_SW_bias.shape)
    agg_gcm_LW_bias = np.zeros(gcm_SW_bias.shape)
    agg_obs_clisccp_bias = np.zeros(gcm_SW_bias.shape)
    agg_gcm_clisccp_bias = np.zeros(gcm_SW_bias.shape)

    obs_NET_bias = np.where(np.isnan(obs_NET_bias), 0, obs_NET_bias)
    gcm_NET_bias = np.where(np.isnan(gcm_NET_bias), 0, gcm_NET_bias)
    obs_SW_bias = np.where(np.isnan(obs_SW_bias), 0, obs_SW_bias)
    gcm_SW_bias = np.where(np.isnan(gcm_SW_bias), 0, gcm_SW_bias)
    obs_LW_bias = np.where(np.isnan(obs_LW_bias), 0, obs_LW_bias)
    gcm_LW_bias = np.where(np.isnan(gcm_LW_bias), 0, gcm_LW_bias)
    anom_obs_clisccp_dom = np.where(np.isnan(anom_obs_clisccp_dom), 0, anom_obs_clisccp_dom)
    clisccp_bias_dom = np.where(np.isnan(clisccp_bias_dom), 0, clisccp_bias_dom)

    tt=-1
    for Tsec in Tsec_name:
        tt += 1
        TT = Tsec_dic[Tsec]
        pp = -1
        for Psec in Psec_name:
            pbot,ptop = Psec_dic[Psec]
            PP = np.where(np.logical_and(CTPmids<=pbot, CTPmids>ptop))[0]
            if len(CTPmids[PP])>0:
                pp += 1
                #print('obs_NET_bias:', Tsec, Psec, tt, pp, TT, PP, agg_obs_NET_bias.shape, obs_NET_bias.shape)
                agg_obs_NET_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(obs_NET_bias)[:,TT,PP,:],1),1)
                agg_gcm_NET_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(gcm_NET_bias)[:,TT,PP,:],1),1)
                agg_obs_SW_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(obs_SW_bias)[:,TT,PP,:],1),1)
                agg_gcm_SW_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(gcm_SW_bias)[:,TT,PP,:],1),1)
                agg_obs_LW_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(obs_LW_bias)[:,TT,PP,:],1),1)
                agg_gcm_LW_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(gcm_LW_bias)[:,TT,PP,:],1),1)
                agg_obs_clisccp_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(anom_obs_clisccp_dom)[:,TT,PP,:],1),1)
                agg_gcm_clisccp_bias[:,tt,pp,:] = np.nansum(np.nansum(np.array(clisccp_bias_dom)[:,TT,PP,:],1),1)
    NP = pp + 1
    NT = tt + 1

    ## Compute E_ctp-tau -- Cloud properties error
    ctot1 = np.nansum(np.nansum(agg_gcm_clisccp_bias**2,1),1)/(NT*NP)
    ctot2 = np.nansum(np.nansum(agg_obs_clisccp_bias**2,1),1)/(NT*NP)

    ## Compute E_LW -- LW-relevant cloud properties error
    ctot3 = np.nansum(np.nansum(agg_gcm_LW_bias**2,1),1)/(NT*NP)
    ctot4 = np.nansum(np.nansum(agg_obs_LW_bias**2,1),1)/(NT*NP)

    ## Compute E_SW -- SW-relevant cloud properties error
    ctot5 = np.nansum(np.nansum(agg_gcm_SW_bias**2,1),1)/(NT*NP)
    ctot6 = np.nansum(np.nansum(agg_obs_SW_bias**2,1),1)/(NT*NP)

    ## Compute E_NET -- NET-relevant cloud properties error
    ctot7 = np.nansum(np.nansum(agg_gcm_NET_bias**2,1),1)/(NT*NP)
    ctot8 = np.nansum(np.nansum(agg_obs_NET_bias**2,1),1)/(NT*NP)

    # compute one metric
    E_ctpt_numer = np.sqrt(np.nansum(WTS_dom*ctot1)) # (scalar)
    E_ctpt_denom = np.sqrt(np.nansum(WTS_dom*ctot2)) # (scalar)
    E_LW_numer = np.sqrt(np.nansum(WTS_dom*ctot3)) # (scalar)
    E_LW_denom = np.sqrt(np.nansum(WTS_dom*ctot4)) # (scalar)
    E_SW_numer = np.sqrt(np.nansum(WTS_dom*ctot5)) # (scalar)
    E_SW_denom = np.sqrt(np.nansum(WTS_dom*ctot6)) # (scalar)
    E_NET_numer = np.sqrt(np.nansum(WTS_dom*ctot7)) # (scalar)
    E_NET_denom = np.sqrt(np.nansum(WTS_dom*ctot8)) # (scalar)

    E_ctpt = E_ctpt_numer / E_ctpt_denom
    E_LW = E_LW_numer / E_LW_denom
    E_SW = E_SW_numer / E_SW_denom
    E_NET = E_NET_numer / E_NET_denom

    return(E_TCA, E_ctpt, E_LW, E_SW, E_NET)

def apply_land_mask_v2(data, land_mask_dir=Path('../data/')):
    """
    apply land mask (data):
    this will read in and reshape the land-sea mask to match data
    """
    # Call the cdutil function to generate a mask, 0 for ocean, 1 for land.
    ds_mask = xr.open_dataset(land_mask_dir / 'era_land_t42.nc', decode_times=False)
    if len(data.lat) != len(ds_mask.lat):
        ds_mask = ds_mask.interp(lat=data.lat, lon=data.lon)
    data.coords['mask'] = (('lat', 'lon'), ds_mask.land_mask.values)

    ocean_data = data.where(data.mask==0)
    land_data = data.where(data.mask==1)

    return (ocean_data, land_data)

def do_klein_calcs(ctl_clisccp, LWK, SWK, obs_clisccp_AC, ctl_clisccp_wap,
                   LWK_wap, SWK_wap, obs_clisccp_AC_wap, obs_N_AC_wap):
    KEM_dict = {} # dictionary to contain all computed Klein error metrics
    for sec in sections:
        print('sec:', sec)
        KEM_dict[sec]={}
        PP=sec_dic[sec]
        C1 = ctl_clisccp[:,:,PP,:]
        Klw = LWK[:,:,PP,:]
        Ksw = SWK[:,:,PP,:]

        obs_C1 = obs_clisccp_AC[:,:,PP,:]
        ocn_obs_C1,lnd_obs_C1 = apply_land_mask_v2(obs_C1, land_mask_dir=Path(land_mask_dir))
        ocn_C1, lnd_C1 = apply_land_mask_v2(C1, land_mask_dir=Path(land_mask_dir))

        WTS = get_area_wts(obs_C1[:,0,0,:]) # summing this over lat and lon = 1
        #print('WTS:', WTS.shape)
        # assessed feedback regions + Klein region (eq60)
        for region in ['eq90', 'eq60', 'eq30', '30-60', '30-80', '40-70', 'Arctic']:
            print('  region:', region)
            KEM_dict[sec][region]={}
            obs_C1_dom = select_regions(obs_C1, region)
            ocn_obs_C1_dom = select_regions(ocn_obs_C1, region)
            lnd_obs_C1_dom = select_regions(lnd_obs_C1, region)
            C1_dom = select_regions(C1, region)
            ocn_C1_dom = select_regions(ocn_C1, region)
            lnd_C1_dom = select_regions(lnd_C1, region)
            Klw_dom = select_regions(Klw, region)
            Ksw_dom = select_regions(Ksw, region)
            WTS_dom = select_regions(WTS, region)
            for sfc in ['all', 'ocn', 'lnd', 'ocn_asc', 'ocn_dsc']:
                print('    sfc:', sfc)
                # print('    obs_C1, C1, Klw, Ksw, WTS shapes:', obs_C1_dom.shape,  
                #     C1_dom.shape, Klw_dom.shape, Ksw_dom.shape, WTS_dom.shape)
                KEM_dict[sec][region][sfc]={}
                if sfc=='all':
                    (E_TCA, E_ctpt, E_LW, E_SW, E_NET) = klein_metrics(obs_C1_dom,
                                                C1_dom, Klw_dom, Ksw_dom, WTS_dom)
                elif sfc=='ocn':
                    (E_TCA, E_ctpt, E_LW, E_SW, E_NET) = klein_metrics(ocn_obs_C1_dom,
                                                ocn_C1_dom, Klw_dom, Ksw_dom, WTS_dom)
                elif sfc=='lnd':
                    (E_TCA, E_ctpt, E_LW, E_SW, E_NET) = klein_metrics(lnd_obs_C1_dom,
                                                  lnd_C1_dom,Klw_dom,Ksw_dom,WTS_dom)
                else:
                    continue
                KEM_dict[sec][region][sfc]['E_TCA'] = E_TCA
                KEM_dict[sec][region][sfc]['E_ctpt'] = E_ctpt
                KEM_dict[sec][region][sfc]['E_LW'] = E_LW
                KEM_dict[sec][region][sfc]['E_SW'] = E_SW
                KEM_dict[sec][region][sfc]['E_NET'] = E_NET

        C1 = ctl_clisccp_wap[:,:,PP,:]
        obs_C1 = obs_clisccp_AC_wap[:,:,PP,:-1] # ignore the land bin
        WTS = obs_N_AC_wap[:,:-1] # ignore the land bin
        Klw = LWK_wap[:,:,PP,:] # ignore the land bin
        Ksw = SWK_wap[:,:,PP,:]

        # print('Binned data: obs_C1, C1, Klw, Ksw, WTS shapes:', obs_C1.shape,  
        #             C1.shape, Klw.shape, Ksw.shape, WTS.shape)

        (E_TCA, E_ctpt, E_LW, E_SW, E_NET) = klein_metrics(obs_C1[...,:cutoff], 
            C1[...,:cutoff], Klw[...,:cutoff], Ksw[...,:cutoff], WTS[:,:cutoff])
        KEM_dict[sec]['eq30']['ocn_asc']['E_TCA'] = E_TCA
        KEM_dict[sec]['eq30']['ocn_asc']['E_ctpt'] = E_ctpt
        KEM_dict[sec]['eq30']['ocn_asc']['E_LW'] = E_LW
        KEM_dict[sec]['eq30']['ocn_asc']['E_SW'] = E_SW
        KEM_dict[sec]['eq30']['ocn_asc']['E_NET'] = E_NET
        #(E_TCA,E_ctpt,E_LW,E_SW,E_NET) = klein_metrics(obs_C1[...,cutoff:-1],  
        #   C1[...,cutoff:-1], Klw[...,cutoff:-1],Ksw[...,cutoff:-1],WTS[:,cutoff:-1])
        (E_TCA,E_ctpt,E_LW,E_SW,E_NET) = klein_metrics(obs_C1[...,cutoff:],
            C1[...,cutoff:], Klw[...,cutoff:], Ksw[...,cutoff:], WTS[:,cutoff:])
        KEM_dict[sec]['eq30']['ocn_dsc']['E_TCA'] = E_TCA
        KEM_dict[sec]['eq30']['ocn_dsc']['E_ctpt'] = E_ctpt
        KEM_dict[sec]['eq30']['ocn_dsc']['E_LW'] = E_LW
        KEM_dict[sec]['eq30']['ocn_dsc']['E_SW'] = E_SW
        KEM_dict[sec]['eq30']['ocn_dsc']['E_NET'] = E_NET
    # end for sec in sections:
    KEM_dict['metadata'] = {}
    meta = {"date_modified" : str(date.today()),
            "author"        : "Mark D. Zelinka <zelinka1@llnl.gov>",
           }
    KEM_dict['metadata'] = meta

    return(KEM_dict)

def compute_fbk(ctl, fut, DT):
    DR = fut - ctl
    fbk = DR / DT
    baseline = ctl
    return fbk, baseline

def CloudRadKernel(ds_arr):
    print('Get CRK data')
    (ctl_clisccp, fut_clisccp, LWK, SWK, dTs, 
        ctl_clisccp_wap, fut_clisccp_wap, LWK_wap, SWK_wap,
        ctl_N, fut_N) = get_CRK_data_from_isca(ds_arr[0], ds_arr[1])

    area_wts = get_area_wts(ctl_clisccp[:12,0,0,:]) # summing this over lat and lon = 1

    ###########################################################################
    # Compute Klein et al cloud error metrics and their breakdown into components
    ###########################################################################
    print('Computing Klein et al error metrics')
    KEM_dict = do_klein_calcs(ctl_clisccp, LWK, SWK, obs_clisccp_AC, ctl_clisccp_wap,
                                LWK_wap, SWK_wap, obs_clisccp_AC_wap, obs_N_AC_wap)
    # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]

    ###########################################################################
    # Compute cloud feedbacks and their breakdown into components
    ###########################################################################
    print('Computing feedbacks')
    clisccp_fbk, clisccp_base = compute_fbk(ctl_clisccp, fut_clisccp, dTs)
    dummy, LWK_base = compute_fbk(LWK, LWK, dTs)
    dummy, SWK_base = compute_fbk(SWK, SWK, dTs)

    #AX = ctl_clisccp[:12,0,0,:].getAxisList()
    TLL_dims = ctl_clisccp[:12,0,0,:].dims
    TLL_coords = {}
    for dim in TLL_dims:
        TLL_coords[dim] = ctl_clisccp[:12,0,0,:][dim]

    fbk_dict = {}

    # The following performs the amount/altitude/optical depth decomposition of
    # Zelinka et al., J Climate (2012b), as modified in Zelinka et al., J. Climate (2013)
    for sec in sections:
        print('    for section '+sec)
        # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
        fbk_dict[sec] = {}

        PP = sec_dic[sec]

        C1 = clisccp_base[:,:,PP,:]
        C2 = C1 + clisccp_fbk[:,:,PP,:]
        Klw = LWK_base[:,:,PP,:]
        Ksw = SWK_base[:,:,PP,:]

        output1 = KT_decomposition_general(C1, C2, Klw, Ksw)
        #(LWcld_tot,LWcld_amt,LWcld_alt,LWcld_tau,LWcld_err,
        # SWcld_tot,SWcld_amt,SWcld_alt,SWcld_tau,SWcld_err,dc_star,dc_prop)=output1
        fbk_names = ['LWcld_tot', 'LWcld_amt', 'LWcld_alt', 'LWcld_tau', 'LWcld_err',
                        'SWcld_tot', 'SWcld_amt', 'SWcld_alt', 'SWcld_tau', 'SWcld_err']

        # Compute spatial averages over various geographical regions, for ocean, land, and both:
        mx = np.arange(10,101,10) # max latitude of region (i.e., from -mx to mx); last one is for Arctic
        for n, fbk_name in enumerate(fbk_names):
            # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
            fbk_dict[sec][fbk_name] = {}
            data = xr.DataArray(output1[n], dims=TLL_dims, coords=TLL_coords)
            ocn_data, lnd_data = apply_land_mask_v2(data, land_mask_dir=Path(land_mask_dir)) #apply_land_mask_v3(data,OCN,LND)
            lats = data.lat
            for r in mx:
                if r == 100:
                    region = 'Arctic'
                    #domain = cdutil.region.domain(latitude = (70,90))
                    l_lat = (lats >= 70) & (lats <=90)
                else:
                    region = 'eq'+str(r)
                    #domain = cdutil.region.domain(latitude = (-r,r))
                    l_lat = (lats >= -r) & (lats <=r)
                # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
                fbk_dict[sec][fbk_name][region] = {}
                # ocn = np.average(np.nansum(np.nansum(domain.select(ocn_data*area_wts),1),1),0)
                # lnd = np.average(np.nansum(np.nansum(domain.select(lnd_data*area_wts),1),1),0)
                # all = np.average(np.nansum(np.nansum(domain.select(    data*area_wts),1),1),0)
                ocn = np.average(np.nansum((ocn_data*area_wts).where(l_lat, drop=True), axis=(1,2)), 0)
                lnd = np.average(np.nansum((lnd_data*area_wts).where(l_lat, drop=True), axis=(1,2)), 0)
                all = np.average(np.nansum((    data*area_wts).where(l_lat, drop=True), axis=(1,2)), 0)

                # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
                fbk_dict[sec][fbk_name][region]['all'] = all
                fbk_dict[sec][fbk_name][region]['ocn'] = ocn
                fbk_dict[sec][fbk_name][region]['lnd'] = lnd
                fbk_dict[sec][fbk_name][region]['ocn_asc'] = np.nan # these will later be replaced for eq30 only
                fbk_dict[sec][fbk_name][region]['ocn_dsc'] = np.nan # these will later be replaced for eq30 only

        dummy, LWK_wap_base = compute_fbk(LWK_wap,LWK_wap,dTs)
        dummy, SWK_wap_base = compute_fbk(SWK_wap,SWK_wap,dTs)

        Klw = LWK_wap_base[:,:,PP,:] # ignore the land bin
        Ksw = SWK_wap_base[:,:,PP,:]
        C1 = ctl_clisccp_wap[:,:,PP,:]
        C2 = fut_clisccp_wap[:,:,PP,:]
        # N1 = ctl_N[:,:]
        # N2 = fut_N[:,:]
        N1 = ctl_N[:]
        N2 = fut_N[:]

        print('cloud feedback for binned data')
        # no breakdown (this is identical to within + between + covariance)
        a = np.moveaxis(np.array(C1),1,0)
        b = np.moveaxis(a,2,1)              # [TAU,CTP,month,regime]
        # !!!!!! I guess b is already averaged weighted by area, so do no need to so again?
        C1N1 = np.moveaxis(b * np.array(N1), 2, 0)       # [month,TAU,CTP,regime]
        #C1N1 = np.moveaxis(b,2,0)           # [month,TAU,CTP,regime]
        a = np.moveaxis(np.array(C2), 1, 0)
        b = np.moveaxis(a, 2, 1)              # [TAU,CTP,month,regime]
        C2N2 = np.moveaxis(b * np.array(N2), 2, 0)       # [month,TAU,CTP,regime] 
        #C2N2 = np.moveaxis(b, 2, 0)           # [month,TAU,CTP,regime]
        pert, C_base = compute_fbk(C1N1, C2N2, dTs)
        output2 = KT_decomposition_general(C_base, C_base+pert, Klw, Ksw)

        # Put all the ascending and descending region quantities in a dictionary
        for n,fbk_name in enumerate(fbk_names):
            # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
            fbk_dict[sec][fbk_name]['eq30']['ocn_asc'] = np.average(np.nansum((output2[n])[:,:cutoff],1),0)
            fbk_dict[sec][fbk_name]['eq30']['ocn_dsc'] = np.average(np.nansum((output2[n])[:,cutoff:],1),0)
    # end for sec in sections

    fbk_dict['metadata'] = {}
    meta = {
        "date_modified" : str(date.today()),
        "author"        : "Mark D. Zelinka <zelinka1@llnl.gov>",
    }
    fbk_dict['metadata'] = meta

    # [sec][flavor][region][all / ocn / lnd / ocn_asc / ocn_dsc]
    return (fbk_dict, KEM_dict)

if __name__ == '__main__':
    out_dt_dir = Path('../data/zelinka_data/')
    if not out_dt_dir.exists():
        out_dt_dir.mkdir()

    # =================== Read Isca dataset ================== #
    base_dir = Path('/disco/share/ql260/data_isca/')
    ppe_dir = base_dir / 'PPE_extracted_3d/'

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exps_arr = list(exp_tbl.iloc[:, 1])

    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']
    file_clisccp_nms = ['extracted_clisccp_data_301_360.nc', 
                        'extracted_clisccp_data_661_720.nc']

    isca_cld_fbk_dict = {}
    isca_cld_err_dict = {}

    for exp_grp in exp_grps: #[0:2]:
        print('begin:', isca_cld_fbk_dict.keys())
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        ds_clisccp_arr = []
        
        for file_nm, file_clisccp_nm in zip(file_nms, file_clisccp_nms):
            fn = ppe_dir / file_nm.replace('.nc', '_'+exp_grp+'.nc')
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

            fn = ppe_dir / file_clisccp_nm.replace('.nc', '_'+exp_grp+'.nc')
            ds = xr.open_dataset(fn, decode_times=False)
            ds_clisccp_arr.append(ds)

        # Keep the time coordinates the same
        ds_arr[1]['time'] = ds_arr[0].time

        print('Construct clisccp')
        for ds, ds_clisccp in zip(ds_arr, ds_clisccp_arr):
            add_datetime_info(ds)
            #construct_clisccp(ds)
            construct_clisccp_tau_pres(ds_clisccp, ds)
        print('Calc CRE')
        calc_toa_cre_for_isca(ds_arr)

        fbk_dict, err_dict = CloudRadKernel(ds_arr)

        variant = 'r1i1p1f1'
        model = exp_grp
        updated_err_dict = OJ.organize_err_jsons(err_dict, model, variant) 
        updated_fbk_dict = OJ.organize_fbk_jsons(fbk_dict, model, variant)

        isca_cld_fbk_dict[model] = updated_fbk_dict[model]
        isca_cld_err_dict[model] = updated_err_dict[model]
    
        # fn = 'cld_fbk_' + exp_grp +'.json'
        # with open(out_dt_dir / fn, 'w') as outfile: 
        #     json.dump(updated_fbk_dict, outfile)

        # fn = 'cld_err_' + exp_grp +'.json'
        # with open(out_dt_dir / fn, 'w') as outfile:
        #     json.dump(updated_err_dict, outfile)
        print('end:', isca_cld_fbk_dict.keys())

    meta = 'metadata'
    isca_cld_fbk_dict[meta] = updated_fbk_dict[meta]
    isca_cld_err_dict[meta] = updated_err_dict[meta]
    print('final:', isca_cld_fbk_dict.keys())

    fn = 'isca_cld_fbks.json'
    with open(out_dt_dir / fn, 'w') as outfile: 
        # https://stackoverflow.com/questions/53110610
        json.dump(isca_cld_fbk_dict, outfile, separators=(',', ':'))

    fn = 'isca_cld_errs.json'
    with open(out_dt_dir / fn, 'w') as outfile:
        json.dump(isca_cld_err_dict, outfile, separators=(',', ':'))

    print('Done!')
