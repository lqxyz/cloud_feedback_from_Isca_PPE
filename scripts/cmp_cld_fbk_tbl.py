import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca #, calc_total_cwp_for_isca

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

def KT_decomposition_4D(c1, c2, Klw, Ksw):
    # this function takes in a (tau,CTP,lat,lon) matrix and performs the 
    # decomposition of Zelinka et al 2013 doi:10.1175/JCLI-D-12-00555.1

    # reshape to be (CTP,tau,lat,lon)
    # This is inefficient but done purely because Mark can't think in tau,CTP space
    c1  = np.transpose(c1,  axes=(1,0,2,3)) # control cloud fraction histogram
    c2  = np.transpose(c2,  axes=(1,0,2,3)) # perturbed cloud fraction histogram
    Klw = np.transpose(Klw, axes=(1,0,2,3)) # LW Kernel histogram
    Ksw = np.transpose(Ksw, axes=(1,0,2,3)) # SW Kernel histogram

    P = c1.shape[0]
    T = c1.shape[1]

    c = c1
    sum_c = np.tile(np.nansum(c, axis=(0,1)), (P,T,1,1))                       # Eq. B2
    dc = c2 - c1
    sum_dc = np.tile(np.nansum(dc, axis=(0,1)), (P,T,1,1))
    dc_prop = c * (sum_dc / sum_c)
    dc_star = dc - dc_prop                                                  # Eq. B1

    # LW components
    Klw0 = np.tile(np.nansum(Klw * c / sum_c, axis=(0,1)), (P,T,1,1))          # Eq. B4
    Klw_prime = Klw - Klw0                                                  # Eq. B3
    this = np.nansum(Klw_prime * np.tile(np.nansum(c / sum_c, 0), (P,1,1,1)), 1)  # Eq. B7a
    Klw_p_prime = np.tile(np.tile(this, (1,1,1,1)), (T,1,1,1))              # Eq. B7b
    Klw_p_prime = np.transpose(Klw_p_prime, axes=[1,0,2,3])
    
    that = np.tile(np.tile(np.nansum(c / sum_c, 1), (1,1,1,1)), (T,1,1,1))     # Eq. B8a
    that = np.transpose(that, axes=[1,0,2,3])
    Klw_t_prime = np.tile(np.nansum(Klw_prime * that, 0), (P,1,1,1))           # Eq. B8b
    Klw_resid_prime = Klw_prime - Klw_p_prime - Klw_t_prime                 # Eq. B9
    dRlw_true = np.nansum(Klw * dc, axis=(0,1))                                # LW total

    dRlw_prop = Klw0[0,0,:,:] * sum_dc[0,0,:,:]                             # LW amount component
    dRlw_dctp = np.nansum(Klw_p_prime * dc_star, axis=(0,1))                   # LW altitude component
    dRlw_dtau = np.nansum(Klw_t_prime * dc_star, axis=(0,1))                   # LW optical depth component
    dRlw_resid = np.nansum(Klw_resid_prime * dc_star, axis=(0,1))              # LW residual
    dRlw_sum = dRlw_prop + dRlw_dctp + dRlw_dtau + dRlw_resid               # sum of LW components -- should equal LW total

    # SW components
    Ksw0 = np.tile(np.nansum(Ksw * c / sum_c, axis=(0,1)), (P,T,1,1))          # Eq. B4
    Ksw_prime = Ksw - Ksw0                                                  # Eq. B3
    this = np.nansum(Ksw_prime * np.tile(np.nansum(c / sum_c, 0), (P,1,1,1)), 1)  # Eq. B7a 
    Ksw_p_prime = np.tile(np.tile(this, (1,1,1,1)), (T,1,1,1))              # Eq. B7b  
    Ksw_p_prime = np.transpose(Ksw_p_prime, axes=[1,0,2,3])
    that = np.tile(np.tile(np.nansum(c / sum_c, 1), (1,1,1,1)), (T,1,1,1))     # Eq. B8a
    that = np.transpose(that, axes=[1,0,2,3])
    Ksw_t_prime = np.tile(np.nansum(Ksw_prime * that, 0), (P,1,1,1))           # Eq. B8b
    Ksw_resid_prime = Ksw_prime - Ksw_p_prime - Ksw_t_prime                 # Eq. B9
    dRsw_true = np.nansum(Ksw * dc, axis=(0,1))                                # SW total

    dRsw_prop = Ksw0[0,0,:,:] * sum_dc[0,0,:,:]                             # SW amount component
    dRsw_dctp = np.nansum(Ksw_p_prime * dc_star, axis=(0,1))                   # SW altitude component
    dRsw_dtau = np.nansum(Ksw_t_prime * dc_star, axis=(0,1))                   # SW optical depth component
    dRsw_resid = np.nansum(Ksw_resid_prime * dc_star, axis=(0,1))              # SW residual
    dRsw_sum = dRsw_prop + dRsw_dctp + dRsw_dtau + dRsw_resid               # sum of SW components -- should equal SW total

    dc_star = np.transpose(dc_star, (1,0,2,3)) 
    dc_prop = np.transpose(dc_prop, (1,0,2,3)) 

    return (dRlw_true, dRlw_prop, dRlw_dctp, dRlw_dtau, dRlw_resid, 
            dRsw_true, dRsw_prop, dRsw_dctp, dRsw_dtau, dRsw_resid,
            dc_star, dc_prop)

def calc_cld_fbk_with_cld_kernel(ds_ctrl, ds_perturb, kernel_lats, kernel_lons):
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
        avgclisccp1 = clisccp1.groupby('time.month').mean('time') #(12, TAU, CTP, lat_2d, lon_2d)
        avgclisccp2 = clisccp2.groupby('time.month').mean('time') #(12, TAU, CTP, lat_2d, lon_2d)
    except:
        avgclisccp1 = clisccp1.groupby('month').mean('time') #(12, TAU, CTP, lat_2d, lon_2d)
        avgclisccp2 = clisccp2.groupby('month').mean('time') #(12, TAU, CTP, lat_2d, lon_2d)

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

    # Load surface air temperature
    tas1 = ds_ctrl.t_surf #temp_2m
    tas2 = ds_perturb.t_surf #temp_2m

    # Compute climatological annual cycle:
    try:
        avgtas1 = tas1.groupby('time.month').mean('time') #(12, 90, 144)
        avgtas2 = tas2.groupby('time.month').mean('time') #(12, 90, 144)
    except:
        avgtas1 = tas1.groupby('month').mean('time') #(12, 90, 144)
        avgtas2 = tas2.groupby('month').mean('time') #(12, 90, 144)

    # Compute global annual mean tas anomalies
    anomtas = avgtas2 - avgtas1
    coslat1 = np.cos(np.deg2rad(anomtas.lat))
    avgdtas = np.average(anomtas.mean(('month', 'lon')), axis=0, weights=coslat1) # (scalar)
    print('avgdtas =', avgdtas)

    avgalbcs1_grd = avgalbcs1.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avgclisccp1_grd = avgclisccp1.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avgclisccp2_grd = avgclisccp2.interp(lat=kernel_lats, lon=kernel_lons, method="linear")
    avganomclisccp_grd = avganomclisccp.interp(lat=kernel_lats, lon=kernel_lons, method="linear")

    # Use control albcs to map SW kernel to appropriate longitudes
    SWkernel_map = map_SWkern_to_lon(SWkernel, avgalbcs1_grd)

    # Compute clisccp anomalies normalized by global mean delta tas
    anomclisccp = avganomclisccp_grd / avgdtas
    # anomclisccp = avganomclisccp_grd

    # Compute feedbacks: Multiply clisccp anomalies by kernels
    SW0 = SWkernel_map * anomclisccp
    LW_cld_fbk = LWkernel_map * anomclisccp

    # Set the SW cloud feedbacks to zero in the polar night
    # The sun is down if every bin of the SW kernel is zero:
    sundown = np.nansum(SWkernel_map, axis=(1,2))  #12,90,144
    repsundown = np.tile(np.tile(sundown, (1,1,1,1,1)), (7,7,1,1,1))
    repsundown = np.transpose(repsundown, axes=[2,1,0,3,4])

    SW1 = np.where(repsundown==0, 0, SW0)
    SW_cld_fbk = np.where(np.isnan(repsundown), 0, SW1)

    sumLW = np.average(np.nansum(LW_cld_fbk, axis=(1,2)), axis=0)
    coslat = np.cos(np.deg2rad(kernel_lats))
    avgLW_cld_fbk = np.average(np.mean(sumLW, axis=1), axis=0, weights=coslat)
    sumSW = np.average(np.nansum(SW_cld_fbk, axis=(1,2)), axis=0)
    avgSW_cld_fbk = np.average(np.mean(sumSW, axis=1), axis=0, weights=coslat)
    cld_fbk_arr = [avgLW_cld_fbk, avgSW_cld_fbk, avgLW_cld_fbk + avgSW_cld_fbk]

    # Estimated from CRE changes
    cld_fbk_CRE_arr = []
    for v in ['toa_lw_cre', 'toa_sw_cre', 'toa_net_cre']:
        diff_cre = ds_perturb[v] - ds_ctrl[v]
        coslat1 = np.cos(np.deg2rad(diff_cre.lat))
        avg_cld_fbk_CRE = np.average(diff_cre.mean(('time', 'lon')),
                          axis=0, weights=coslat1) / avgdtas
        # print('delta ' + v + '/K', avg_cld_fbk_CRE)
        cld_fbk_CRE_arr.append(avg_cld_fbk_CRE)

    ###########################################################################
    # Part 2: Compute cloud feedbacks and their breakdown into components
    ###########################################################################
    print('Decomposing the cloud feedbacks...')        
    sections_tbl = decomp_cld_fbk(kernel_lats, kernel_lons, LWkernel_map,
                    SWkernel_map, sundown, anomclisccp, avgclisccp1_grd)

    return cld_fbk_CRE_arr, cld_fbk_arr, sections_tbl

def decomp_cld_fbk(kernel_lats, kernel_lons, LWkernel_map,
        SWkernel_map, sundown, anomclisccp, avgclisccp1_grd):
    # Define a python dictionary containing the sections of the histogram to consider
    # These are the same as in Zelinka et al, GRL, 2016
    sections = ['ALL', 'HI680', 'LO680']
    Psections = [slice(0,7), slice(2,7), slice(0,2)]
    sec_dic = dict(zip(sections, Psections))

    nmons = 12
    nlats = len(kernel_lats)
    nlons = len(kernel_lons)
    night = np.where(sundown == 0)

    coslat = np.cos(np.deg2rad(kernel_lats))

    lw_nms = ['LWcld_tot', 'LWcld_amt', 'LWcld_alt' ,'LWcld_tau', 'LWcld_err']
    sw_nms = ['SWcld_tot', 'SWcld_amt', 'SWcld_alt' ,'SWcld_tau', 'SWcld_err']
    lw_sw_nms = lw_nms + sw_nms

    #sections_dt_dict = {}
    sections_tbl = np.zeros((len(sections), len(lw_sw_nms)))
    for kk, sec in enumerate(sections):
        print('Using ' + sec + ' CTP bins')
        choose = sec_dic[sec]
        LC = len(np.ones(100)[choose])

        # Preallocation of arrays:
        LWcld_tot = np.ones((nmons, nlats, nlons)) * np.nan
        LWcld_amt = np.ones((nmons, nlats, nlons)) * np.nan
        LWcld_alt = np.ones((nmons, nlats, nlons)) * np.nan
        LWcld_tau = np.ones((nmons, nlats, nlons)) * np.nan
        LWcld_err = np.ones((nmons, nlats, nlons)) * np.nan
        SWcld_tot = np.ones((nmons, nlats, nlons)) * np.nan
        SWcld_amt = np.ones((nmons, nlats, nlons)) * np.nan
        SWcld_alt = np.ones((nmons, nlats, nlons)) * np.nan
        SWcld_tau = np.ones((nmons, nlats, nlons)) * np.nan
        SWcld_err = np.ones((nmons, nlats, nlons)) * np.nan
        dc_star = np.ones((nmons, 7, LC, nlats, nlons)) * np.nan
        dc_prop = np.ones((nmons, 7, LC, nlats, nlons)) * np.nan

        for mm in np.arange(nmons):
            dcld_dT = anomclisccp[mm,:,choose,:]

            c1 = avgclisccp1_grd[mm,:,choose,:]
            c2 = c1 + dcld_dT
            Klw = LWkernel_map[mm,:,choose,:]
            Ksw = SWkernel_map[mm,:,choose,:]

            # The following performs the amount/altitude/optical depth decomposition of
            # Zelinka et al., J Climate (2012b), as modified in Zelinka et al., J. Climate (2013)
            (LWcld_tot[mm,:], LWcld_amt[mm,:], LWcld_alt[mm,:], LWcld_tau[mm,:], LWcld_err[mm,:],
            SWcld_tot[mm,:], SWcld_amt[mm,:], SWcld_alt[mm,:], SWcld_tau[mm,:], SWcld_err[mm,:],
            dc_star[mm,:], dc_prop[mm,:]) = KT_decomposition_4D(c1.values, c2.values, Klw, Ksw)

        # Set the SW cloud feedbacks to zero in the polar night
        # Do this since they may come out of previous calcs as undefined, but should be zero:
        SWcld_tot[night] = 0
        SWcld_amt[night] = 0
        SWcld_alt[night] = 0
        SWcld_tau[night] = 0
        SWcld_err[night] = 0

        # Sanity check: print global and annual mean cloud feedback components
        #AX = avgalbcs1_grd[0,:].getAxisList()         
        # Plot Maps
        # lon_2d, lat_2d = np.meshgrid(kernel_lons, kernel_lats)
        '''
        data_dict = {'LW': {'names': ['LWcld_tot', 'LWcld_amt', 'LWcld_alt' ,'LWcld_tau', 'LWcld_err'], 
                            'var_arr': [LWcld_tot, LWcld_amt, LWcld_alt, LWcld_tau, LWcld_err]},
                    'SW': {'names': ['SWcld_tot', 'SWcld_amt', 'SWcld_alt' ,'SWcld_tau', 'SWcld_err'],
                        'var_arr': [SWcld_tot, SWcld_amt, SWcld_alt, SWcld_tau, SWcld_err]} }
        data_gm_dict = {}
        for key, val in data_dict.items():
            names = val['names']
            var_arr = val['var_arr']
            for n, name in enumerate(names):
                print(key, name)
                dt = np.ma.masked_array(var_arr[n], np.isnan(var_arr[n]))
                DATA = np.ma.average(dt, axis=0)
                avgDATA = np.ma.average(np.ma.average(DATA, axis=1), axis=0, weights=coslat)
                data_gm_dict[name] = avgDATA 
        sections_dt_dict[sec] = data_gm_dict
        '''
        lw_vals = [LWcld_tot, LWcld_amt, LWcld_alt, LWcld_tau, LWcld_err]
        sw_vals = [SWcld_tot, SWcld_amt, SWcld_alt, SWcld_tau, SWcld_err]
        lw_sw_vals = lw_vals + sw_vals
        lw_sw_gm_vals = []
        for val in lw_sw_vals:
            dt = np.ma.masked_array(val, np.isnan(val))
            dt_tm = np.ma.average(dt, axis=0)
            avgval = np.ma.average(np.ma.average(dt_tm, axis=1), axis=0, weights=coslat)
            lw_sw_gm_vals.append(avgval)
        print(sec, lw_sw_gm_vals)
        sections_tbl[kk, :] = np.array(lw_sw_gm_vals)

    sections_tbl = pd.DataFrame(data=sections_tbl, index=sections, columns=lw_sw_nms)
    # print(sections_tbl)
    return sections_tbl

if __name__ == '__main__':
    P = os.path.join

    dt_dir = '../data'
    # Read kernel data
    kernel_dt_dir = '../inputs'
    # Load in the Zelinka et al 2012 kernels:
    #kernel_nm = P(kernel_dt_dir, 'obs_cloud_kernels3.nc')
    kernel_nm = P(kernel_dt_dir, 'cloud_kernels2.nc')
    f = xr.open_dataset(kernel_nm, decode_times=False)
    LWkernel = f.LWkernel
    SWkernel = f.SWkernel
    LWkernel = xr.where(np.isnan(LWkernel), np.nan, LWkernel)
    SWkernel = xr.where(np.isnan(SWkernel), np.nan, SWkernel)
    print('LWkernel', LWkernel.dims, LWkernel.shape)

    # the clear-sky albedos over which the kernel is computed
    albcs = np.arange(0.0, 1.5, 0.5)
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

    print('Construct clisccp')
    for ds in ds_arr:
        add_datetime_info(ds)
        construct_clisccp(ds)

    calc_toa_cre_for_isca(ds_arr)

    print('Begin kernel calculation...')
    time_arr = np.arange(21*12, 30*12, 1)
    ds1 = ds_arr[0].isel(time=time_arr)
    ds_arr[1]['time'] = ds_arr[0].time
    ds2 = ds_arr[1].isel(time=time_arr)
    delta_cre_per_K, cld_fbk_kernel, sections_tbl = \
        calc_cld_fbk_with_cld_kernel(ds1, ds2, kernel_lats, kernel_lons)
    '''
    ## Save feedback data to tables
    table = np.zeros((2,3))
    table[0,:] = delta_cre_per_K
    table[1,:] = cld_fbk_kernel
    row_names = ['Delta CRE per K', 'Fbk cld kernel']
    col_names = ['LW', 'SW', 'Net']
    tbl = pd.DataFrame(data=table, index=row_names, columns=col_names)
    if 'kernels3' in kernel_nm:
        file_name = P(dt_dir, 'cld_fbk_from_CRE_and_kernel.csv')
    else:
        file_name = P(dt_dir, 'cld_fbk_from_CRE_and_kernel_v2.csv')
    tbl.to_csv(file_name, header=True, index=True, float_format="%.3f")
    print('Annual mean csv file saved: ', file_name)

    if 'kernels3' in kernel_nm:
        file_name = P(dt_dir, 'cld_fbk_decomp.csv')
    else:
        file_name = P(dt_dir, 'cld_fbk_decomp_v2.csv')
    sections_tbl.to_csv(file_name, header=True, index=True, float_format="%.5f")
    print('Annual mean csv file saved: ', file_name)
    '''

    #============== 10-yr data ===========#
    nyr = 10
    lw_fbk = np.zeros(nyr)
    sw_fbk = np.zeros(nyr)
    net_fbk = np.zeros(nyr)
    
    lw_fbk_CRE = np.zeros(nyr)
    sw_fbk_CRE = np.zeros(nyr)
    net_fbk_CRE = np.zeros(nyr)

    for n in range(nyr):
        nn = n + 20
        print('year: ', nn)
        ds1 = ds_arr[0].isel(time=np.arange(nn*12, (nn+1)*12, 1))
        ds2 = ds_arr[1].isel(time=np.arange(nn*12, (nn+1)*12, 1))
        print(ds1.time.values)
        delta_cre_per_K, cld_fbk_kernel, sections_tbl = calc_cld_fbk_with_cld_kernel(
                                         ds1, ds2, kernel_lats, kernel_lons)
        lw_fbk_CRE[n] = delta_cre_per_K[0]
        sw_fbk_CRE[n] = delta_cre_per_K[1]
        net_fbk_CRE[n] = delta_cre_per_K[2]
        print(' cld fbk from CRE is ', delta_cre_per_K)

        lw_fbk[n] = cld_fbk_kernel[0]
        sw_fbk[n] = cld_fbk_kernel[1]
        net_fbk[n] = cld_fbk_kernel[2]
        print(' cld fbk from kernel is ', cld_fbk_kernel)

    print('Saving data...')
    time_arr = np.arange(20*12, 30*12, 1)
    try:
        tsurf = ds_arr[0].t_surf.isel(time=time_arr).groupby('year').mean()
    except:
        tsurf = ds_arr[0].t_surf.isel(time=time_arr).groupby('time.year').mean()
    '''
    fbk_dict = {}
    dims = 'year'
    years = tsurf[dims]

    fbk_dict['lw_cld_fbk_CRE'] = (dims, lw_fbk_CRE)
    fbk_dict['sw_cld_fbk_CRE'] = (dims, sw_fbk_CRE)
    fbk_dict['net_cld_fbk_CRE'] = (dims, net_fbk_CRE)

    fbk_dict['lw_cld_fbk_kernel'] = (dims, lw_fbk)
    fbk_dict['sw_cld_fbk_kernel'] = (dims, sw_fbk)
    fbk_dict['net_cld_fbk_kernel'] = (dims, net_fbk)

    ds_save = xr.Dataset(fbk_dict, coords={dims:years})
    if 'kernels3' in kernel_nm:
        fn_save = P(dt_dir, 'cld_fbk_from_CRE_and_kernel.nc')
    else: # kernels2
        fn_save = P(dt_dir, 'cld_fbk_from_CRE_and_kernel_v2.nc')
    ds_save.to_netcdf(fn_save, mode='w', format='NETCDF3_CLASSIC')
    print(fn_save + ' saved.')
    '''

    ## Save feedback data to tables
    table = np.zeros((2,3))
    table[0,:] =  [lw_fbk_CRE.mean(), sw_fbk_CRE.mean(), net_fbk_CRE.mean()]
    table[1,:] = [lw_fbk.mean(), sw_fbk.mean(), net_fbk.mean()]
    row_names = ['Delta CRE per K', 'Fbk cld kernel']
    col_names = ['LW', 'SW', 'Net']
    tbl = pd.DataFrame(data=table, index=row_names, columns=col_names)
    if 'kernels3' in kernel_nm:
        file_name = P(dt_dir, 'cld_fbk_from_CRE_and_kernel_10yr.csv')
    else:
        file_name = P(dt_dir, 'cld_fbk_from_CRE_and_kernel_10yr_v2.csv')
    tbl.to_csv(file_name, header=True, index=True, float_format="%.3f")

    # Compare the cloud feedbacks from different diagnostics and different methods according to Zelnka et al 2013
    #£nyr = 30
    for nyr in [20, 30]:
        table = np.zeros((2,6))
        # Read slope data
        slope_fn = P(dt_dir, 'slope_of_delta_CRE_and_R_'+str(nyr)+'yr.csv')
        slope_tbl = pd.read_csv(slope_fn, index_col=0)

        table[0,:] =  [lw_fbk_CRE.mean(), sw_fbk_CRE.mean(), net_fbk_CRE.mean(), 
                       slope_tbl.loc['Delta CRE (Slope)']['LW'],
                       slope_tbl.loc['Delta CRE (Slope)']['SW'],
                       slope_tbl.loc['Delta CRE (Slope)']['Net'],
                       ]
        table[1,:] = [lw_fbk.mean(), sw_fbk.mean(), net_fbk.mean(),
                      slope_tbl.loc['Delta R (Slope)']['LW'],
                      slope_tbl.loc['Delta R (Slope)']['SW'],
                      slope_tbl.loc['Delta R (Slope)']['Net'],
                ]
        row_names = ['CRE anomalies', 'Kerenel-derived anomalies']
        col_names = ['LW', 'SW', 'Net', 'LW', 'SW', 'Net']
        tbl = pd.DataFrame(data=table, index=row_names, columns=col_names)
        if 'kernels3' in kernel_nm:
            file_name = P(dt_dir, 'cmp_cld_fbk_nyr_'+str(nyr)+'.csv')
        else:
            file_name = P(dt_dir, 'cmp_cld_fbk_v2_nyr_'+str(nyr)+'.csv')
        tbl.to_csv(file_name, header=True, index=True, float_format="%.3f")

        tex_fn = file_name.replace('.csv', '.tex') 
        tbl.to_latex(tex_fn, float_format="%.2f")
        print('Done.', tex_fn, 'saved')

