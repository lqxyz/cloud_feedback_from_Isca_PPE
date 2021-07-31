from __future__ import print_function
import numpy as np
from scipy.integrate import trapz
import pandas

def calc_toa_cre_for_isca(ds_arr):
    """
    Diagnose the cloud radiative effect (CRE) at TOA for Isca ouputs
    with Socrates radiation scheme
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    for ds in ds_arr1:
        dims = ds.soc_toa_sw.dims
        # cloud radiative effect
        try:
            soc_cre_toa_sw = ds.soc_toa_swup_clr - ds.soc_toa_swup
        except:
            soc_cre_toa_sw = ds.soc_toa_sw_up_clr - ds.soc_toa_sw_up
        ds['toa_sw_cre'] = (dims, soc_cre_toa_sw)

        soc_cre_toa_lw = ds.soc_olr_clr - ds.soc_olr
        ds['toa_lw_cre'] = (dims, soc_cre_toa_lw)

        soc_cre_toa_total =  soc_cre_toa_sw + soc_cre_toa_lw
        ds['toa_net_cre'] = (dims, soc_cre_toa_total)

        '''
        for key in ds.variables.keys():
            if 'cre' in key and '_gm' not in key:
                coslat = np.cos(np.deg2rad(ds.lat))
                ds[key+'_gm'] = np.average(ds[key].mean(('time','lon')), axis=0, weights=coslat)
        '''

def calc_total_cwp_for_isca(ds_arr):
    """
    Calculate total cloud water path for Isca dataset
    Units: kg/m^2
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    
    grav = 9.87
    for ds in ds_arr1:
        mixing_ratio = ds.qcl_rad / (1.0 + ds.qcl_rad)
        dims_4d = ds.qcl_rad.dims
        ds['rcl_rad'] = (dims_4d, mixing_ratio)
        #ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        #lwp = ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2
        dst_dt = ds.rcl_rad
        ma_dt = np.ma.MaskedArray(dst_dt, mask=np.isnan(dst_dt))
        ### Not sure why the cwp is negative if I dont add '-' in front of it.
        cwp = -trapz(ma_dt, x=ds.pfull, axis=1) * 1e2 / grav # kg/m^2
        if np.mean(cwp)<0:
            cwp = -cwp

        # cwp = trapz(ds.rcl_rad, x=ds.pfull, axis=1) * 1e2 / grav # kg/m^2
        dims_3d = list(dims_4d)
        dims_3d.remove('pfull')
        ds['cwp'] = (tuple(dims_3d), cwp)

        '''
        for key in ds.variables.keys():
            if 'cwp' in key and '_gm' not in key:
                coslat = np.cos(np.deg2rad(ds.lat))
                dst_dt = ds[key]
                ma = np.ma.MaskedArray(dst_dt, mask=np.isnan(dst_dt))
                ds[key+'_gm'] = np.ma.average(np.ma.mean(ma, axis=(0,2)), axis=0, weights=coslat) 
                #ds[key+'_gm'] = np.average(ds[key].mean(('time','lon')), axis=0, weights=coslat)
        ds['lwp_gm'] = ds['cwp_gm']
        '''
        #ds['lwp'] = ds['cwp'] #(tuple(dims_3d), cwp)


def calc_low_mid_high_cwp_for_isca2(ds_arr, p_mid=660, p_high=440):
    """
    Calculate total cloud water path for Isca dataset
    Units: kg/m^2
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    
    grav = 9.87
    for ds in ds_arr1:
        mixing_ratio = ds.qcl_rad / (1.0 + ds.qcl_rad)
        dims_4d = ds.qcl_rad.dims
        ds['rcl_rad'] = (dims_4d, mixing_ratio)
        #ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        #lwp = ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2

        var_nms = ['cwp', 'low_cwp', 'mid_cwp', 'high_cwp']
        for var_nm in var_nms:
            if 'low' in var_nm:
                l_pres = ds.pfull >= p_mid
            elif 'mid' in var_nm:
                l_pres = (ds.pfull<p_mid) & (ds.pfull>p_high)
            elif 'high' in var_nm:
                l_pres = ds.pfull <= p_high
            else:
                l_pres = ds.pfull > 0.0

            dst_dt = ds.rcl_rad.where(l_pres, drop=True)

            ma_dt = np.ma.MaskedArray(dst_dt, mask=np.isnan(dst_dt))
            ### Not sure why the cwp is negative if I dont add '-' in front of it.
            cwp = -trapz(ma_dt, x=dst_dt.pfull, axis=1) * 1e2 / grav # kg/m^2
            if np.mean(cwp) < 0:
                cwp = -cwp

            # cwp = trapz(ds.rcl_rad, x=ds.pfull, axis=1) * 1e2 / grav # kg/m^2
            dims_3d = list(dims_4d)
            dims_3d.remove('pfull')
            ds[var_nm] = (tuple(dims_3d), cwp)
            print(var_nm, np.mean(cwp)*1e3)

def calc_low_mid_high_cwp_for_isca(ds_arr, p_mid=660, p_high=440,
        var_nms=['cwp', 'low_cwp', 'mid_cwp', 'high_cwp']):
    """
    Calculate total cloud water path for Isca dataset
    Units: kg/m^2
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    
    grav = 9.87
    for ds in ds_arr1:
        mixing_ratio = ds.qcl_rad / (1.0 + ds.qcl_rad)
        dims_4d = ds.qcl_rad.dims
        ds['rcl_rad'] = (dims_4d, mixing_ratio)
        #ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        #lwp = ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2

        #var_nms = ['cwp', 'low_cwp', 'mid_cwp', 'high_cwp']
        for var_nm in var_nms:
            if 'low' in var_nm:
                l_pres = ds.pfull >= p_mid
            elif 'mid' in var_nm:
                l_pres = (ds.pfull<p_mid) & (ds.pfull>p_high)
            elif 'high' in var_nm:
                l_pres = ds.pfull <= p_high
            else:
                l_pres = ds.pfull > 0.0

            dst_dt = ds.rcl_rad.where(l_pres, drop=True)

            ma_dt = np.ma.MaskedArray(dst_dt, mask=np.isnan(dst_dt))
            ### Not sure why the cwp is negative if I dont add '-' in front of it.
            cwp = -trapz(ma_dt, x=dst_dt.pfull, axis=1) * 1e2 / grav # kg/m^2
            if np.mean(cwp) < 0:
                cwp = -cwp

            # cwp = trapz(ds.rcl_rad, x=ds.pfull, axis=1) * 1e2 / grav # kg/m^2
            dims_3d = list(dims_4d)
            dims_3d.remove('pfull')
            ds[var_nm] = (tuple(dims_3d), cwp)
            print(var_nm, np.mean(cwp)*1e3)

def calc_lwp_for_isca(ds_arr):
    """
    Calculate total cloud water path for Isca dataset
    Units: kg/m^2
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    
    grav = 9.87
    for ds in ds_arr1:
        mixing_ratio = ds.qcl_rad / (1.0 + ds.qcl_rad)
        dims_4d = ds.qcl_rad.dims
        ds['rcl_rad'] = (dims_4d, mixing_ratio)
        #ds['dpfull'] = (ds.pfull.dims, np.diff(ds.phalf))
        #lwp = ds.rcl_rad * ds.dpfull * 1e2 / grav # kg/m^2
        lwp = trapz(ds.rcl_rad, x=ds.pfull, axis=1) * 1e2 / grav # kg/m^2
        dims_3d = list(dims_4d)
        dims_3d.remove('pfull')
        
        ds['lwp'] = (tuple(dims_3d), lwp)

        '''
        for key in ds.variables.keys():
            if 'lwp' in key and '_gm' not in key:
                coslat = np.cos(np.deg2rad(ds.lat))
                ds[key+'_gm'] = np.average(ds[key].mean(('time','lon')), axis=0, weights=coslat)
        '''

def calc_surf_cre_for_isca(ds_arr):
    """
    Diagnose the cloud radiative effect (CRE) at surface for Isca ouputs
    with Socrates radiation scheme
    """
    if type(ds_arr)!=list and 'Dataset' in str(type(ds_arr)):
        ds_arr1 = [ds_arr]
    else:
        ds_arr1 = ds_arr
    for ds in ds_arr1:
        dims = ds.soc_surf_flux_sw.dims
        # cloud radiative effect
        surf_sw_cre = ds.soc_surf_flux_sw - ds.soc_surf_flux_sw_clr
        ds['surf_sw_cre'] = (dims, surf_sw_cre)
        surf_lw_cre = ds.soc_surf_flux_lw_clr - ds.soc_surf_flux_lw #As up is positive
        ds['surf_lw_cre'] = (dims, surf_lw_cre)
        surf_net_cre = surf_sw_cre + surf_lw_cre
        ds['surf_net_cre'] = (dims, surf_net_cre)

        '''
        for key in ds.variables.keys():
            if 'cre' in key and '_gm' not in key:
                coslat = np.cos(np.deg2rad(ds.lat))
                ds[key+'_gm'] = np.average(ds[key].mean(('time','lon')), axis=0, weights=coslat)
        '''

def add_toa_net_flux_to_ds_arr(ds_arr):
    """TOA net flux, net sw - olr"""
    var_name = 'soc_toa_net_flux'
    for ds in ds_arr:
        try:
            var_dims = ds.soc_olr.dims
            var_val = ds.soc_toa_sw - ds.soc_olr
        except:
            soc_olr = ds.soc_flux_lw.sel(phalf=0)
            var_dims = soc_olr.dims
            var_val = -ds.soc_flux_sw.sel(phalf=0) - soc_olr
        #add_var_to_ds_arr(ds_arr, var_name, var_val, var_dims)
        ds[var_name] = (var_dims, var_val)

def get_gm(var, ax=0):
    coslat = np.cos(np.deg2rad(var.lat))
    var_m = var.mean(('time', 'lon'), skipna=True)
    var_mask = np.ma.masked_array(var_m, mask=np.isnan(var_m))
    return np.ma.average(var_mask, axis=ax, weights=coslat)


def print_flux_table(ds_arr, col_names, file_name=None, float_fmt='%.2f'):
    """Print LW and SW flux at TOA and surface
    Table is like this:
                      obs, ds1, ds2, ds3...
    TOA upwelling SW: xx   xx   xx  xx
    TOA net SW:
    TOA outgoing LW:
    TOA net flux:  

    Surf downwelling SW:
    Surf upwelling SW:
    Surf net SW:

    Surf downwelling LW:
    Surf upwelling LW:
    Surf net LW:
    
    Surf net flux:
    """

    row_names = ['TOA upwelling SW', 'TOA net SW', 'TOA outgoing LW', 'TOA net flux',
                 "Surf downwelling SW", "Surf upwelling SW", "Surf net SW",
                 "Surf downwelling LW", "Surf upwelling LW", "Surf net LW",
                 "Surf net flux" ]

    N = len(ds_arr)
    flux_table =  np.zeros((len(row_names), N), dtype='double')
    for j, ds in enumerate(ds_arr):
        # ---------- TOA -------------- #
        try:
            toa_sw_up = get_gm(ds.soc_toa_swup)
        except:
            toa_sw_up = get_gm(ds.soc_toa_sw_up)
        toa_net_sw = get_gm(ds.soc_toa_sw)
        olr = get_gm(ds.soc_olr)
        toa_net_flux = toa_net_sw - olr
        
        # ---------- Surface ----------- #
        surf_sw_dn = get_gm(ds.soc_surf_flux_sw_down)
        surf_net_sw = get_gm(ds.soc_surf_flux_sw)
        surf_sw_up = surf_sw_dn - surf_net_sw

        surf_lw_dn = get_gm(ds.soc_surf_flux_lw_down)
        surf_net_lw = get_gm(-ds.soc_surf_flux_lw) # change to downward is positive...
        surf_lw_up = surf_lw_dn - surf_net_lw

        surf_net_flux = surf_net_sw + surf_net_lw
    
        
        for i, dt in enumerate([toa_sw_up, toa_net_sw, olr, toa_net_flux,
                                surf_sw_dn, surf_sw_up, surf_net_sw,
                                surf_lw_dn, surf_lw_up, surf_net_lw,
                                surf_net_flux ]):
            flux_table[i, j] = dt

    pd = pandas.DataFrame(data=flux_table, index=row_names, columns=col_names)
        
    if file_name is None:
        print(pd.to_latex(float_format=float_fmt))
    else:
        pd.to_latex(buf=file_name, float_format=float_fmt)


def print_flux_diff_table(ds, obs_dict, ds_name, file_name=None, with_ratio=True, float_fmt='%.2f'):
    """Print LW and SW flux at TOA and surface"""

    row_names = ['TOA upwelling SW', 'TOA net SW', 'TOA outgoing LW', 'TOA net flux',
                 "Surf downwelling SW", "Surf upwelling SW", "Surf net SW",
                 "Surf downwelling LW", "Surf upwelling LW", "Surf net LW",
                 "Surf net flux" ]

    if with_ratio:
        N = 4
        col_names = ['Obs.', ds_name, ds_name+' - obs', ds_name+' - obs (%)']
    else:
        N = 3
        col_names = ['Obs.', ds_name, ds_name+' - obs']
    flux_table =  np.zeros((len(row_names), N), dtype='double')

    # =============== Obs results ================ #
    var_names = ['toa_sw_up', 'toa_net_sw', 'olr', 'toa_net_flux',
                'surf_sw_down', 'surf_sw_up', 'surf_net_sw',
                'surf_lw_down', 'surf_lw_up', 'surf_net_lw',
                'surf_net_flux' ]
    obs_arr = []
    for i, key in enumerate(var_names):
        val = get_gm(obs_dict[key])
        obs_arr.append(val)
        flux_table[i, 0] = val

    # =============== Model results ================ #
    # ---------- TOA -------------- #
    try:
        toa_sw_up = get_gm(ds.soc_toa_swup)
    except:
        toa_sw_up = get_gm(ds.soc_toa_sw_up)
    toa_net_sw = get_gm(ds.soc_toa_sw)
    olr = get_gm(ds.soc_olr)
    toa_net_flux = toa_net_sw - olr
    
    # ---------- Surface ----------- #
    surf_sw_dn = get_gm(ds.soc_surf_flux_sw_down)
    surf_net_sw = get_gm(ds.soc_surf_flux_sw)
    surf_sw_up = surf_sw_dn - surf_net_sw

    surf_lw_dn = get_gm(ds.soc_surf_flux_lw_down)
    # The net_lw is up as default
    # # change to downward is positive...
    surf_net_lw = get_gm(-ds.soc_surf_flux_lw)
    surf_lw_up = surf_lw_dn - surf_net_lw

    surf_net_flux = surf_net_sw + surf_net_lw
    
    mod_arr = [toa_sw_up, toa_net_sw, olr, toa_net_flux,
                surf_sw_dn, surf_sw_up, surf_net_sw,
                surf_lw_dn, surf_lw_up, surf_net_lw,
                surf_net_flux ]

    for i, dt in enumerate(mod_arr):
        flux_table[i, 1] = dt

    for i, (obs_dt, mod_dt) in enumerate(zip(obs_arr, mod_arr)):
        diff_dt = mod_dt - obs_dt
        flux_table[i, 2] = diff_dt
        if with_ratio:
            flux_table[i, 3] = diff_dt / obs_dt * 1e2

    pd = pandas.DataFrame(data=flux_table, index=row_names, columns=col_names)
    if file_name is None:
        print(pd.to_latex(float_format=float_fmt))
    else:
        pd.to_latex(buf=file_name, float_format=float_fmt)


def print_flux_diff_table_two_exps(ds_arr, ds_names, obs_dict, file_name=None, float_fmt='%.2f'):
    """Print LW and SW flux at TOA and surface"""

    row_names = ['TOA upwelling SW', 'TOA net SW', 'TOA outgoing LW', 'TOA net flux',
                 "Surf downwelling SW", "Surf upwelling SW", "Surf net SW",
                 "Surf downwelling LW", "Surf upwelling LW", "Surf net LW",
                 "Surf net flux" ]
    N = 6
    col_names = ['Obs.', ds_names[0], ds_names[1], ds_names[0]+' - obs', 
                ds_names[1]+' - obs', ds_names[1]+' - '+ds_names[0]]
    flux_table =  np.zeros((len(row_names), N), dtype='double')

    # =============== Obs results ================ #
    var_names = ['toa_sw_up', 'toa_net_sw', 'olr', 'toa_net_flux',
                'surf_sw_down', 'surf_sw_up', 'surf_net_sw',
                'surf_lw_down', 'surf_lw_up', 'surf_net_lw',
                'surf_net_flux' ]
    obs_arr = []
    for i, key in enumerate(var_names):
        val = get_gm(obs_dict[key])
        obs_arr.append(val)
        flux_table[i, 0] = val

    # =============== Model results ================ #
    def get_model_flux_arr(ds):
        # ---------- TOA -------------- #
        try:
            toa_sw_up = get_gm(ds.soc_toa_swup)
        except:
            toa_sw_up = get_gm(ds.soc_toa_sw_up)
        toa_net_sw = get_gm(ds.soc_toa_sw)
        olr = get_gm(ds.soc_olr)
        toa_net_flux = toa_net_sw - olr
        
        # ---------- Surface ----------- #
        surf_sw_dn = get_gm(ds.soc_surf_flux_sw_down)
        surf_net_sw = get_gm(ds.soc_surf_flux_sw)
        surf_sw_up = surf_sw_dn - surf_net_sw

        surf_lw_dn = get_gm(ds.soc_surf_flux_lw_down)
        # The net_lw is up as default
        # # change to downward is positive...
        surf_net_lw = get_gm(-ds.soc_surf_flux_lw)
        surf_lw_up = surf_lw_dn - surf_net_lw

        surf_net_flux = surf_net_sw + surf_net_lw
        
        mod_arr = [toa_sw_up, toa_net_sw, olr, toa_net_flux,
                    surf_sw_dn, surf_sw_up, surf_net_sw,
                    surf_lw_dn, surf_lw_up, surf_net_lw,
                    surf_net_flux ]
        return mod_arr

    mod_dt_arrs = []
    for nn, ds in enumerate(ds_arr):
        mod_arr = get_model_flux_arr(ds)
        mod_dt_arrs.append(mod_arr)
        for i, dt in enumerate(mod_arr):
            flux_table[i, nn+1] = dt

    # difference with observation
    for nn, mod_arr in enumerate(mod_dt_arrs):
        for i, (obs_dt, mod_dt) in enumerate(zip(obs_arr, mod_arr)):
            diff_dt = mod_dt - obs_dt
            flux_table[i, nn+3] = diff_dt
            #if with_ratio:
            #    flux_table[i, 3] = diff_dt / obs_dt * 1e2
    
    # model difference
    for i, (mod_dt1, mod_dt2) in enumerate(zip(mod_dt_arrs[0],  mod_dt_arrs[1])):
        diff_dt = mod_dt2 - mod_dt1
        flux_table[i, 5] = diff_dt

    pd = pandas.DataFrame(data=flux_table, index=row_names, columns=col_names)
    if file_name is None:
        print(pd.to_latex(float_format=float_fmt))
    else:
        pd.to_latex(buf=file_name, float_format=float_fmt)


def print_flux_table_in_exps(ds_arr, ds_names, obs_dict, file_name=None, float_fmt='%.2f'):
    row_names = ['TOA upwelling SW', 'TOA net SW', 'TOA outgoing LW', 'TOA net flux',
                 "Surf downwelling SW", "Surf upwelling SW", "Surf net SW",
                 "Surf downwelling LW", "Surf upwelling LW", "Surf net LW",
                 "Surf net flux" ]
    N = len(ds_arr) + 1
    col_names = ['Obs.']
    for ds_nm in ds_names:
        col_names.append(ds_nm)
    
    flux_table =  np.zeros((len(row_names), N), dtype='double')

    # =============== Obs results ================ #
    var_names = ['toa_sw_up', 'toa_net_sw', 'olr', 'toa_net_flux',
                'surf_sw_down', 'surf_sw_up', 'surf_net_sw',
                'surf_lw_down', 'surf_lw_up', 'surf_net_lw',
                'surf_net_flux' ]
    obs_arr = []
    for i, key in enumerate(var_names):
        val = get_gm(obs_dict[key])
        obs_arr.append(val)
        flux_table[i, 0] = val

    # =============== Model results ================ #
    def get_model_flux_arr(ds):
        # ---------- TOA -------------- #
        try:
            toa_sw_up = get_gm(ds.soc_toa_swup)
        except:
            toa_sw_up = get_gm(ds.soc_toa_sw_up)
        toa_net_sw = get_gm(ds.soc_toa_sw)
        olr = get_gm(ds.soc_olr)
        toa_net_flux = toa_net_sw - olr
        
        # ---------- Surface ----------- #
        surf_sw_dn = get_gm(ds.soc_surf_flux_sw_down)
        surf_net_sw = get_gm(ds.soc_surf_flux_sw)
        surf_sw_up = surf_sw_dn - surf_net_sw

        surf_lw_dn = get_gm(ds.soc_surf_flux_lw_down)
        # The net_lw is up as default
        # # change to downward is positive...
        surf_net_lw = get_gm(-ds.soc_surf_flux_lw)
        surf_lw_up = surf_lw_dn - surf_net_lw

        surf_net_flux = surf_net_sw + surf_net_lw
        
        mod_arr = [toa_sw_up, toa_net_sw, olr, toa_net_flux,
                    surf_sw_dn, surf_sw_up, surf_net_sw,
                    surf_lw_dn, surf_lw_up, surf_net_lw,
                    surf_net_flux ]
        return mod_arr

    mod_dt_arrs = []
    for nn, ds in enumerate(ds_arr):
        mod_arr = get_model_flux_arr(ds)
        mod_dt_arrs.append(mod_arr)
        for i, dt in enumerate(mod_arr):
            flux_table[i, nn+1] = dt

    pd = pandas.DataFrame(data=flux_table, index=row_names, columns=col_names)
    if file_name is None:
        print(pd.to_latex(float_format=float_fmt))
    else:
        pd.to_latex(buf=file_name, float_format=float_fmt)

