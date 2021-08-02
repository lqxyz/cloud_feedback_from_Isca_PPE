from __future__ import print_function
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import proplot as plot
import warnings
warnings.simplefilter(action='ignore')
import metpy.calc
from metpy.units import units
import model_constants as mc
from calc_Tadv import calc_Tadv
from analysis_functions import detrend_dim

def print_python_and_package_info():
    '''Print the package version'''
    import types
    print('Python', sys.version)

    for val in globals().values():
        if isinstance(val, types.ModuleType):
            try:
                print(val.__name__, val.__version__)
            except:
                pass

def moist_lapse_rate_Gamma(T_in_K, p_in_hPa):
    T_in_C = T_in_K - 273.15
    # https://unidata.github.io/MetPy/latest/_modules/metpy/calc/thermo.html#saturation_vapor_pressure
    es = 6.1078 * np.exp(17.27 * T_in_C / (T_in_C+237.3))
    qs = 0.622 * es / (p_in_hPa - es)
    Gamma = (mc.grav / mc.cp) * (1.0 - (1.0 + mc.Lv * qs  /mc.RDGAS / T_in_K) 
            / (1.0 + mc.Lv**2 * qs / mc.cp / mc.RVGAS / T_in_K**2))
    return Gamma

def calc_theta(T_in_K, p_in_hPa):
    p0 = 1.0e3
    return T_in_K * (p0/p_in_hPa)**mc.kappa

def calc_low_cld_proxy_for_obs(temp, q, sp, tas):
    levels = temp.level
    lats = temp.lat
    lons = temp.lon

    # potential temperature
    print('Calc potential temperature...')
    p0 = 1.0e3
    theta = temp * (p0 / levels)**mc.kappa

    # =========== Lower Tropospheric Stability (LTS) ========= #
    # Klein, S. A., & Hartmann, D. L. (1993). The seasonal cycle 
    # of low stratiform clouds. Journal of Climate, 6(8), 1587-1606.
    # doi: 10.1175/1520-0442(1993)006<1587:tscols>2.0.co;2
    
    print('Calc LTS...')
    k700 =  min(range(len(levels)), key=lambda i: abs(levels[i]-7.0e2))
    print('The level closest to the 700hPa is %.2f'%levels[k700] + ' hPa.')
    LTS = theta[:,k700,:,:] - theta[:,-1,:,:]

    # =========== Estimated Inversion Stability (EIS) ========= #
    # Refer to: Wood and Bertherton, 2006, Journal of Climate
    #
    # EIS = LTS - Gamma_850_m * (z700-LCL)
    # LTS = theta_700 - theta_s

    print('Calc EIS...')
    if np.mean(sp.values) > 1e4:
        sp = sp / 1e2

    LTS =  temp[:,k700,:,:]*(p0 / levels[k700])**mc.kappa - tas *(p0/sp)**mc.kappa
    T850 = (temp[:,k700,:,:] + tas) / 2.0
    Gamma_850 = moist_lapse_rate_Gamma(T850, 850.0)
    z700  = mc.RDGAS * tas / mc.grav * np.log(1.0e3 / levels[k700])

    # Calc LCL
    rh0 = 0.8
    tas_units = tas.values * units.kelvin
    dewpt = metpy.calc.dewpoint_from_relative_humidity(tas_units, rh0)

    (plcl, tlcl) = metpy.calc.lcl(sp.values*units.hPa, tas_units, dewpt)
    # plcl in hPa, tlcl in K, but old version of metpy seems to ouput tlcl in degC
    plcl = xr.DataArray(np.array(plcl), coords=[tas.time, lats, lons], 
                        dims=['time', 'lat', 'lon'])
    tlcl = xr.DataArray(np.array(tlcl), coords=[tas.time, lats, lons],
                        dims=['time', 'lat', 'lon'])
    zlcl = mc.RDGAS * tas / mc.grav * np.log(1e3 / plcl)
    EIS   = LTS - Gamma_850 * (z700 - zlcl)

        # ============= Estimated Cloud-Top Entraiment Index (ECTEI) ============= #
    # Kawai, H., Koshiro, T., & Webb, M. J. (2017). Interpretation of factors controlling 
    # low cloud cover and low cloud feedback using a unified predictive index. 
    # Journal of Climate, 30(22), 9119-9131.
    # doi: 10.1175/jcli-d-16-0825.1
    
    print('Calc ECTEI...')
    k_en = 0.7
    C_qgap = 0.76
    beta = (1 - k_en) * C_qgap
    ECTEI = EIS - beta * mc.Lv / mc.cp * (q[:,-1,:,:] - q[:,k700,:,:])
    
    # =============== ELF: estimated low-level cloud fraction ================= #
    # Park, S., & Shin, J. (2019). Heuristic estimation of low-level cloud fraction
    #  over the globe based on a decoupling parameterization. 
    #  Atmospheric Chemistry and Physics, 19(8), 5635-5660.
    #  https://acp.copernicus.org/articles/19/5635/2019/
    
    print('Calc ELF...')
    
    delta_zs = 2750.0  # metre
    # ML is the LCL
    # For simplicity, we assume that z_ML=z_LCL over the entire globe. (Page 5638)
    Gamma_DL = moist_lapse_rate_Gamma(tlcl, plcl)
    Gamma_700 = moist_lapse_rate_Gamma(temp[:,k700,:,:], levels[k700])
    LTS_ML = theta[:,k700,:,:] - calc_theta(tlcl, plcl)

    # Eq(4) in Park and Shin (2019)
    z_inv = -LTS_ML / Gamma_700 + z700 + delta_zs * (Gamma_DL / Gamma_700)
    ind_z = np.array(z_inv<zlcl)
    z_inv.values[ind_z] = zlcl.values[ind_z]

    # Eq(7)
    beta1 = (z_inv + zlcl) / delta_zs
    # Eq(8)
    beta2 = np.sqrt(z_inv * zlcl) / delta_zs

    #q_ML = q_in_ML(levels, plcl, q)
    ptest = np.abs(plcl - levels)
    p_dim = ptest.dims.index('level')
    p_ind1 = np.argmin(ptest.values, axis=p_dim)
    p_ind = xr.DataArray(p_ind1, coords=[plcl.time, lats, lons], dims=['time', 'lat', 'lon'])
    q_ML = q[:,p_ind,:,:]

    # Eq(10) 
    f_para = np.array(q_ML / 0.003)
    f_para[f_para<0.15] = 0.15
    f_para[f_para>1] = 1.0
    f_para = xr.DataArray(f_para, coords=[q_ML.time, lats, lons], dims=['time', 'lat', 'lon'])
    
    # Eq(9) in Park and Shin (2019)
    ELF = f_para * (1 - beta2)

    return LTS, EIS, ECTEI, ELF

if __name__ == '__main__':
    print_python_and_package_info()

    out_dt_dir = '../data'

    s_year = 1979
    e_year = 2018

    print('For single level dataset...')
    dt_dir = '../inputs/'
    file_nm = os.path.join(dt_dir, 'ERA5_temp_wind_seaice_monthly_single_level_' 
              + str(s_year)+'_'+str(e_year)+'_t42.nc')
    ds1 = xr.open_dataset(file_nm)

    sst = ds1.sst
    u10 = ds1.u10
    v10 = ds1.v10

    Tadv = calc_Tadv(u10, v10, sst)
    # seconds_per_day = 24 * 3600
    # plot.close()
    # (Tadv.mean('time') * seconds_per_day).plot.contourf(levels=np.arange(-2.5,2.6,0.1), extend='both')
    # plot.show()

    # Get omega700, RH700
    fn_postfix = 'mon_pres_levs_' + str(s_year) + '_' + str(e_year) + '_t42.nc'
    # var_nm_arr = ['temperature', 'vertical_velocity', 'relative_humidity']

    var_nm = 'vertical_velocity'
    file_nm = os.path.join(dt_dir, '_'.join(['ERA5', var_nm, fn_postfix]))
    ds_omega = xr.open_dataset(file_nm)
    omega700 = ds_omega.w.sel(level=7e2)

    # plot.close()
    # (omega700.mean('time') * seconds_per_day / 1e2).plot.contourf(levels=np.arange(-95,100,5), extend='both')
    # plot.show()

    var_nm = 'relative_humidity'
    file_nm = os.path.join(dt_dir, '_'.join(['ERA5', var_nm, fn_postfix]))
    ds_rh = xr.open_dataset(file_nm)
    rh700 = ds_rh.r.sel(level=7e2)

    # plot.close()
    # rh700.mean('time').plot.contourf(levels=np.arange(0,105,5), extend='both')
    # plot.show()

    var_nm = 'temperature'
    file_nm = os.path.join(dt_dir, '_'.join(['ERA5', var_nm, fn_postfix]))
    ds_t = xr.open_dataset(file_nm)
    temp = ds_t.t
    
    var_nm = 'specific_humidity'
    file_nm = os.path.join(dt_dir, '_'.join(['ERA5', var_nm, fn_postfix]))
    ds_q = xr.open_dataset(file_nm)
    sphum = ds_q.q
    
    LTS, EIS, ECTEI, ELF = calc_low_cld_proxy_for_obs(temp, sphum, ds1.sp, ds1.t2m)

    # plot.close()
    # LTS.mean('time').plot.contourf(levels=np.arange(-20,21,2), extend='both')
    # plot.show()

    # plot.close()
    # EIS.mean('time').plot.contourf(levels=np.arange(-10,11,1), extend='both')
    # plot.show()

    # ================= Get and write stddev for ERA5 dataset ================= #
    var_names = ['SST', 'LTS', 'EIS', 'ELF', 'ECTEI', 'Tadv', 'RH700', 'omega700']
    var_list = [sst, LTS, EIS, ELF, ECTEI, Tadv, rh700, omega700]
    var_mean = []
    var_std = []
    anomalies_arr = []
    stand_anomalies_arr = []
    for var, var_nm in zip(var_list, var_names):
        var_de = detrend_dim(var, 'time', deg=1, skipna=True)
        clim_mean = var_de.groupby("time.month").mean("time")
        clim_std = var_de.groupby("time.month").std("time")
        anomalies = var_de.groupby("time.month") - clim_mean

        stand_anomalies = xr.apply_ufunc(lambda x, m, s: (x - m) / s,
                    var_de.groupby("time.month"), clim_mean, clim_std)
        stand_anomalies_arr.append(stand_anomalies)

        var_mean.append(clim_mean)
        var_std.append(clim_std)
        anomalies_arr.append(anomalies)

    # Save stddev to files
    save_dt = {}
    for dt, var_nm in zip(var_std, var_names):
        save_dt[var_nm] = (dt.dims, dt)

    coords = {}
    for d in dt.dims:
        coords[d] = dt[d]

    out_ds = xr.Dataset(save_dt, coords=coords)
    fn = os.path.join(out_dt_dir, 'ERA5_monthly_stddev_for_CCFs.nc')
    out_ds.to_netcdf(fn, mode='w', format='NETCDF3_CLASSIC')

    print(fn, 'saved.')

    # plot.close()
    # fig, axes = plot.subplots(nrows=3, ncols=2, sharey=False)
    # for ax, var, var_nm in zip(axes[0:-1], var_std, var_names):
    #     ax.plot(var.month, var.mean(('lat', 'lon')))
    # plot.show()

    # plot.close()
    # fig, axes = plot.subplots(nrows=3, ncols=2, sharey=False)
    # for ax, var, var_nm in zip(axes[0:-1], anomalies_arr, var_names):
    #     ax.plot(var.time, var.mean(('lat', 'lon')))
    # plot.show()

    # plot.close()
    # fig, axes = plot.subplots(nrows=3, ncols=2, sharey=False)
    # for ax, var, var_nm in zip(axes[0:-1], stand_anomalies_arr, var_names):
    #     ax.plot(var.time, var.mean(('lat', 'lon')))
    # plot.show()

