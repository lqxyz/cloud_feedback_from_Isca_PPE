from __future__ import print_function
import numpy as np
import xarray as xr
from lcl_array import lcl   # Romps, 2017, JAS

def estimated_inversion_strength(ds):
    # tsurf and sp z should be xarray Dataset
    # Define some parameters
    RDGAS = 287.04          # J/kg K
    RVGAS = 461.50          # J/kg K
    zerodegc = 273.15       # K
    TFREEZE = 273.16        # K
    grav = 9.80             # kg/m^2 Gravitational acceleration
    cp = 1005.0
    kappa = RDGAS / cp
    Lv = 2.47e6             # J/kg

    p0 = 1.0e3
    levels = ds.pfull
    sp = ds.ps / 1e2
    tsurf = ds.t_surf
    temp = ds.temp

    k700 =  min(range(len(levels)), key=lambda i: abs(levels[i]-7.0e2))
    ksurf = min(range(len(levels)), key=lambda i: abs(levels[i]-p0))
    k850 = min(range(len(levels)), key=lambda i: abs(levels[i]-8.5e2))
    LTS = temp[...,k700,:,:] * (p0/np.array(levels[k700]))**kappa - tsurf*(p0/sp)**kappa
    T850 = temp[...,k850,:,:] #(temp[...,k700,:,:] + temp[...,ksurf,:,:]) / 2.0

    T850_C = T850 - 273.15
    es = 6.1078 * np.exp(17.27 * T850_C / (T850_C + 237.3))
    qs = 0.622 * es / (850.0 - es)
    Gamma = (grav / cp) * (1.0 - (1.0 + Lv * qs / RDGAS / T850) / (1.0 + Lv**2 * qs / cp / RVGAS / T850**2))
    z700 = ds.height[:,k700,:,:].values #RDGAS * tsurf / grav * np.log(p0/7.0e2)

    # calc lcl
    # rh0 = 0.8
    # #rh0 = np.array(rh_surf)
    # tas_units = tsurf.values*units.kelvin
    # dewpt = metpy.calc.dewpoint_rh(tas_units, rh0)
    # (plcl, tlcl) = metpy.calc.lcl(sp.values/1e2*units.hPa, tas_units, dewpt)
    # if len(data_shp)==4:
    #     plcl = xr.DataArray(np.array(plcl), coords=[time, lats, lons], dims=['time', 'lat', 'lon'])
    # elif len(data_shp)==3:
    #     plcl = xr.DataArray(np.array(plcl), coords=[lats, lons], dims=['lat', 'lon'])
    # else:
    #     print('Error: the length of shape should be 3 or 4.')
    # zlcl = RDGAS * tsurf / grav * np.log(sp/1e2 / plcl)

    zlcl = lcl(sp * 1e2, tsurf, rh=ds.rh_2m/1e2) # P should be in pascals
    EIS = LTS - Gamma * (z700-zlcl)
    dims = ds.t_surf.dims
    ds['eis'] = (dims, EIS)
    ds['lts'] = (dims, LTS)
