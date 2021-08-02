import numpy as np
import xarray as xr

def calc_Tadv(u10, v10, sst):
    '''
    Refer to: Eq(2) in Scott et al. (2020), 
    Journal of Climate, 33(18), 7717-7734.
    Observed Sensitivity of Low-Cloud Radiative Effects to 
    Meteorological Perturbations over the Global Oceans. 
    DOI: https://doi.org/10.1175/JCLI-D-19-1028.1
    
    Tadv = - U10 / (R_earth * cos(phi)) * dSST/dlambda 
           - V10 /  R_earth             * dSST/dphi
    where phi and lambda are latitude and longitude respectively.
    Using second-order centered finite-difference scheme
    '''
    # metpy does not work... due to xarray index error?
    # from metpy.calc import advection
    # Tadv = advection(sst, u10, v10)

    lats_rad = xr.ufuncs.deg2rad(u10.lat)
    lons_rad = xr.ufuncs.deg2rad(u10.lon)
    R = 6.371e6 # m, Earth's radius

    coords = {}
    dims = u10.dims
    for d in dims:
        coords[d] = u10[d]

    dsst_dlon = np.gradient(sst, lons_rad, axis=dims.index('lon'))
    dsst_dlon = xr.DataArray(dsst_dlon, dims=dims, coords=coords)
    dsst_dlat = np.gradient(sst, lats_rad, axis=dims.index('lat'))
    dsst_dlat = xr.DataArray(dsst_dlat, dims=dims, coords=coords)

    # Units: K/s
    Tadv = -u10 / (R * xr.ufuncs.cos(lats_rad)) * dsst_dlon - v10 / R * dsst_dlat

    return Tadv

