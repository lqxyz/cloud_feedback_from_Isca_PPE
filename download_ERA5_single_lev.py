#!/usr/bin/env python
from __future__ import print_function
import os
import sys
try:
    import cdsapi
except:
    print("Error: Please install cdsapi following the instructions on https://cds.climate.copernicus.eu/api-how-to")
    sys.exit(1)

if __name__ == '__main__':
    c = cdsapi.Client()

    dt_dir = '../inputs/ERA5'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)
    target_fn = os.path.join(dt_dir, 'ERA5_temp_wind_seaice_monthly_single_level_1979_2019.nc')

    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'grid': '1.0/1.0', # resolution
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 
            '2m_dewpoint_temperature', '2m_temperature', 
            'mean_sea_level_pressure',
            'sea_ice_cover', 'sea_surface_temperature',
            'skin_temperature', 'surface_pressure',
        ],
        'year': [
        '1979', '1980', '1981',
        '1982', '1983', '1984',
        '1985', '1986', '1987',
        '1988', '1989', '1990',
        '1991', '1992', '1993',
        '1994', '1995', '1996',
        '1997', '1998', '1999',
        '2000', '2001', '2002',
        '2003', '2004', '2005',
        '2006', '2007', '2008',
        '2009', '2010', '2011',
        '2012', '2013', '2014',
        '2015', '2016', '2017',
        '2018', '2019',
        ],
        'month': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        ],
        'time': '00:00',
        },
        target_fn )

    print('Remap the data to T42')
    os.system("cdo remapbil,t42grid " + target_fn + " " + target_fn.replace('.nc', '_t42.nc'))
    os.remove(target_fn)

