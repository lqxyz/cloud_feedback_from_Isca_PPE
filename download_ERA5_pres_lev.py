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
    var_list = ['vertical_velocity',
                'relative_humidity',
                'specific_humidity',
                'temperature',
                # 'vorticity',
                # 'potential_vorticity',
                # 'divergence',
                # 'specific_snow_water_content',
                # 'fraction_of_cloud_cover',
                # 'geopotential',
                # 'specific_cloud_ice_water_content',
                # 'specific_cloud_liquid_water_content',
                # 'specific_rain_water_content',
                # 'u_component_of_wind',
                # 'v_component_of_wind',
                ]

    c = cdsapi.Client()

    dt_dir = './inputs/'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)

    for var in var_list:
        target_fn = os.path.join(dt_dir, 'ERA5_'+ var + '_mon_pres_levs_1979_2019.nc')

        c.retrieve(
            'reanalysis-era5-pressure-levels-monthly-means',
            {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'grid': '1.0/1.0', # resolution
            'variable': [ var ],
            'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
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
        target_fn)

        print('Remap ' + var +' to T42')
        os.system("cdo remapbil,t42grid " + target_fn + " " + target_fn.replace('.nc', '_t42.nc'))
        os.remove(target_fn)
