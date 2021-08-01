#!/usr/bin/env python

from __future__ import print_function
import os
try:
    from ecmwfapi import ECMWFDataServer
except:
    print("Error: Please install ecmwfapi following the instructions on https://github.com/ecmwf/ecmwf-api-client")

server = ECMWFDataServer()
start_year = 1979
end_year = 2017

month_list = []
for year in range(start_year, end_year+1):
    for mon in range(1, 12+1):
        month_list.append('%d%02d01' % (year,mon))

month_str = '/'.join(month_list)

#var_names = ['uwind', 'vwind', 'omega']
#params = ["131.128", "132.128", "135.128"]
var_names = ['omega']
params = ["135.128"]
for var, param in zip(var_names, params):
    print(var)
    target_fn = 'ecmwf_' + var + '_' + str(start_year) + '_' + str(end_year) + '.nc'

    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": month_str,
        "expver": "1",
        "grid": "1.0/1.0",
        "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "levtype": "pl",
        "param": param,
        "stream": "moda",
        "type": "an",
        "format":"netcdf",
        "target": target_fn,
    })
    print('Remap the data to T42')
    os.system("cdo remapbil,t42grid " + target_fn + " " + target_fn.replace('.nc', '_t42.nc'))
    os.remove(target_fn)

