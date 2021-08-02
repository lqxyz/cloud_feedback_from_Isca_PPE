#!/bin/bash

outdir=./inputs/

# # Delete the symbol link files before downloading 
# find $outdir -type l -delete

echo "Download cloud radiative kernel data from Zelinka et al (2012)"
# https://stackoverflow.com/questions/4944295/skip-download-if-files-already-exist-in-wget
wget -c -N --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/cloud_kernels2.nc
wget -c -N --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/obs_cloud_kernels3.nc
echo "Cloud kernel data saved in $outdir"

echo "Download ISCCP observation data" 
wget -c -N --directory-prefix=$outdir --no-check-certificate https://climserv.ipsl.polytechnique.fr/cfmip-obs/data/ISCCP/clisccp_198307-200806.nc

echo "Download ERA-Interim omega data"
python -u download_ERA_Interim_omega.py
echo "Done! ERA-Interim omega downloaded!"

echo "Download ERA5 dataset"
python -u download_ERA5_single_lev.py
python -u download_ERA5_pres_lev.py
echo "Done! ERA5 dataset downloaded!"

