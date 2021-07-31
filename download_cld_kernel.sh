#!/bin/bash

outdir=./inputs/
echo Download cloud radiative kernel data from Zelinka et al (2012)

wget -c --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/cloud_kernels2.nc
wget -c --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/obs_cloud_kernels3.nc

echo Done! Cloud kernel data saved in $outdir

