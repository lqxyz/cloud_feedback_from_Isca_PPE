#!/bin/bash

outdir=./inputs/

# # Delete the symbol link files before downloading
# find $outdir -type l -delete

echo "Download cloud radiative kernel data from Zelinka et al (2012)"
# https://stackoverflow.com/questions/4944295/skip-download-if-files-already-exist-in-wget
wget -c -N --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/cloud_kernels2.nc
wget -c -N --directory-prefix=$outdir https://github.com/mzelinka/cloud-radiative-kernels/raw/master/data/obs_cloud_kernels3.nc
echo "Cloud kernel data saved in $outdir"

echo "Download Zelinka et al (2021) data"
dt_dir=$outdir/zelinka_data
[[ ! -f $dt_dir ]] && mkdir $dt_dir

fn_arr=("cmip56_forcing_feedback_ecs.json" \
        "cmip5_amip4K_cld_fbks.json" \
        "cmip5_amip_cld_errs.json" \
        "cmip6_amip-p4K_cld_fbks.json" \
        "cmip6_amip_cld_errs.json" \
        "AC_clisccp_wap_ISCCP_HGG_198301-200812.nc" \
        "AC_clisccp_ISCCP_HGG_198301-200812.nc")
for fn in "${fn_arr[@]}"
do
    wget -c -N --directory-prefix=$dt_dir https://raw.githubusercontent.com/mzelinka/assessed-cloud-fbks/master/data/$fn
done
cd $dt_dir
for fn in cloud_kernels2.nc obs_cloud_kernels3.nc
do
    [[ -L $fn ]] && rm $fn && ln -s ../$fn
done
cd ../..

echo "Download ISCCP observation data"
wget -c -N --directory-prefix=$outdir --no-check-certificate https://climserv.ipsl.polytechnique.fr/cfmip-obs/data/ISCCP/clisccp_198307-200806.nc

echo "Download ERA-Interim omega data"
python -u download_ERA_Interim_omega.py
echo "Done! ERA-Interim omega downloaded!"

echo "Download ERA5 dataset"
python -u download_ERA5_single_lev.py
python -u download_ERA5_pres_lev.py
echo "Done! ERA5 dataset downloaded!"

