#!/bin/bash

cd scripts/

echo -e "\n$(date) Verify COSP implementaion in Isca"
python -u verify_cosp_in_isca.py

echo -e "\n$(date) Evaluate clisccp in ascending and descending dynamical regimes in tropical regions"
python -u evaluate_clisccp_obs_isca_together.py

# Ensemble mean profile of PPE
echo -e "\n$(date) PPE profiles changes for baisc states"
python -u profile_changes_PPE.py

echo -e "\n$(date) Ensemble mean profiles for Temp, RH, CF and cloud water in control and 4xCO2 exps"
python -u cmp_ppe_profiles.py

echo -e "\n$(date) Compare different methods to calc cld fbk with gregory plots..."
python -u calc_toa_cld_induced_flux_from_zelinka_kernel.py
python -u cmp_cld_fbk_gregory.py
python -u cmp_cld_fbk_tbl.py

echo -e "\n$(date) Calculate and decompose cloud feedback"
python -u write_cld_feedback_final_nyr.py
 
echo -e "\n$(date) Calculate and decompose cloud feedback for regions"
python -u write_regional_cld_fbk.py
 
echo -e "\n$(date) Plot map and zonal mean of cloud feedback"
python -u plot_cld_fbk_maps.py

echo -e "\n$(date) Run Zelinka et al (2021) analyssi for Isca"
python -u run_zelinka_assessment.py

echo -e "\n$(date) Scatter plot of decomposed cloud feedback"
python -u plot_cld_fbk_decomp_with_ensemble_mean.py
 
echo -e "\n$(date) Scatter plot of decomposed cloud feedback for regions"
python -u plot_cld_fbk_decomp_regions_v2.py
 
echo -e "\n$(date) Calc the contribution to cldfbk covariance"
python -u plot_components_covriance.py
 
echo -e "\n$(date) Composite analysis for cloud feedbacks"
python -u get_binned_data_qflux_cld_fbk.py
python -u cld_fbk_composite_qflux.py

echo -e "\n$(date) Cloud response in trade-wind cumulus (Cu) and stratocumulus (Sc) cloud regions"
python -u two_low_clouds_metric_analysis.py

echo -e "\n$(date) EIS change analysis" 
python -u write_low_cld_proxy_and_temp_change.py
python -u EIS_change.py

echo -e "\n$(date) Prepare ERA5 data"
python -u calc_era5_CCF.py

echo -e "\n$(date) Cloud controling factor (CCF) analysis"
method='EIS' # or 'LTS'
T0=4  # Thresold to distinguish Cu and Sc: EIS:4, LTS:18.5
num_proxy=5 #2 (SST, EIS, ...)

echo -e "\n$(date) For all low clouds"
python -u ccf_analysis_all_metric.py 'eis' $num_proxy $method &> tmp.eis.all.txt &
# python -u ccf_analysis_all_metric.py 'ELF' $num_proxy $method &> tmp.elf.all.txt &
 
echo -e "\n$(date) For trade-wind Cu and Sc"
# python -u ccf_analysis_cu_sc_metric.py 'Cu' 'ELF' $T0 $num_proxy $method &> tmp.elf.cu.$T0.txt & 
# sleep 1
# python -u ccf_analysis_cu_sc_metric.py 'Sc' 'ELF' $T0 $num_proxy $method &> tmp.elf.sc.$T0.txt &
# sleep 1
python -u ccf_analysis_cu_sc_metric.py 'Cu' 'eis' $T0 $num_proxy $method &> tmp.eis.cu.$T0.txt & 
sleep 1
python -u ccf_analysis_cu_sc_metric.py 'Sc' 'eis' $T0 $num_proxy $method &> tmp.eis.sc.$T0.txt 

echo -e "\n$(date) Plot for the CCF analysis"
python -u plot_CCF_Cu_Sc_single_sum_metric.py 'EIS'
#python -u plot_CCF_Cu_Sc_single_sum_metric.py 'ELF'

echo -e "\n$(date) Get equilibrium climate sensitivity (ECS), effective radiative foring and total climate feedbacks"
python -u get_forcing_slope_ecs.py 
python -u construct_PPE_forcing_feedback_ECS_table.py

python -u land_sea_cld_fbk_from_cmip_isca_models.py 0
python -u land_sea_cld_fbk_from_cmip_isca_models.py 1

echo -e "\n$(date) Done"

