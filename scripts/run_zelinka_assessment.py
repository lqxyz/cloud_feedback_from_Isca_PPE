#=============================================
# Performs the cloud feedback and cloud error
# metric calculations in preparation for comparing
# to expert-assessed values from Sherwood et al (2020)
#=============================================
import numpy as np
import xarray as xr
import pandas as pd
import json
from pathlib import Path
from datetime import date
import sys
from analysis_functions import add_datetime_info
from isca_cre_cwp import calc_toa_cre_for_isca
from bin_clisccp_area_weights_time import (construct_clisccp_tau_pres, bin_obs_exp_clisccp_data, get_final_nyr_mean)
import organize_jsons as OJ
import cal_CloudRadKernel_isca as CRK
import zelinka_cld_fbk_ecs_assessment_for_isca as dataviz

def get_isca_cld_fbk_and_err(exp_grps, isca_cld_fbk_fn, isca_cld_err_fn):
    # =================== Read Isca dataset ================== #
    file_nms = ['extracted_data_301_360.nc', 'extracted_data_661_720.nc']
    file_clisccp_nms = ['extracted_clisccp_data_301_360.nc', 
                        'extracted_clisccp_data_661_720.nc']

    isca_cld_fbk_dict = {}
    isca_cld_err_dict = {}

    for exp_grp in exp_grps: #[0:2]:
        print('begin:', isca_cld_fbk_dict.keys())
        print(exp_grp, ': Read dataset...')
        ds_arr = []
        ds_clisccp_arr = []
        
        for file_nm, file_clisccp_nm in zip(file_nms, file_clisccp_nms):
            fn = ppe_dir / file_nm.replace('.nc', '_'+exp_grp+'.nc')
            ds = xr.open_dataset(fn, decode_times=False)
            ds_arr.append(ds)

            fn = ppe_clisccp_dir / file_clisccp_nm.replace('.nc', '_'+exp_grp+'.nc')
            ds = xr.open_dataset(fn, decode_times=False)
            ds_clisccp_arr.append(ds)

        # Keep the time coordinates the same
        ds_arr[1]['time'] = ds_arr[0].time

        print('Construct clisccp')
        for ds, ds_clisccp in zip(ds_arr, ds_clisccp_arr):
            add_datetime_info(ds)
            #construct_clisccp(ds)
            construct_clisccp_tau_pres(ds_clisccp, ds)
        print('Calc CRE')
        calc_toa_cre_for_isca(ds_arr)

        fbk_dict, err_dict = CRK.CloudRadKernel(ds_arr)

        variant = 'r1i1p1f1'
        model = exp_grp
        updated_err_dict = OJ.organize_err_jsons(err_dict, model, variant) 
        updated_fbk_dict = OJ.organize_fbk_jsons(fbk_dict, model, variant)

        isca_cld_fbk_dict[model] = updated_fbk_dict[model]
        isca_cld_err_dict[model] = updated_err_dict[model]
    
        print('end:', isca_cld_fbk_dict.keys())

    # end for loop
    meta = 'metadata'
    isca_cld_fbk_dict[meta] = updated_fbk_dict[meta]
    isca_cld_err_dict[meta] = updated_err_dict[meta]
    print('final:', isca_cld_fbk_dict.keys())

    with open(isca_cld_fbk_fn, 'w') as outfile: 
        # https://stackoverflow.com/questions/53110610
        json.dump(isca_cld_fbk_dict, outfile, separators=(',', ':'))

    with open(isca_cld_err_fn, 'w') as outfile:
        json.dump(isca_cld_err_dict, outfile, separators=(',', ':'))

    return isca_cld_fbk_dict, isca_cld_err_dict

def get_isca_forcing_ecs(exp_grps, isca_forcing_ecs_fn):
    # format 'CMIP5'/model_name/variant (r1i1)/ fbks and foricngs 
    #         (ERF2x:2.9, CLD:, ECS:, LWCLD:, SWCLD:)
    dt_dir = Path('../data')
    # This file is produced by 'get_forcing_slope_ecs.py'
    # ecs_fn = dt_dir / 'forcing_slope_ECS_qflux_PPE_nyr_20_22.csv'
    ecs_fn = dt_dir / 'forcing_slope_ECS_qflux_PPE_nyr_30_12.csv'
    ecs_tbl = pd.read_csv(ecs_fn, index_col=0)
    
    variant = 'r1i1p1f1'
    forcing_ecs_dict = {}
    for exp_grp in exp_grps:
        forcing_ecs_dict[exp_grp] = {}
        dt_dict = {}
        for key in ['slope', 'ERF_2x', 'ECS']:
            if 'ERF_2x' == key:
                key1 = 'ERF2x'
            elif 'slope' == key:
                key1 = 'NET'
            else:
                key1 = key
            dt_dict[key1] = ecs_tbl[key][exp_grp]
        forcing_ecs_dict[exp_grp] = {variant: dt_dict}

    out_dict = {
        'isca': forcing_ecs_dict,
        'metadata': {
            'author': 'QL, ql260@exeter.ac.uk',
            'date_modified': str(date.today()),
        }
    }
    with open(isca_forcing_ecs_fn, 'w') as outfile:
        json.dump(out_dict, outfile, separators=(',', ':'))

    return out_dict

if __name__ == '__main__':
    RERUN = False #True
    out_dt_dir = Path('../data/zelinka_data/')
    if not out_dt_dir.exists():
        out_dt_dir.mkdir()

    base_dir = Path('../inputs')
    ppe_dir = base_dir / 'qflux_extracted_data'
    ppe_clisccp_dir = base_dir / 'qflux_clisccp_data'

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exp_markers = list(exp_tbl.iloc[:, 2])

    ## ========= Get Isca cloud fbk and err data ========= ##
    print('Get the json file for Isca simulations...')
    isca_cld_fbk_fn = out_dt_dir / 'isca_cld_fbks.json'
    isca_cld_err_fn = out_dt_dir / 'isca_cld_errs.json'
    if RERUN or (not isca_cld_fbk_fn.exists())  or (not isca_cld_err_fn.exists()):
        isca_cld_fbk_dict, isca_cld_err_dict = \
            get_isca_cld_fbk_and_err(exp_grps, isca_cld_fbk_fn, isca_cld_err_fn)

    # ========= Get the ECS for Isca simulations ========= ##
    isca_forcing_ecs_fn = out_dt_dir / 'isca_forcing_ECS.json'
    if RERUN or (not isca_forcing_ecs_fn.exists()):
        isca_forcing_ecs_dict = get_isca_forcing_ecs(exp_grps, isca_forcing_ecs_fn)

    # ==================== Plot the figures ============== ## 
    result = dataviz.make_all_figs(exp_grps, exp_markers, 
        isca_cld_fbk_fn, isca_cld_err_fn, isca_forcing_ecs_fn)
    print('Done!')
