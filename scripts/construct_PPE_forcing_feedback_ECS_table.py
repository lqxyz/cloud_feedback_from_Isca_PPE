import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')

def get_global_annual_mean(dt):
    dims = dt.dims
    try:
        lat_ind = dims.index('latitude')
        lat_nm = 'latitude'
        lon_nm = 'longitude'
    except:
        lat_ind = dims.index('lat')
        lat_nm = 'lat'
        lon_nm = 'lon'

    coslat = np.cos(np.deg2rad(dt[lat_nm]))
    dt_gm = np.average(dt.mean(lon_nm), axis=lat_ind, weights=coslat)
    dims1 = [d for d in dims if d!=lat_nm and d!=lon_nm]
    coords = [dt[d] for d in dims1]
    dt_gm = xr.DataArray(dt_gm, coords=coords, dims=dims1)
    add_datetime_info(dt_gm)
    dt_gm_annual = dt_gm.groupby('year').mean()

    return dt_gm_annual

def get_reg_cld_fbk(exp_grps, reg='global', land_or_sea='',
                hi_or_low='ALL', dt_dir='../data'):
    """
    reg: ['tropical', 'mid', 'high', 'global']
    land_or_sea: ['', 'land', 'sea']
    hi_or_low: ['HI680', 'LO680', 'ALL']
    """
    # 'cld_fbk_decomp_v2_regional_mean_'
    # The first 3 cols show the change of cloud feedback 
    #   estimated by Delta CRE / delta Ts (LW, SW, Net)
    # The last 3 cols show the cld fbk from kernel method
    N = len(exp_grps)
    out_tbl = np.zeros((N, 3))

    #row_name = '_'.join([reg, land_or_sea, hi_or_low])
    row_name = '_'.join(filter(None, [reg, land_or_sea, hi_or_low]))
    print(row_name)
    fbk_names = ['lw_cld_tot', 'sw_cld_tot', 'net_cld_tot']
    for i, exp in enumerate(exp_grps):
        file_nm = P(dt_dir, 'cld_fbk_decomp_v2_regional_mean_land_sea_' + exp + '.csv')
        tbl = pd.read_csv(file_nm, index_col=0)
        row_ind = list(tbl.index.values).index(row_name)
        for j, fbk_nm in enumerate(fbk_names):
            out_tbl[i,j] = tbl.iloc[row_ind][fbk_nm]
    
    col_names = ['LW cldfbk', 'SW cldfbk', 'net cldfbk']
    out_tbl = pd.DataFrame(data=out_tbl, index=exp_grps, columns=col_names)

    return out_tbl


if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dt_dir = '../data'
    if not os.path.exists(dt_dir):
        os.mkdir(dt_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    #exp_tbl = pd.read_csv('isca_qflux_exps.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    ECS_tbl_fn = P(dt_dir, 'forcing_slope_ECS_qflux_PPE_nyr_30_12.csv')
    # column_nms = ['slope', 'intercept', 'rvalue', 'pvalue', 
    #              'stderr', 'intercept_stderr', 'ERF_2x', 'ECS']
    ECS_tbl = pd.read_csv(ECS_tbl_fn, index_col=0)

    cld_fbk_tbl = get_reg_cld_fbk(exp_grps, reg='global', land_or_sea='',
                hi_or_low='ALL', dt_dir='../data')
    # fbk_names = ['lw_cld_tot', 'sw_cld_tot', 'net_cld_tot']

    # 'Exp', 
    # column_nms = [r'ERF$_{2x}$', '$\lambda$', r'$\lambda_{cld}$',
    #              r'$\lambda_{cld\_SW}$',  r'$\lambda_{cld\_LW}$', 'ECS']
    column_nms = ['ERF_2x', 'lambda', 'lambda_cld',
                 'lambda_cld_SW',  'lambda_cld_LW', 'ECS']

    out_tbl = np.zeros((len(exp_grps), len(column_nms)))

    for i, exp_grp in enumerate(exp_grps):
        col_val = [ECS_tbl.loc[exp_grp]['ERF_2x'], ECS_tbl.loc[exp_grp]['slope'],
                  cld_fbk_tbl.loc[exp_grp]['net cldfbk'], cld_fbk_tbl.loc[exp_grp]['SW cldfbk'],
                  cld_fbk_tbl.loc[exp_grp]['LW cldfbk'], ECS_tbl.loc[exp_grp]['ECS']]
        out_tbl[i,:] = col_val
    
    out_tbl = pd.DataFrame(data=out_tbl, index=exp_grps, columns=column_nms) # 

    file_name = P(dt_dir, 'ECS_forcing_feedbacks_for_Isca_PPE.csv')
    out_tbl.to_csv(file_name, header=True, index=True, float_format="%.10f")
    
    # Latex format
    file_name = P(dt_dir, 'ECS_forcing_feedbacks_for_Isca_PPE_latex.csv')
    out_tbl.to_latex(file_name, float_format="%.2f")
    
