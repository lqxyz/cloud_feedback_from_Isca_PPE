import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
import warnings
warnings.simplefilter(action='ignore')

def get_cldfbk_component_table_for_cov(exp_grps, float_fmt='%.4f', dt_dir='./data'):
    # The first 5 cols for LW components, the mid 5 cols for SW components, 
    # and the last 5 cols for net components

    N = len(exp_grps)
    sections = ['ALL', 'HI680', 'LO680']
    row_names = []
    reg = 'global'
    for sec in sections:
        row_names.append('_'.join([reg, sec]))

    out_tbl_dict = {}
    out_tbl = np.zeros((N, 15*len(row_names)))

    for i, exp in enumerate(exp_grps):
        file_nm = P(dt_dir, 'cld_fbk_decomp_v2_regional_mean_' + exp + '.csv')
        tbl = pd.read_csv(file_nm, index_col=0)

        for nn, row_nm in enumerate(row_names):
            kk = tbl.index.tolist().index(row_nm)
            out_tbl[i,nn*15:(nn+1)*15] = tbl.iloc[kk]

    col_names = ['_'.join([x.replace(reg+'_', ''), y]) for x in row_names for y in tbl.columns]
    out_tbl = pd.DataFrame(data=out_tbl, index=exp_grps, columns=col_names)

    file_name = os.path.join(dt_dir, 'cld_fbk_components_for_cov.csv')
    out_tbl.to_csv(file_name, header=True, index=True, float_format=float_fmt)

    return out_tbl

def cal_cov_matrix_for_cld_fbk_components(tbl, var_names, with_err=False):
    #sections = ['ALL', 'HI680', 'LO680']
    #var_names = ['net_cld_amt', 'net_cld_alt', 'net_cld_tau', 'net_cld_err']
    cov_col_names = ['_'.join([s, vn]) for s in ['LO680', 'HI680'] for vn in var_names]
    
    print(cov_col_names)

    N = len(cov_col_names)
    # cov_mat = np.zeros((N,N))
    # for i, col1 in enumerate(cov_col_names):
    #     for j, col2 in enumerate(cov_col_names):
    #         cov_mat[i,j] = np.cov(tbl[col1], tbl[col2])[0,1]

    # Each row represents a variable, 
    # and each column a single observation of all those variables.
    # https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    dt_mat = np.zeros((N, len(tbl[cov_col_names[0]])))
    for i, col in enumerate(cov_col_names):
        dt_mat[i,:] = tbl[col]
    cov_mat = np.cov(dt_mat)
    # print('')
    # print(dt_mat)
    # print('')
    # print(cov_col_names, 'Cov_mat is:')
    # print(cov_mat)
    # print('')
    # print('')
    # print(cov_col_names, 'stddev Cov_mat is:')
    # print(np.sqrt(cov_mat))
    # print('')
    # print('Correlation')
    # cor = np.corrcoef(dt_mat)
    # print(cor)
    # print('End')

    tot_cov = np.sum(cov_mat)
    # print(tot_cov)
    # print(np.sqrt(tot_cov))

    cov_mat_ratio = cov_mat / tot_cov

    # calculate total variance of net feedback
    net_col = 'ALL_net_cld_tot'
    # print(np.cov(tbl[net_col]))
    # print(np.sqrt(np.cov(tbl[net_col])))

    # Remove the columns
    if not with_err:
        ind_arr = []
        for col in cov_col_names:
            if 'err' in col.lower():
                ind_arr.append(cov_col_names.index(col))
        for ind in ind_arr:
            cov_mat[ind,:] = np.nan
            cov_mat[:,ind] = np.nan
        NN = N - len(ind_arr)
        cov_mat2 = cov_mat[~np.isnan(cov_mat)].reshape((NN, NN))
        cov_mat_ratio2 =  cov_mat2 / tot_cov
    else:
        NN = N
        cov_mat_ratio2 = cov_mat_ratio

    # fill lower matrix with nan, double upper matrix
    il1 = np.tril_indices(NN, -1)
    iu1 = np.triu_indices(NN, 1)
    #cov_mat_ratio2[il1] = np.nan
    #cov_mat_ratio2[iu1] = cov_mat_ratio2[iu1] * 2

    cov_mat_ratio2[iu1] = np.nan
    cov_mat_ratio2[il1] = cov_mat_ratio2[il1] * 2

    return cov_mat_ratio2

if __name__ == '__main__':
    P = os.path.join
    dt_dir = '../data'
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])

    tbl = get_cldfbk_component_table_for_cov(exp_grps, 
                    float_fmt='%.4f', dt_dir=dt_dir)
    var_names = ['net_cld_amt', 'net_cld_alt', 'net_cld_tau', 'net_cld_err']
    cov_mat = cal_cov_matrix_for_cld_fbk_components(tbl, var_names)
    cov_mat_with_err = cal_cov_matrix_for_cld_fbk_components(tbl, var_names, with_err=True)
    var_names = ['net_cld_tot']
    cov_mat_low_nonlow = cal_cov_matrix_for_cld_fbk_components(tbl, var_names)

    # Covariance matrix plot
    labels = ['Low\n amount', 'Low\n altitude', 'Low\n optical depth',
              'Non-low\n amount', 'Non-low\n altitude', 'Non-low\n optical depth']
    data = pd.DataFrame(cov_mat, columns=labels, index=labels)

    plot.close()
    fig, ax = plot.subplots(nrows=1, ncols=1, axwidth=4.5)
    m = ax.heatmap(
        data, cmap='ColdHot', vmin=-1, vmax=1, N=100,
        lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
        clip_on=False,  # turn off clipping so box edges are not cut in half
    )
    ax.format(title='Breakdown by altitude & type', titleweight='bold', alpha=0,
        xloc='bottom', yloc='left', #xloc='top', yloc='right',
        yreverse=True, linewidth=0, # ticklabelweight='bold',
        ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
    )
    fig_name = P(fig_dir, 'covariance_contribution_breakdown.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    # plot.show()

    # ======================== WITH ERROR ========================== #
    labels = ['Low\n amount', 'Low\n altitude', 'Low\n optical depth', 'Low\n residual',
              'Non-low\n amount', 'Non-low\n altitude', 'Non-low\n optical depth', 'Non-low\n residual']
    data_with_err = pd.DataFrame(cov_mat_with_err, columns=labels, index=labels)

    # Covariance matrix plot
    plot.close()
    fig, ax = plot.subplots(axwidth=5)
    m = ax.heatmap(
        data_with_err, cmap='ColdHot', vmin=-1, vmax=1, N=100,
        lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
        clip_on=False,  # turn off clipping so box edges are not cut in half
    )
    ax.format(title='Breakdown by altitude & type', titleweight='bold', alpha=0,
        xloc='bottom', yloc='left', #xloc='top', yloc='right',
        yreverse=True, linewidth=0, # ticklabelweight='bold',
        ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
    )

    fig_name = P(fig_dir, 'covariance_contribution_breakdown_with_err.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    # plot.show()

    '''
    labels = ['Low', 'Non-low']
    data_low_nonlow = pd.DataFrame(cov_mat_low_nonlow, columns=labels, index=labels)
    # plot.close()
    # fig, ax = plot.subplots(nrows=1, ncols=1, axwidth=4.5)
    # m = ax.heatmap(
    #     data_low_nonlow, cmap='ColdHot', vmin=-1, vmax=1, N=100,
    #     lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
    #     clip_on=False,  # turn off clipping so box edges are not cut in half
    # )
    # ax.format(title='Breakdown by altitude', titleweight='bold', alpha=0,
    #     xloc='bottom', yloc='left', #xloc='top', yloc='right',
    #     yreverse=True, linewidth=0, # ticklabelweight='bold',
    #     ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
    # )
    # fig_name = P(fig_dir, 'covariance_contribution_breakdown_by_altitude.pdf')
    # fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    # plot.show()

    plot.close()
    fig, axes = plot.subplots(nrows=1, ncols=2, axwidth=4.5, share=False)
    titles = ['Breakdown by altitude', 'Breakdown by altitude & type']
    dt_arr = [data_low_nonlow, data_with_err]
    for i, (ax, dt, title) in enumerate(zip(axes, dt_arr, titles)):
        m = ax.heatmap(
            dt, cmap='ColdHot', vmin=-1, vmax=1, N=100,
            lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
            clip_on=False,  # turn off clipping so box edges are not cut in half
        )
        ax.set_title('('+chr(97+i)+') '+title)
    
    axes.format(titleweight='bold', alpha=0,
            xloc='bottom', yloc='left', #xloc='top', yloc='right',
            yreverse=True, linewidth=0, # ticklabelweight='bold',
            ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
            #abc=True, abcstyle='(a)',
        )
    fig_name = P(fig_dir, 'covariance_contribution_breakdown_by_altitude_and_type.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    # plot.show()


    ################################################################################
    # ================== Breakdown into SW and LW cloud feedbacks =================
    ################################################################################

    var_names = ['sw_cld_tot', 'lw_cld_tot']
    var_labels = ['SW', 'LW']
    section_labels = ['Low', 'Non-low']
    labels = [s+' '+v for s in section_labels for v in var_labels]
    cov_mat_low_nonlow_sw_lw = cal_cov_matrix_for_cld_fbk_components(tbl, var_names)
    data_low_nonlow_sw_lw = pd.DataFrame(cov_mat_low_nonlow_sw_lw, columns=labels, index=labels)

    plot.close()
    fig, ax = plot.subplots(nrows=1, ncols=1, axwidth=4.5)
    m = ax.heatmap(
        data_low_nonlow_sw_lw, cmap='ColdHot', vmin=-1, vmax=1, N=100,
        lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
        clip_on=False,  # turn off clipping so box edges are not cut in half
    )
    ax.format(title='Breakdown by altitude and SW/LW', titleweight='bold', alpha=0,
        xloc='bottom', yloc='left', #xloc='top', yloc='right',
        yreverse=True, linewidth=0, # ticklabelweight='bold',
        ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
    )
    fig_name = P(fig_dir, 'covariance_contribution_breakdown_by_altitude_sw_lw.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

    var_names = ['sw_cld_amt', 'lw_cld_amt', 'sw_cld_alt', 'lw_cld_alt',
                 'sw_cld_tau',  'lw_cld_tau', 'sw_cld_err', 'lw_cld_err']
    var_labels = ['SW amt', 'LW amt', 'SW alt', 'LW alt',
                  'SW tau','LW tau', 'SW re', 'LW re']
    section_labels = ['Low', 'Non-low']
    labels = [s+'\n '+v for s in section_labels for v in var_labels]
    cov_mat_all_sw_lw = cal_cov_matrix_for_cld_fbk_components(tbl, var_names, with_err=True)
    data_all_sw_lw = pd.DataFrame(cov_mat_all_sw_lw, columns=labels, index=labels)

    plot.close()
    fig, ax = plot.subplots(nrows=1, ncols=1, axwidth=8)
    m = ax.heatmap(
        data_all_sw_lw, cmap='ColdHot', vmin=-1, vmax=1, N=100,
        lw=0.5, edgecolor='k', labels=True, #labels_kw={'weight': 'bold'},
        clip_on=False,  # turn off clipping so box edges are not cut in half
    )
    ax.format(title='Breakdown by altitude, type and SW/LW', titleweight='bold', alpha=0,
        xloc='bottom', yloc='left', #xloc='top', yloc='right',
        yreverse=True, linewidth=0, # ticklabelweight='bold',
        ytickmajorpad=4,  # the ytick.major.pad rc setting; adds extra space
    )
    fig_name = P(fig_dir, 'covariance_contribution_breakdown_by_altitude_and_type_sw_lw.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()

    '''
