import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import proplot as plot
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
from matplotlib.patches import Patch
from analysis_functions import get_unique_line_labels

if __name__ == '__main__':
    P = os.path.join
    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    markers = list(exp_tbl.iloc[:, 2])

    '''
    tbl_names = ['toa_sw_cre', 'toa_lw_cre', 'toa_net_cre',
                'low_cld_amt', 'mid_cld_amt', 'high_cld_amt', 'tot_cld_amt',
                'cwp', 'low_cwp', 'mid_cwp', 'high_cwp',
                #'tauisccp',
                ]
    tbl_titles = ['SWCRE', 'LWCRE', 'Net CRE',
                 'low cldamt', 'mid cldamt', 'high cldamt', 'total cldamt',
                 'CWP', r'CWP$_{low}$', r'CWP$_{mid}$', r'CWP$_{high}$',
                 #'optical depth'
                ]
    units_arr1 = [r'W m$^{-2}$'] * 3 + ['%'] * 4 + [r'g m$^{-2}$'] * 4 #+ ['']
    '''
    tbl_names = ['low_cld_amt']
    tbl_titles = ['low cldamt']
    units_arr1 = ['%']

    proxy = sys.argv[1].upper() # ELF, or EIS
    #proxy = 'EIS'
    cld_type_arr = ['All', 'Cu', 'Sc']
    for tbl_nm, tbl_title, units in zip(tbl_names, tbl_titles, units_arr1):
        print(proxy, tbl_nm)
        tbl_dict = {}
        keys = ['Var per CCF sigma', 'CCF sigma per K', 'Var per K']
        for key in keys:
            tbl_dict[key] = {}
        real_chg_tbl = {}
        for cld_type in cld_type_arr:
            if 'all' in cld_type.lower():
                dt_dir = '../data/CCF_analysis_metric'
                file_nms = [tbl_nm+'_CCF_ctrl_'+proxy+'.csv',
                        'CCF_change_perK_'+proxy+'.csv',
                        tbl_nm+'_chg_per_K_'+proxy+'.csv']
                file_real_nm = tbl_nm + '_chg_per_K_real_'+proxy+'.csv'
            else:
                dt_dir = '../data/CCF_analysis/cu_sc_metric'
                file_nms = [tbl_nm+'_CCF_ctrl_'+cld_type+'_'+proxy+'.csv',
                        'CCF_change_perK_'+cld_type+'_'+proxy+'.csv',
                        tbl_nm+'_chg_per_K_'+cld_type+'_'+proxy+'.csv']
                file_real_nm = tbl_nm + '_chg_per_K_real_'+cld_type+'_'+proxy+'.csv'
            for key, file_nm in zip(keys, file_nms):
                tbl = pd.read_csv(P(dt_dir, file_nm), index_col=0)
                tbl_dict[key][cld_type] = tbl
            real_chg_tbl[cld_type] = pd.read_csv(P(dt_dir, file_real_nm), index_col=0)

        titles = [tbl_title + ' sensitivity to CCFs',
                'Temperature-mediated change in CCFs',
                'CCF-driven '+ tbl_title +' changes']
        units_arr = [units + r' $\sigma^{-1}$', r'$\sigma$ K$^{-1}$ ', units + r' K$^{-1}$']

        plt.close()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4.5,6))

        box_colors = ['palegreen', 'tan', 'lightblue']
        lines = []
        box_width = 0.4
        xspace = box_width * len(cld_type_arr) + 0.2

        # https://matplotlib.org/stable/gallery/statistics/boxplot.html
        flierprops = dict(markeredgecolor='gray', marker='o', markersize=2) # markerfacecolor=None,
        boxprops = dict(linewidth=0.5, color='k') #gray')
        medianprops = dict(linestyle='--', linewidth=0.5, color='k') #boxprops
        whiskerprops = dict(linewidth=0.5, color='k')
        capprops = whiskerprops
        #xlim = [-2.4, 0.5+(3*box_width+0.2)*5]

        for ff, (ax, key, title, units) in enumerate(zip(axes, keys, titles, units_arr)):
            for cc, (cld_type, box_color) in enumerate(zip(cld_type_arr, box_colors)):
                xticks = []
                tbl = tbl_dict[key][cld_type]
                row_names = tbl.index
                col_names = tbl.columns
                '''
                xlocs = [-box_width, 0, box_width]
                for i, col_nm in enumerate(col_names):
                    l = ax.boxplot(tbl[col_nm], positions=[2*i+xlocs[cc]], patch_artist=True, label=)
                    lines.extend(l)
                    for patch, color in zip(l['boxes'], [box_color]):
                        patch.set_facecolor(color)
                '''
                xlocs = []
                labels = []
                colors = []
                xlocs1 = [-box_width, 0, box_width]

                for i, col_nm in enumerate(col_names):
                    if i == 0:
                        x = 1
                    else:
                        x = x + xspace #2 * i + 1
                    xticks.append(x)
                    xlabels1 = ['', col_nm, '']
                    xlocs.append(x + xlocs1[cc])
                    labels.append(xlabels1[cc])
                    colors.append(box_color)
                widths = [box_width] * len(col_names)
                b = ax.boxplot(tbl.T, positions=xlocs, widths=widths, patch_artist=True,
                        labels=labels, boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
                for patch, color in zip(b['boxes'], colors):
                    patch.set_facecolor(color)

                if 'Var per K' in key:
                    # == For the predicted changes == #
                    predicted = np.zeros(len(row_names))
                    for kk, row_nm  in enumerate(row_names):
                        for i, col_nm in enumerate(col_names):
                            predicted[kk] = predicted[kk] + tbl[col_nm][row_nm]
                    x1 = 1 - xspace
                    xticks.insert(0, x1)
                    xlocs = [x1 + xlocs1[cc]]
                    xlabels1 = ['', 'Predicted', '']
                    labels = [xlabels1[cc]]
                    b = ax.boxplot(predicted, positions=xlocs, widths=[box_width], patch_artist=True,
                            labels=labels, boxprops=boxprops, medianprops=medianprops,
                            whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
                    for patch, color in zip(b['boxes'], colors):
                        patch.set_facecolor(color)

                    # == For the actual changes == #
                    x2 = x1 - xspace
                    xticks.insert(0, x2)
                    xlocs = [x2 + xlocs1[cc]]
                    xlabels1 = ['', 'Sum', '']
                    labels = [xlabels1[cc]]
                    b = ax.boxplot(real_chg_tbl[cld_type].T, positions=xlocs, widths=[box_width],
                        patch_artist=True, labels=labels, boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
                    for patch, color in zip(b['boxes'], colors):
                        patch.set_facecolor(color)
                    ax.axvline(x=x1+(1-x1)/2, linewidth=0.5, linestyle='--', color='gray')

            ax.xaxis.set_tick_params(which='minor', bottom=False)
            #ax.set_xlim(xlim)
            ax.set_ylabel(units)
            ax.set_title('('+chr(97+ff)+') '+title) #, fontweight='bold')
            
            if len(col_names)>2:
                xticklabels1 = ['Actual', 'Predicted', 'SST', proxy, r'$T_{adv}$', r'RH$_{700}$', r'$\omega_{700}$']
            else:
                xticklabels1 = ['Actual', 'Predicted', 'SST', proxy] #, r'$T_{adv}$', r'RH$_{700}$', r'$\omega_{700}$']
            if ff == 2:
                xticklabels = xticklabels1[0:]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            else:
                xticklabels = xticklabels1[2:]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            ax.grid(b=False)
        for ax in axes:
            xlim = [min(xticks) - xspace / 2, max(xticks) + xspace / 2]
            ax.set_xlim(xlim)
            ax.plot(xlim, [0,0], ':', color='gray', lw=0.5)
        legend_elements = []
        for cld_type, color in zip(cld_type_arr, box_colors):
            legend_elements.append(Patch(facecolor=color, edgecolor=color, label=cld_type))
        # Need to add handles=xxx
        axes[0].legend(handles=legend_elements, loc='upper left', ncol=1)
        fig.tight_layout()
        fig_name = P(fig_dir, 'CCF_analysis_for_diff_cld_types_'+tbl_nm+'_'+proxy+'.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1)
        #plt.show()
        print(fig_name, 'saved')
