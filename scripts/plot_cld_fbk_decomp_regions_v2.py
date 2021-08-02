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
from matplotlib.lines import Line2D
#from matplotlib.patches import Patch
from analysis_functions import get_unique_line_labels

def get_cldfbk_component_table(exp_grps, float_fmt='%.4f', dt_dir='./data'):
    # The first 5 cols for LW components, the mid 5 cols for SW components, 
    # and the last 5 cols for net components

    N = len(exp_grps)
    sections = ['ALL', 'HI680', 'LO680']
    row_names = []
    for reg in ['tropical', 'mid_lat', 'high_lat', 'global']:
        for sec in sections:
            row_names.append('_'.join([reg, sec]))

    out_tbl_dict = {}
    out_tbl = np.zeros((N, 15*len(row_names)))

    for i, exp in enumerate(exp_grps):
        file_nm = P(dt_dir, 'cld_fbk_decomp_v2_regional_mean_' + exp + '.csv')
        tbl = pd.read_csv(file_nm, index_col=0)

        for nn, row_nm in enumerate(row_names):
            out_tbl[i,nn*15:(nn+1)*15] = tbl.iloc[nn]
    
    col_names = ['_'.join([x, y]) for x in row_names for y in tbl.columns]
    out_tbl = pd.DataFrame(data=out_tbl, index=exp_grps, columns=col_names)

    file_name = os.path.join(dt_dir, 'cld_fbk_components_regions2.csv')
    out_tbl.to_csv(file_name, header=True, index=True, float_format=float_fmt)

    return out_tbl

if __name__ == '__main__':
    P = os.path.join
    dt_dir = '../data'
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    markers = list(exp_tbl.iloc[:, 2])

    tbl = get_cldfbk_component_table(exp_grps, float_fmt='%.4f', dt_dir=dt_dir)
    
    # plot the tropical, extropical and global mean SW low cloud feedbacks
    regions = ['tropical', 'mid_lat', 'high_lat', 'global'] #['tropical', 'extropical', 'global']
    #sec = 'LO680'
    sections = ['ALL', 'HI680', 'LO680']
    sec_labels = ['ALL', 'Non-low', 'Low' ]
    for sec, sec_labl in zip(sections, sec_labels):
        var_names = ['sw_cld_tot', 'sw_cld_amt', 'sw_cld_alt', 'sw_cld_tau',
                    'lw_cld_tot', 'lw_cld_amt', 'lw_cld_alt', 'lw_cld_tau', 
                    'net_cld_tot', 'net_cld_amt', 'net_cld_alt', 'net_cld_tau']
        titles = ['SW ' + sec_labl + ' total', 'SW ' + sec_labl + ' amount', 'SW ' + sec_labl + ' altitude', 'SW ' + sec_labl + ' optical depth',
                'LW ' + sec_labl + ' total', 'LW ' + sec_labl + ' amount', 'LW ' + sec_labl + ' altitude', 'LW ' + sec_labl + ' optical depth',
                'Net ' + sec_labl + ' total', 'Net ' + sec_labl + ' amount', 'Net ' + sec_labl + ' altitude', 'Net ' + sec_labl + ' optical depth',]
        row_names = tbl.index

        plot.close()
        fig, axes = plot.subplots(nrows=3, ncols=4, aspect=(1.5, 1), axwidth=1.8) #,sharey=True,  axwidth=4)
        xlim = [-0.5, len(regions)-0.5]

        lines = []
        for ax, var_nm, title in zip(axes, var_names, titles):
            col_names = []
            for reg in regions:
                var_name = '_'.join([reg, sec, var_nm])
                col_names.append(var_name)
            #print(col_names)
            for i, col_nm in enumerate(col_names):
                for j, (row_nm, marker) in enumerate(zip(row_names, markers)):
                    l = ax.plot(i, tbl[col_nm][row_nm],  linestyle='None', color='C'+str(j),
                        marker=marker, markersize=8, label=row_nm, clip_on=False)
                    lines.extend(l)

            ax.set_xlim(xlim)
            #ax.set_ylabel('Wm$^{-2}$K$^{-1}$')
            ax.set_title(title, fontweight='bold')
            ax.plot(xlim, [0,0], 'k-', lw=0.5)
            #ax.plot(xlim, [-0.5,-0.5], ':', color='gray', lw=0.5)
            #ax.plot(xlim, [0.5,0.5], ':', color='gray', lw=0.5)

        axes.format(ylabel='Wm$^{-2}$K$^{-1}$', grid=False, xlocator=[0,1,2,3], ylim=[-0.5, 1.1],
                xticklabels=['Tropics', 'Mid-lat', 'High-lat', 'Global'],
                xtickminor=False, ytickminor=False, abc=True, abcstyle='(a)')
                #xspineloc='bottom', yspineloc='left')
        new_lines, new_labels = get_unique_line_labels(lines)
        #axes[-1].legend(new_lines, new_labels, ncols=2)
        fig.legend(new_lines, new_labels, loc='b', ncols=6)

        fig_name = P(fig_dir, 'region_mean_cld_fbk_decomp_scatter_' + sec + '.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        #plot.show()
        print(fig_name, 'saved')
