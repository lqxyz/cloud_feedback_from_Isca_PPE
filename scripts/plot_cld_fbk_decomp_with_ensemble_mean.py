"""
This figure follows the Fig.1 of Zelinka et al, Insights from a refined 
    decomposition of cloud feedbacks, 2016, GRL, doi: 10.1002/2016GL069917
"""
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
    row_names = ['ALL', 'HI680', 'LO680']
    out_tbl_dict = {}
    for nn, row_nm in enumerate(row_names):
        out_tbl = np.zeros((N, 15))
        for i, exp in enumerate(exp_grps):
            file_nm = P(dt_dir, 'cld_fbk_decomp_v2_' + exp + '.csv')
            tbl = pd.read_csv(file_nm, index_col=0)
            out_tbl[i,0:5] = tbl.iloc[nn,0:5]
            out_tbl[i,5:10] = tbl.iloc[nn,5:10]
            out_tbl[i,10:15] = out_tbl[i,0:5] + out_tbl[i,5:10]
        col_names = [x for x in tbl.columns]
        col_names = col_names + [x.replace('lw', 'net') for x in col_names[0:5]]
        out_tbl = pd.DataFrame(data=out_tbl, index=exp_grps, columns=col_names)

        file_name = os.path.join(dt_dir, row_nm + '_cld_fbk_components.csv')
        out_tbl.to_csv(file_name, header=True, index=True, float_format=float_fmt)

        out_tbl_dict[row_nm] = out_tbl

    return out_tbl_dict

if __name__ == '__main__':
    P = os.path.join
    dt_dir = '../data'
    fig_dir = '../figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    markers = list(exp_tbl.iloc[:, 2])

    tbl_dict = get_cldfbk_component_table(exp_grps, float_fmt='%.4f', dt_dir=dt_dir)
    keys = ['ALL', 'HI680', 'LO680']
    titles = ['All clouds', r'Non-low clouds (CTP$\leq$680hPa)', 'Low clouds (CTP>680hPa)']

    plot.close()
    fig, axes = plot.subplots(nrows=3, ncols=1, aspect=(2.5, 1),sharey=False,  axwidth=4)
    xlim = [-0.5, 4.5]

    for ax, key, title in zip(axes, keys, titles):
        tbl = tbl_dict[key]
        
        row_names = tbl.index
        col_names = tbl.columns

        col_names_arr = [col_names.values[0:5], col_names.values[5:10], col_names.values[10:16]]
        colors =['indigo4', 'orange7', 'k']
        xlocs = [-0.25, 0.25, 0]

        # https://stackoverflow.com/questions/21285885/remove-line-through-marker-in-matplotlib-legend
        for col_nms, color, xloc in zip(col_names_arr, colors, xlocs):
            for i, col_nm in enumerate(col_nms):
                for row_nm, marker in zip(row_names, markers):
                    ax.plot(i + xloc, tbl[col_nm][row_nm], color=color, linestyle='None',
                        marker=marker, markersize=8, label=row_nm, clip_on=False)
                # Plot PPE mean
                ax.bar(i+xloc, tbl[col_nm].mean(), fill=False, edgecolor=color, width=0.5)
        ax.set_xlim(xlim)
        ax.set_ylabel('Wm$^{-2}$K$^{-1}$')
        ax.set_title(title, fontweight='bold')
        ax.plot(xlim, [0,0], 'k-', lw=0.5)
        #ax.plot(xlim, [-0.5,-0.5], ':', color='gray', lw=0.5)
        ax.plot(xlim, [0.5,0.5], ':', color='gray', lw=0.5)

    axes.format(grid=False, xlocator=[0,1,2,3,4], ylim=[-0.5, 1.1],
            xticklabels=['Total', 'Amount', 'Altitude', 'Optical depth', 'Residual'],
            xtickminor=False, ytickminor=False, abc=True, abcstyle='(a)')
            #xspineloc='bottom', yspineloc='left')

    legend_elements = []
    for exp_nm, marker in zip(exp_grps, markers):
        legend_elements.append(Line2D([0], [0], linestyle=None,
                        marker=marker, markersize=6, color='k', lw=0, label=exp_nm))
    fig.legend(legend_elements, loc='b', ncol=3)
    # Add LW, SW, Net with color
    axes[0].text(3.3, 0.9, 'LW', fontweight='bold', color=colors[0])
    axes[0].text(3.7, 0.9, 'Net', fontweight='bold', color=colors[2])
    axes[0].text(4.1, 0.9, 'SW', fontweight='bold', color=colors[1])

    #fig_name = P(fig_dir, 'global_mean_cld_fbk_decomp_scatter2.pdf')
    fig_name = P(fig_dir, 'global_mean_cld_fbk_decomp_scatter.pdf')
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
    #plot.show()
    print(fig_name, 'saved')
