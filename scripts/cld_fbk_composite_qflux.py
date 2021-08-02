from __future__ import print_function
from __future__ import division
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import warnings
warnings.simplefilter(action='ignore') #, category=FutureWarning)
import proplot as plot
from analysis_functions import get_unique_line_labels


if __name__ == '__main__':
    P = os.path.join

    fig_dir = '../figs/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    dt_dir = '../data/qflux_area_weight_cld_fbk'

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exp_names = exp_grps
    #============ Read binned data ===========#
    land_sea_mask = 'ocean'
    grp_time = None
    s_lat = -30
    n_lat = 30

    '''
    bin_names = ['lts', 'omega500', 'eis', 'ELF',
                 'lts_percentile', 'omega500_percentile',
                 'eis_percentile', 'ELF_percentile',
                 ]
    xlabel_arr = ['LTS (K)', '$\omega_{500}$ (hPa day${^{-1}}$)', 'EIS (K)', 'ELF',
                'LTS percentile (%)', '$\omega_{500}$  percentile (%)',
                'EIS percentile (%)', 'ELF percentile (%)']
    '''

    bin_names = ['omega500_percentile',]
    xlabel_arr = ['$\omega_{500}$  percentile (%)',]

    plot.rc['cycle'] = 'bmh'
    colors = ['k', 'b'] + ['C'+str(i) for i in range(0,11)]

    for bin_nm, xlabel in zip(bin_names, xlabel_arr):
        print(bin_nm)
        dt_arr = []
        for file_id in exp_names:
            dt_fn = '_'.join(filter(None, ['ds_bin', bin_nm, file_id, grp_time, land_sea_mask, '3d']))
            ds = xr.open_dataset(P(dt_dir, dt_fn + '.nc'), decode_times=False)
            dt_arr.append(ds)
        
        bins = dt_arr[0].bin
        # ========================================================== #
        print('Plot cloud feedback composite...')
        var_names = ['ALL_net_cld_tot', 'ALL_sw_cld_tot', 'ALL_lw_cld_tot']
        titles = ['Net cloud feedback', 'Shortwave cloud feedback', 'Longwave cloud feedback']

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=2, aspect=(1.5,1), sharey=False, sharex=False) #, sharex=False, sharey=False) # axwidth=4)
        axes[-1].set_axis_off()
        lines = []
        for ax, varnm, title in zip(axes[:-1], var_names, titles):
            N = len(exp_names)
            for i in range(N):
                dt = dt_arr[i][varnm] # * dt_arr[i].pdf * np.mean(np.diff(dt_arr[i].bin))
                if grp_time is not None:
                    time_axis = 0
                    dt_mean = np.nanmean(dt, axis=time_axis)
                    dt_std = np.nanstd(dt, axis=time_axis)
                else:
                    dt_mean = dt
                l = ax.plot(bins, dt_mean, '-x', color=colors[i], label=exp_names[i])
                lines.extend(l)
            ax.set_title(title)
            ax.set_ylabel('Wm$^{-2}$K$^{-1}$')

        axes[:-1].format(xlabel=xlabel, abc=True, abcstyle='(a)') #, abcloc='ul')
        new_lines, new_labels = get_unique_line_labels(lines)
        #axes[-1].legend(new_lines, new_labels, ncols=2)
        axes[-1].legend(new_lines, new_labels, ncols=2, loc='c')

        fig_name = P(fig_dir, 'cld_fbk_composited_by_'+bin_nm+'.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        # plot.show()

        ##  Plot low cloud feedback
        print('Plot low cloud feedback composite...')
        var_names = ['LO680_net_cld_tot', 'LO680_net_cld_amt', 
                    'LO680_net_cld_alt', 'LO680_net_cld_tau']
        titles = ['Total low cloud feedback', 'Low cloud amount feedback',
                  'Low cloud altitude feedback', 'Low cloud optical depth feedback']

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=2, aspect=(1.5,1)) #, sharex=False) #, sharey=False,  sharex=False, sharey=False) # axwidth=4)
        #axes[-1].set_axis_off()
        lines = []
        for ax, varnm, title in zip(axes, var_names, titles):
            N = len(exp_names)
            for i in range(N):
                dt = dt_arr[i][varnm] # * dt_arr[i].pdf * np.mean(np.diff(dt_arr[i].bin))
                if grp_time is not None:
                    time_axis = 0
                    dt_mean = np.nanmean(dt, axis=time_axis)
                    dt_std = np.nanstd(dt, axis=time_axis)
                else:
                    dt_mean = dt
                l = ax.plot(bins, dt_mean, '-x', color=colors[i], label=exp_names[i])
                lines.extend(l)
            ax.set_title(title)
            if 'omega500_percentile' in bin_nm:
                ax.set_ylim([-0.7, 1.5])
            ax.set_ylabel('Wm$^{-2}$K$^{-1}$')

        axes.format(xlabel=xlabel, abc=True, abcstyle='(a)', abcloc='ul',
                    ylabel='Wm$^{-2}$K$^{-1}$')
        new_lines, new_labels = get_unique_line_labels(lines)
        #axes[-1].legend(new_lines, new_labels, ncols=2)
        fig.legend(new_lines, new_labels, ncols=4, loc='b')

        fig_name = P(fig_dir, 'low_cld_fbk_composited_by_'+bin_nm+'.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)

        '''
        ##  Plot high cloud feedback
        print('Plot high cloud feedback composite...')
        var_names = ['HI680_net_cld_tot', 'HI680_net_cld_amt', 
                    'HI680_net_cld_alt', 'HI680_net_cld_tau']
        titles = ['Total high cloud feedback', 'high cloud amount feedback',
                  'High cloud altitude feedback', 'High cloud optical depth feedback']

        plot.close()
        fig, axes = plot.subplots(nrows=2, ncols=2, aspect=(1.5,1)) #, sharex=False) #, sharey=False,  sharex=False, sharey=False) # axwidth=4)
        #axes[-1].set_axis_off()
        lines = []
        for ax, varnm, title in zip(axes, var_names, titles):
            N = len(exp_names)
            for i in range(N):
                dt = dt_arr[i][varnm] # * dt_arr[i].pdf * np.mean(np.diff(dt_arr[i].bin))
                if grp_time is not None: time_axis = 0
                    dt_mean = np.nanmean(dt, axis=time_axis)
                    dt_std = np.nanstd(dt, axis=time_axis)
                else:
                    dt_mean = dt
                l = ax.plot(bins, dt_mean, '-x', color=colors[i], label=exp_names[i])
                lines.extend(l)
            ax.set_title(title)
            # if 'omega500_percentile' in bin_nm:
            #     ax.set_ylim([-0.7, 1.5])
            ax.set_ylabel('Wm$^{-2}$K$^{-1}$')

        axes.format(xlabel=xlabel, abc=True, abcstyle='(a)', abcloc='ul',
                    ylabel='Wm$^{-2}$K$^{-1}$')
        new_lines, new_labels = get_unique_line_labels(lines)
        #axes[-1].legend(new_lines, new_labels, ncols=2)
        fig.legend(new_lines, new_labels, ncols=4, loc='b')

        fig_name = P(fig_dir, 'high_cld_fbk_composited_by_'+bin_nm+'.pdf')
        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        '''
    print('Done')
