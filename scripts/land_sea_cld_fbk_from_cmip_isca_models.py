import matplotlib
matplotlib.use('Agg')
import sys
import proplot as plot
import numpy as np
import pandas as pd
import json
import string
from scipy import stats
from pathlib import Path
# # https://stackoverflow.com/questions/15740682/
# # wrapping-long-y-labels-in-matplotlib-tight-layout-using-setp
# from textwrap import wrap
import warnings
warnings.simplefilter(action='ignore')
from analysis_functions import get_unique_line_labels


# Define unique markers for each model (with same symbol for centres)
def get_markers():
    markers_dict={}
    markers_dict['CanAM4']='v'
    markers_dict['CanESM2']='v'
    markers_dict['CanESM5']='v'
    markers_dict['HadGEM2-A']='^'
    markers_dict['HadGEM2-ES']='^'
    markers_dict['HadGEM3-GC31-LL']='^'
    markers_dict['MIROC-ESM']='<'
    markers_dict['MIROC-ES2L']='<'
    markers_dict['MIROC5']='>'
    markers_dict['MIROC6']='>'
    markers_dict['MRI-CGCM3']='X'
    markers_dict['MRI-ESM2-0']='X'
    markers_dict['CCSM4']='o'
    markers_dict['CESM2']='o'
    markers_dict['E3SM-1-0']='*'
    markers_dict['IPSL-CM5A-LR']='D'
    markers_dict['IPSL-CM6A-LR']='D'
    markers_dict['IPSL-CM5B-LR']='d'
    markers_dict['GFDL-CM4']='P'
    markers_dict['MPI-ESM-LR']='s'
    markers_dict['UKESM1-0-LL']='h'
    markers_dict['BCC-CSM2-MR']='H'
    markers_dict['CNRM-CM5']='8'
    markers_dict['CNRM-CM6-1']='8'

    return markers_dict

def get_colors():
    colors_dict={}
    colors_dict['CanAM4']='darkgray'
    colors_dict['CanESM2']='gray'
    colors_dict['CanESM5']='silver'
    colors_dict['HadGEM2-A']='indianred'
    colors_dict['HadGEM2-ES']='lightcoral'
    colors_dict['HadGEM3-GC31-LL']='brown'
    colors_dict['MIROC-ESM']='blue'
    colors_dict['MIROC-ES2L']='darkblue'
    colors_dict['MIROC5']='darkgreen'
    colors_dict['MIROC6']='green'
    colors_dict['MRI-CGCM3']='orange'
    colors_dict['MRI-ESM2-0']='wheat'
    colors_dict['CCSM4']='slateblue'
    colors_dict['CESM2']='purple'
    colors_dict['E3SM-1-0']='plum'
    colors_dict['IPSL-CM5A-LR']='olivedrab'
    colors_dict['IPSL-CM6A-LR']='yellowgreen'
    colors_dict['IPSL-CM5B-LR']='greenyellow'
    colors_dict['GFDL-CM4']='hotpink'
    colors_dict['MPI-ESM-LR']='lime'
    colors_dict['UKESM1-0-LL']='C0'
    colors_dict['BCC-CSM2-MR']='C1'
    colors_dict['CNRM-CM5']='C2'
    colors_dict['CNRM-CM6-1']='C3'

    return colors_dict

def get_reg_cld_fbk(fbk_dict, reg='tropical', land_or_sea='', hi_or_low='ALL'):
    """
    reg: ['tropical', 'mid', 'high', 'global']
    land_or_sea: ['', 'land', 'ocean', 'ocn_asc', 'ocn_dsc']
    hi_or_low: ['HI680', 'LO680', 'ALL']
    """

    if land_or_sea == '':
        land_sea_key = 'all'
    elif land_or_sea == 'land':
        land_sea_key = 'lnd'
    elif land_or_sea == 'ocean':
        land_sea_key = 'ocn'
    else:
        land_sea_key = land_or_sea

    cld_fbk_types = ['LWcld_tot', 'SWcld_tot', 'NETcld_tot']
    fbk_arr = []
    for cld_fbk_type in cld_fbk_types:
        if 'global' in reg.lower():
            fbk = fbk_dict['eq90'][land_sea_key][hi_or_low][cld_fbk_type]
        elif 'tropical' in reg.lower():
            fbk = fbk_dict['eq30'][land_sea_key][hi_or_low][cld_fbk_type]
        elif 'mid' in reg.lower():
            fbk = fbk_dict['eq60'][land_sea_key][hi_or_low][cld_fbk_type] - fbk_dict['eq30'][land_sea_key][hi_or_low][cld_fbk_type]
        else:
            fbk = fbk_dict['eq90'][land_sea_key][hi_or_low][cld_fbk_type] - fbk_dict['eq60'][land_sea_key][hi_or_low][cld_fbk_type]
        fbk_arr.append(fbk)

    # NET = fbk_dict['eq90']['all']['HI680']['NETcld_alt']
    # # 30-60 marine low cloud amount
    # NET = fbk_dict['eq60']['ocn']['LO680']['NETcld_amt'] - fbk_dict['eq30']['ocn']['LO680']['NETcld_amt']
    return np.array(fbk_arr)

def get_fbks(cld_fbks, ecs_dict, reg='tropical', land_or_sea='', hi_or_low='ALL'):
    # Load in all the json files and get assessed/unassessed feedbacks
    assessed = []
    models = []
    ripfs = []
    ECS = {}

    MODELS = list(cld_fbks.keys())

    N = len(MODELS)
    out_tbl = np.zeros((N, 3))

    for kk, mo in enumerate(MODELS):
        if mo == 'metadata':
            continue
        models.append(mo)
        RIPFS = list(cld_fbks[mo].keys())
        for ripf in RIPFS:
            # print(mo, ripf)
            ripfs.append(ripf)
            out_tbl[kk,:] = get_reg_cld_fbk(cld_fbks[mo][ripf], reg=reg, land_or_sea=land_or_sea, hi_or_low=hi_or_low)

            # Get the abrupt4xCO2 Gregory-derived ECS values
            if mo=='HadGEM2-A':
                mo2 = 'HadGEM2-ES'
            elif mo=='CanAM4':
                mo2 = 'CanESM2'
            else:
                mo2 = mo
            try:
                if mo2 in ecs_dict.keys():
                    if ripf in ecs_dict[mo2].keys():
                        ripf2 = ripf
                    else:
                        ripf2 = list(ecs_dict[mo2].keys())[0] # take the first available ripf
                        print(mo+': Using ECS from '+ripf2+' rather than '+ripf)
                try:
                    ecs = ecs_dict[mo2][ripf2]['ECS']
                except:
                    print('No ECS for '+mo2+'.'+ripf2)
                    ecs = np.nan
            except:
                print('ECS dict not a dict...')
                ecs = np.nan

            ECS[mo] = ecs

    col_names = ['LW cldfbk', 'SW cldfbk', 'net cldfbk']
    out_tbl = pd.DataFrame(data=out_tbl, index=MODELS, columns=col_names)

    return out_tbl, ECS

if __name__ == '__main__':
    #newmod = 'GFDL-CM4'

    # # Add markers for Isca runs
    # exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    # models = list(exp_tbl.iloc[:, 0])
    # exp_markers = list(exp_tbl.iloc[:, 2])
    # for mo, exp_marker in zip(models, exp_markers):
    #     markers_dict[mo] = exp_marker

    # #make_all_figs()

    #add_legend = True
    if len(sys.argv) == 2:
        add_legend = int(sys.argv[1]) == 1
    else:
        add_legend = True

    fig_dir = Path('../figs/')
    if not fig_dir.exists():
        fig_dir.mkdir()

    data_dir = Path('../inputs/zelinka_data')
    isca_data_dir = Path('../data/zelinka_data')

    ##################################################################
    # READ IN CLOUD FEEDBACK VALUES FOR CMIP5/6 and Isca
    ##################################################################
    file = data_dir / 'cmip5_amip4K_cld_fbks.json'
    with open(file, 'r') as f:
        cld_fbks5 = json.load(f)
    file = data_dir / 'cmip5_amip_cld_errs.json'
    with open(file, 'r') as f:
        cld_errs5 = json.load(f)
    file = data_dir / 'cmip6_amip-p4K_cld_fbks.json'
    with open(file, 'r') as f:
        cld_fbks6 = json.load(f)
    file = data_dir / 'cmip6_amip_cld_errs.json'
    with open(file, 'r') as f:
        cld_errs6 = json.load(f)

    # For Isca
    isca_cld_fbk_fn = isca_data_dir / 'isca_cld_fbks.json'
    isca_cld_err_fn = isca_data_dir / 'isca_cld_errs.json'
    with open(isca_cld_fbk_fn, 'r') as f:
        cld_fbks_isca = json.load(f)
    with open(isca_cld_err_fn, 'r') as f:
        cld_errs_isca = json.load(f)

    ##################################################################
    # READ IN GREGORY ECS VALUES DERIVED IN ZELINKA ET AL (2020) GRL #
    ##################################################################
    with open(data_dir / 'cmip56_forcing_feedback_ecs.json', 'r') as f:
        ecs = json.load(f)
    ecs_dict5 = ecs['CMIP5']
    ecs_dict6 = ecs['CMIP6']

    isca_forcing_ecs_fn = isca_data_dir / 'isca_forcing_ECS.json'
    with open(isca_forcing_ecs_fn, 'r') as f:
        ecs_dict_isca = json.load(f)['isca']

    cld_fbk_tbl5, ECS5 = get_fbks(cld_fbks5, ecs_dict5, reg='tropical', land_or_sea='', hi_or_low='ALL')
    cld_fbk_tbl6, ECS6 = get_fbks(cld_fbks6, ecs_dict6, reg='tropical', land_or_sea='', hi_or_low='ALL')
    cld_fbk_tbl_isca, ECS_isca = get_fbks(cld_fbks_isca, ecs_dict_isca, reg='tropical', land_or_sea='', hi_or_low='ALL')
    fbk_tbl = cld_fbk_tbl5.append(cld_fbk_tbl6).append(cld_fbk_tbl_isca)
    ECS_dict = {**ECS5, **ECS6, **ECS_isca}

    markers_dict = get_markers()
    colors_dict = get_colors()

    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exp_markers = list(exp_tbl.iloc[:, 2])
    for exp_grp, exp_marker in zip(exp_grps, exp_markers):
        markers_dict[exp_grp] = exp_marker

    models = ECS_dict.keys()
    # models.extend(exp_grps)

    for mo in ecs_dict5.keys():
        if mo == 'HadGEM2-ES':
            mo2 = 'HadGEM2-A'
        elif mo == 'CanESM2':
            mo2 = 'CanAM4'
        else:
            mo2 = mo
        colors_dict[mo2] = 'C0' #'lightskyblue'
    for mo in ecs_dict6.keys():
        colors_dict[mo] = 'C1' #'lightsalmon'
    for mo in exp_grps:
        colors_dict[mo] = 'C2' #'palegreen'

    ##reg: ['tropical', 'mid', 'high', 'global']
    # land_or_sea_arr = ['', 'land', 'ocean']
    # hi_or_low_arr = ['HI680', 'LO680', 'ALL']
    land_or_sea_arr = ['', 'ocean']
    hi_or_low_arr = ['ALL']

    titles1 =['Global', r'30$^\circ$S$-$30$^\circ$N', r'$30-60^\circ$', r'$60-90^\circ$']
    regions = ['global', 'tropical', 'mid', 'high']

    # Add shading of Sherwood et al 2020 baseline PDF of ECS
    ecs5, ecs17, ecs83, ecs95 = 2.3, 2.6, 3.9, 4.7 # K

    for land_or_sea in land_or_sea_arr:
        for hi_or_low in hi_or_low_arr:
            if land_or_sea == '':
                land_or_sea1 = 'all'
                land_or_sea2 = ''
            else:
                land_or_sea1 = land_or_sea
                land_or_sea2 = land_or_sea + ', '
            fig_str = land_or_sea1 + '_' + hi_or_low
            print(fig_str)
            if hi_or_low == 'ALL':
                if land_or_sea2 == '':
                    titles = titles1
                else:
                    titles = [t + ' (' + land_or_sea2.replace(', ', '') +')' for t in titles1]
            else:
                titles = [t + ' (' + land_or_sea2 + hi_or_low +')' for t in titles1]
            fn = 'ECS_vs_cldfbk_' + fig_str + '_cmip56_isca.pdf'
            fig_name = fig_dir / fn

            lines = []
            plot.close()
            fig, axes_all = plot.subplots(nrows=4, ncols=4,
                    aspect=[1.5, 1], axwidth=1.6, share=False)

            for i in range(len(regions)):
                cld_fbk_tbl5, ECS5 = get_fbks(cld_fbks5, ecs_dict5, reg=regions[i],
                    land_or_sea=land_or_sea, hi_or_low=hi_or_low)
                cld_fbk_tbl6, ECS6 = get_fbks(cld_fbks6, ecs_dict6, reg=regions[i],
                    land_or_sea=land_or_sea, hi_or_low=hi_or_low)
                cld_fbk_tbl_isca, ECS_isca = get_fbks(cld_fbks_isca, ecs_dict_isca, reg=regions[i],
                    land_or_sea=land_or_sea, hi_or_low=hi_or_low)
                fbk_tbl = cld_fbk_tbl5.append(cld_fbk_tbl6).append(cld_fbk_tbl_isca)
                ECS_dict = {**ECS5, **ECS6, **ECS_isca}

                net_fbk_dict = {}
                sw_fbk_dict = {}
                ecs_dict = {}
                for mo in models:
                    net_fbk_dict[mo] = fbk_tbl['net cldfbk'][mo]
                    sw_fbk_dict[mo] = fbk_tbl['SW cldfbk'][mo]
                    ecs_dict[mo] = ECS_dict[mo]

                ylim = [1.5, 6.5]
                axes = axes_all[:,i]

                models_list = [exp_grps, ECS5.keys(), ECS6.keys(), models]
                model_name_list = ['Isca', 'CMIP5', 'CMIP6', 'ALL']
        
                for nn, (ax, i_models, i_mod_name) in enumerate(zip(axes, models_list, model_name_list)):
                    i_fbk_arr = []
                    i_ecs_arr = []
                    for mo in i_models:
                        fbk = net_fbk_dict[mo]
                        ecs = ecs_dict[mo]
                        i_fbk_arr.append(fbk)
                        i_ecs_arr.append(ecs)
                        l = ax.plot(fbk, ecs, linestyle='None', marker=markers_dict[mo],
                                color=colors_dict[mo], markersize=5, label=mo, clip_on=False)
                        lines.extend(l)

                    xlim = [np.min(i_fbk_arr)-0.05, np.max(i_fbk_arr)+0.05]
                    result = stats.linregress(i_fbk_arr, i_ecs_arr)
                    x2 = np.linspace(min(xlim), max(xlim), 100)
                    y2 = result.slope * x2 + result.intercept
                    ax.plot(x2, y2, '--', color='k', linewidth=1)
                    # Add expert assessed ECS
                    ax.fill_between(x2, ecs17, ecs83, color='C0', alpha=0.2)
                    ax.fill_between(x2, ecs83, ecs95, color='C1', alpha=0.15)
                    ax.fill_between(x2, ecs5, ecs17, color='C1', alpha=0.15)
                    xloc = min(xlim) + (max(xlim)-min(xlim)) * 0.2
                    yloc = min(ylim) + (max(ylim)-min(ylim)) * 0.84
                    if result.pvalue < 0.05:
                        ax.text(xloc, yloc, 'R = %.2f*'%result.rvalue)
                    else:
                        ax.text(xloc, yloc, 'R = %.2f'%result.rvalue)
                    if nn == 3:
                        ax.set_xlabel(r'Net cloud feedback (Wm$^{-2}$K$^{-1}$)')
                    if i == 0:
                        ax.set_ylabel(i_mod_name + ', ECS (K)')
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if nn == 0:
                        ax.set_title(titles[i])

            axes_all.format(abc=True, abcloc='ul', abcstyle='(a)',
                    xtickminor=False, yminorlocator=1)#ytickminor=False) #, ylabel='ECS (K)')
            new_lines, new_labels = get_unique_line_labels(lines)
            if add_legend:
                fig.legend(new_lines, new_labels, loc='b', ncols=6)
            else:
                fig_name = fig_name.replace('.pdf', '_no_legend.pdf')

            fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
            fig.savefig(Path(str(fig_name).replace('.pdf', '.png')), bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=200)
            #plot.show()
            print(fig_name, 'saved')
    
    # ============================================================================ #
    # Plot ECS vs ascending/subsiding ocn region
    #reg: ['tropical', 'mid', 'high', 'global']
    land_or_sea_arr = ['ocn_asc', 'ocn_dsc']
    #hi_or_low_arr = ['HI680', 'LO680', 'ALL']
    hi_or_low_arr = ['ALL']

    for hi_or_low in hi_or_low_arr:
        fig_str = hi_or_low
        fn = 'ECS_vs_cldfbk_' + fig_str + '_cmip56_isca_tropical_asc_dsc.pdf'
        fig_name = fig_dir / fn

        region = 'tropical'
        lines = []
        plot.close()
        fig, axes_all = plot.subplots(nrows=2, ncols=4,
                aspect=[1.5, 1], axwidth=1.6, share=False)

        for i, land_or_sea in enumerate(land_or_sea_arr):
            cld_fbk_tbl5, ECS5 = get_fbks(cld_fbks5, ecs_dict5, reg=region,
                land_or_sea=land_or_sea, hi_or_low=hi_or_low)
            cld_fbk_tbl6, ECS6 = get_fbks(cld_fbks6, ecs_dict6, reg=region,
                land_or_sea=land_or_sea, hi_or_low=hi_or_low)
            cld_fbk_tbl_isca, ECS_isca = get_fbks(cld_fbks_isca, ecs_dict_isca, reg=region,
                land_or_sea=land_or_sea, hi_or_low=hi_or_low)
            fbk_tbl = cld_fbk_tbl5.append(cld_fbk_tbl6).append(cld_fbk_tbl_isca)
            ECS_dict = {**ECS5, **ECS6, **ECS_isca}

            net_fbk_dict = {}
            sw_fbk_dict = {}
            ecs_dict = {}
            for mo in models:
                net_fbk_dict[mo] = fbk_tbl['net cldfbk'][mo]
                sw_fbk_dict[mo] = fbk_tbl['SW cldfbk'][mo]
                ecs_dict[mo] = ECS_dict[mo]

            ylim = [1.5, 6.5]
            axes = axes_all[i,:]

            models_list = [exp_grps, ECS5.keys(), ECS6.keys(), models]
            model_name_list = ['Isca', 'CMIP5', 'CMIP6', 'ALL']
    
            for nn, (ax, i_models, i_mod_name) in enumerate(zip(axes, models_list, model_name_list)):
                i_fbk_arr = []
                i_ecs_arr = []
                for mo in i_models:
                    fbk = net_fbk_dict[mo]
                    ecs = ecs_dict[mo]
                    i_fbk_arr.append(fbk)
                    i_ecs_arr.append(ecs)
                    l = ax.plot(fbk, ecs, linestyle='None', marker=markers_dict[mo],
                            color=colors_dict[mo], markersize=5, label=mo, clip_on=False)
                    lines.extend(l)

                xlim = [np.min(i_fbk_arr)-0.05, np.max(i_fbk_arr)+0.05]
                result = stats.linregress(i_fbk_arr, i_ecs_arr)
                x2 = np.linspace(min(xlim), max(xlim), 100)
                y2 = result.slope * x2 + result.intercept
                ax.plot(x2, y2, '--', color='k', linewidth=1)
                ax.fill_between(x2, ecs17, ecs83, color='C0', alpha=0.2)
                ax.fill_between(x2, ecs83, ecs95, color='C1', alpha=0.15)
                ax.fill_between(x2, ecs5, ecs17, color='C1', alpha=0.15)
                xloc = min(xlim) + (max(xlim)-min(xlim)) * 0.2
                yloc = min(ylim) + (max(ylim)-min(ylim)) * 0.84
                if result.pvalue < 0.05:
                    ax.text(xloc, yloc, 'R = %.2f*'%result.rvalue)
                else:
                    ax.text(xloc, yloc, 'R = %.2f'%result.rvalue)
                if i == 1:
                    ax.set_xlabel(r'Net cloud feedback (Wm$^{-2}$K$^{-1}$)')
                if i == 0 and nn == 0:
                    ax.set_ylabel('Ascedning, ECS (K)')
                elif i == 1 and nn == 0:
                    ax.set_ylabel('Descedning, ECS (K)')
                else:
                    ax.set_ylabel('')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if i == 0:
                    ax.set_title(i_mod_name)

        axes_all.format(abc=True, abcloc='ul', abcstyle='(a)',
                xtickminor=False, yminorlocator=1) #ytickminor=False) #, ylabel='ECS (K)')
        new_lines, new_labels = get_unique_line_labels(lines)
        if add_legend:
            fig.legend(new_lines, new_labels, loc='b', ncols=6)
        else:
            fig_name = fig_name.replace('.pdf', '_no_legend.pdf')

        fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1, transparent=False)
        fig.savefig(Path(str(fig_name).replace('.pdf', '.png')), bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=200)
        #plot.show()
        print(fig_name, 'saved')
