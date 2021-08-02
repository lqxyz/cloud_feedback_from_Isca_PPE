"""Make figures comparing GCM cloud feedback components to 
    expert assessed values from Sherwood et al (2020)
"""
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.use('Agg')
#import proplot as plot
import numpy as np
import pandas as pd
from scipy import stats
import json
import string
from datetime import date
from pathlib import Path
# https://stackoverflow.com/questions/15740682/
# wrapping-long-y-labels-in-matplotlib-tight-layout-using-setp
from textwrap import wrap

Nbar = 3
HEIGHT = Nbar / (Nbar + 2) #0.45
print('Height=', HEIGHT)

fig_dir = Path('../figs/')
if not fig_dir.exists():
    fig_dir.mkdir()
data_dir = Path('../inputs/zelinka_data')

#######################################################
########### DEFINE COLORS FOR ECS COLORBAR ############
#######################################################
cmap = pl.cm.RdBu_r  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.arange(2.5,6.5,0.5)#[0,1.5,2.5,3.5,4.5,6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# 5,17,83,95 percentiles of the Baseline posterior ECS from 
# Table 10 (p. 67) of Sherwood et al (2020):
wcrp_bounds = [1.5, 2.3, 2.6, 3.9, 4.7, 6]
wcrp_cmap = mpl.colors.ListedColormap(['blue', 'cyan', 'white', 'orange', 'red'])
wcrp_norm = mpl.colors.BoundaryNorm(wcrp_bounds, wcrp_cmap.N)

BOUNDS0 = bounds
CMAP0 = cmap
NORM0 = norm

BOUNDS = wcrp_bounds
CMAP = wcrp_cmap
NORM = wcrp_norm

# Define unique markers for each model (with same symbol for centres)
MARK={}
MARK['CanAM4']='v'
MARK['CanESM2']='v'
MARK['CanESM5']='v'
MARK['HadGEM2-A']='^'
MARK['HadGEM2-ES']='^'
MARK['HadGEM3-GC31-LL']='^'
MARK['MIROC-ESM']='<'
MARK['MIROC-ES2L']='<'
MARK['MIROC5']='>'
MARK['MIROC6']='>'
MARK['MRI-CGCM3']='X'
MARK['MRI-ESM2-0']='X'
MARK['CCSM4']='o'
MARK['CESM2']='o'
MARK['E3SM-1-0']='*'
MARK['IPSL-CM5A-LR']='D'
MARK['IPSL-CM6A-LR']='D'
MARK['IPSL-CM5B-LR']='d'
MARK['GFDL-CM4']='P'
MARK['MPI-ESM-LR']='s'
MARK['UKESM1-0-LL']='h'
MARK['BCC-CSM2-MR']='H'
MARK['CNRM-CM5']='8'
MARK['CNRM-CM6-1']='8'


def get_expert_assessed_fbks():
    # ============== WCRP ASSESSMENT VALUES ============== #
    # Table 1 of Sherwood et al, 2020, Review of Geophysics
    fbk_names = [
        'High-Cloud Altitude',
        'Tropical Marine Low-Cloud',
        'Tropical Anvil Cloud Area',
        'Land Cloud Amount',
        'Middle Latitude Marine Low-Cloud Amount',
        'High Latitude Low-Cloud Optical Depth',
        'Implied Unassessed Cloud Feedbacks',
        'Sum of Assessed Cloud Feedbacks',
        'Total Cloud Feedback']

    expert_hi_alt,   err_expert_hi_alt   =  0.20, 0.10
    expert_trop_lo,  err_expert_trop_lo  =  0.25, 0.16
    expert_anvil,    err_expert_anvil    = -0.20, 0.20
    expert_land,     err_expert_land     =  0.08, 0.08
    expert_mid_amt,  err_expert_mid_amt  =  0.12, 0.12
    expert_extr_tau, err_expert_extr_tau =  0.00, 0.10
    expert_sum_assessed = expert_hi_alt + expert_trop_lo + expert_anvil + \
                          expert_land + expert_mid_amt + expert_extr_tau
    expert_imply_not_assessed = 0
    expert_cld_fbks = np.array([expert_hi_alt, expert_trop_lo, expert_anvil, expert_land,
                                expert_mid_amt, expert_extr_tau, expert_imply_not_assessed,
                                expert_sum_assessed, expert_sum_assessed])

    err_expert_sum_assessed = np.sqrt(err_expert_hi_alt**2 + err_expert_trop_lo**2 +
                                      err_expert_anvil**2 + err_expert_land**2 +
                                      err_expert_mid_amt**2 + err_expert_extr_tau**2)
    err_expert_imply_not_assessed = 0
    err_expert_cld_fbks = np.array([err_expert_hi_alt, err_expert_trop_lo, err_expert_anvil,
                                    err_expert_land, err_expert_mid_amt, err_expert_extr_tau,
                                    err_expert_imply_not_assessed, err_expert_sum_assessed,
                                    err_expert_sum_assessed])

    return (expert_cld_fbks, err_expert_cld_fbks, fbk_names)

def get_fbks(cld_fbks, cld_errs, ecs_dict):
    # Load in all the json files and get assessed/unassessed feedbacks
    assessed = []
    models = []
    ripfs = []
    ECS = []

    cnt = -1
    MODELS = list(cld_fbks.keys())
    for mo in MODELS:
        if mo == 'metadata':
            continue
        cnt += 1
        models.append(mo)
        RIPFS = list(cld_fbks[mo].keys())
        for ripf in RIPFS:
            ripfs.append(ripf)
            afbks = get_gcm_assessed_fbks(cld_fbks[mo][ripf])
            ufbks, ufbk_names = get_unassessed_fbks(cld_fbks[mo][ripf])
            if cnt == 0:
                assessed = afbks
                unassessed = ufbks
            else:
                assessed = np.row_stack((assessed, afbks))
                unassessed = np.row_stack((unassessed, ufbks))

            # get the error metrics (only defined for amip versions):
            if mo == 'HadGEM2-ES':
                mo2 = 'HadGEM2-A'
            elif mo == 'CanESM2':
                mo2 = 'CanAM4'
            else:
                mo2 = mo

            if mo2 in cld_errs.keys():
                if ripf in cld_errs[mo2].keys():
                    ripf2 = ripf
                else:
                    ripf2 = list(cld_errs[mo2].keys())[0] # take the first available ripf
                    print(mo+': Using Klein error metrics from '+ripf2+' rather than '+ripf)
                E_TCA0 = get_gcm_cld_errs(cld_errs[mo2][ripf2], 'E_TCA')
                E_ctpt0 = get_gcm_cld_errs(cld_errs[mo2][ripf2], 'E_ctpt')
                E_LW0 = get_gcm_cld_errs(cld_errs[mo2][ripf2], 'E_LW')
                E_SW0 = get_gcm_cld_errs(cld_errs[mo2][ripf2], 'E_SW')
                E_NET0 = get_gcm_cld_errs(cld_errs[mo2][ripf2], 'E_NET')
            else:
                E_TCA0 = np.nan * np.ones(7,)
                E_ctpt0 = np.nan * np.ones(7,)
                E_LW0 = np.nan * np.ones(7,)
                E_SW0 = np.nan * np.ones(7,)
                E_NET0 = np.nan * np.ones(7,)
            if cnt == 0:
                E_TCA = E_TCA0
                E_ctpt = E_ctpt0
                E_LW = E_LW0
                E_SW = E_SW0
                E_NET = E_NET0
            else:
                E_TCA = np.row_stack((E_TCA,E_TCA0))
                E_ctpt = np.row_stack((E_ctpt,E_ctpt0))
                E_LW = np.row_stack((E_LW,E_LW0))
                E_SW = np.row_stack((E_SW,E_SW0))
                E_NET = np.row_stack((E_NET,E_NET0))

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

            ECS.append(ecs)

        # (un)assessed is size [models,fbk_types]
    return (assessed,unassessed,ufbk_names,np.array(ECS),models,ripfs,E_TCA,E_ctpt,E_LW,E_SW,E_NET)

def get_gcm_assessed_fbks(fbk_dict):
    # dictionary is structured: [region][sfc][sec][name]

    assessed = []

    # 1) Global high cloud altitude
    #===========================================
    NET = fbk_dict['eq90']['all']['HI680']['NETcld_alt']
    assessed.append(NET)

    # 2) Tropical Ocean Descent Low AMT + TAU
    #===========================================
    trop_lo = 0
    for type in ['NETcld_amt', 'NETcld_tau']:
        trop_lo += fbk_dict['eq30']['ocn_dsc']['LO680'][type]
    assessed.append(trop_lo)

    # 3) Anvil
    #===========================================
    # Tropical Ocean Ascent HI+LO AMT + TAU
    NET = fbk_dict['eq30']['ocn_asc']['HI680']['NETcld_amt'] + fbk_dict['eq30']['ocn_asc']['HI680']['NETcld_tau'] +  \
          fbk_dict['eq30']['ocn_asc']['LO680']['NETcld_amt'] + fbk_dict['eq30']['ocn_asc']['LO680']['NETcld_tau']
    assessed.append(NET)

    # 4) Global land cloud amount
    #===========================================
    NET = fbk_dict['eq90']['lnd']['HI680']['NETcld_amt'] + fbk_dict['eq90']['lnd']['LO680']['NETcld_amt']
    assessed.append(NET)

    # 5) Middle latitude cloud amount feedback
    #===========================================
    # 30-60 marine low cloud amount
    NET = fbk_dict['eq60']['ocn']['LO680']['NETcld_amt'] - fbk_dict['eq30']['ocn']['LO680']['NETcld_amt']
    assessed.append(NET)

    # 6) Extratropical cloud optical depth feedback
    #===========================================
    # 40-70 low cloud optical depth
    NET = fbk_dict['eq70']['all']['LO680']['NETcld_tau'] - fbk_dict['eq40']['all']['LO680']['NETcld_tau']
    assessed.append(NET)

    # 7) true_total
    #===========================================
    NET = fbk_dict['eq90']['all']['ALL']['NETcld_tot']
    true_total = NET

    sum_assessed = np.array(assessed).sum()
    imply_not_assessed = true_total - sum_assessed
    assessed.append(imply_not_assessed)
    assessed.append(sum_assessed)
    assessed.append(true_total)

    return(np.array(assessed)) # size [fbk_types]

def get_gcm_cld_errs(err_dict, name):
    # dictionary is structured: [region][sfc][sec][name]

    DATA = []

    # 1) Global high cloud altitude
    #===========================================
    DATA.append(err_dict['eq90']['all']['HI680'][name])

    # 2) Tropical Ocean Descent Low AMT + TAU
    #===========================================
    DATA.append(err_dict['eq30']['ocn_dsc']['LO680'][name])

    # 3) Anvil
    #===========================================
    # Tropical Ocean Ascent Total minus High Altitude
    DATA.append(err_dict['eq30']['ocn_asc']['ALL'][name])

    # 4) Global land cloud amount
    #===========================================
    DATA.append(err_dict['eq90']['lnd']['ALL'][name])

    # 5) Middle latitude cloud amount feedback
    #===========================================
    # 30-60 marine low cloud amount
    DATA.append(err_dict['30-60']['ocn']['LO680'][name])

    # 6) Extratropical cloud optical depth feedback
    #===========================================
    # 40-70 low cloud optical depth
    DATA.append(err_dict['40-70']['all']['LO680'][name])

    # 7) true_total -- here I will use the standard Klein et al 60-60 errors
    #===========================================
    DATA.append(err_dict['eq60']['all']['ALL'][name])

    return(np.array(DATA)) # size [fbk_types]

def get_unassessed_fbks(fbk_dict):
    # dictionary is structured: [region][sfc][sec][name]

    fbk_names=[]
    unassessed=[]

    # 1) Global low altitude (Complement to global high cloud altitude)
    #===========================================
    NET = fbk_dict['eq90']['all']['LO680']['NETcld_alt']
    unassessed.append(NET)
    fbk_names.append('Low-Cloud Altitude')

    # 2) Tropical Ocean Descent High AMT+TAU (Complement to Tropical Ocean Descent Low AMT+ALT+TAU)
    #===========================================
    NET = fbk_dict['eq30']['ocn_dsc']['HI680']['NETcld_amt'] + fbk_dict['eq30']['ocn_dsc']['HI680']['NETcld_tau']
    unassessed.append(NET)
    fbk_names.append('Tropical Marine Subsidence\nHigh-Cloud Amount + Optical Depth')

    # 3) Tropical land high+low optical depth (Complement to Global land cloud amount)
    #===========================================
    NET = fbk_dict['eq30']['lnd']['HI680']['NETcld_tau'] + fbk_dict['eq30']['lnd']['LO680']['NETcld_tau']
    unassessed.append(NET)
    fbk_names.append('Tropical Land Cloud Optical Depth')

    # 5) Complement to middle latitude cloud amount feedback
    #===========================================
    # 60-90 Ocean Low AMT
    #===========================================
    NET = fbk_dict['eq90']['ocn']['LO680']['NETcld_amt'] - fbk_dict['eq60']['ocn']['LO680']['NETcld_amt']
    unassessed.append(NET)
    fbk_names.append('60-90 Marine Low-Cloud Amount')

    # 5) Complement to Extratropical cloud optical depth feedback
    #===========================================
    # 30-40/70-90 Ocean+Land Low TAU
    #===========================================
    NET = fbk_dict['eq40']['all']['LO680']['NETcld_tau'] - fbk_dict['eq30']['all']['LO680']['NETcld_tau'] + \
          fbk_dict['eq90']['all']['LO680']['NETcld_tau'] - fbk_dict['eq70']['all']['LO680']['NETcld_tau']
    unassessed.append(NET)
    fbk_names.append('30-40 / 70-90 Low-Cloud Optical Depth')

    # 6) 30-90 Ocean+Land High TAU
    #===========================================
    NET = fbk_dict['eq90']['all']['HI680']['NETcld_tau'] - fbk_dict['eq30']['all']['HI680']['NETcld_tau']
    unassessed.append(NET)
    fbk_names.append('30-90 High-Cloud Optical Depth')

    # 7) 30-90 Ocean High AMT
    #===========================================
    NET = fbk_dict['eq90']['ocn']['HI680']['NETcld_amt'] - fbk_dict['eq30']['ocn']['HI680']['NETcld_amt']
    unassessed.append(NET)
    fbk_names.append('30-90 Marine High-Cloud Amount')

    # 8) Global Residual
    #===========================================
    NET = fbk_dict['eq90']['all']['LO680']['NETcld_err'] + fbk_dict['eq90']['all']['HI680']['NETcld_err']
    unassessed.append(NET)
    fbk_names.append('Zelinka et al (2013) Decomposition Residual')

    sum_unassessed = np.array(unassessed).sum()
    unassessed.append(sum_unassessed)
    fbk_names.append('Sum of Unassessed Cloud Feedbacks')

    return(np.array(unassessed),fbk_names) # size [fbk_types]

def horiz_shade(fbk, err, xlabloc=False):
    dummyx = np.linspace(-1000, 1000, 100)
    ymin1 = fbk - 1.64 * err
    ymax1 = fbk + 1.64 * err
    ymin2 = fbk - 0.95 * err
    ymax2 = fbk + 0.95 * err
    pl.fill_between(dummyx, ymin1, ymax1, color='k', alpha=0.1)
    pl.fill_between(dummyx, ymin2, ymax2, color='k', alpha=0.1)
    pl.axhline(fbk, ls='--', color='k', lw=2)
    if xlabloc:
        pl.text(xlabloc, ymin1, ' 5%', fontsize=9, ha='center', va='center')
        pl.text(xlabloc, ymin2, ' 17%', fontsize=9, ha='center', va='center')
        pl.text(xlabloc, ymax2, ' 83%', fontsize=9, ha='center', va='center')
        pl.text(xlabloc, ymax1, ' 95%', fontsize=9, ha='center', va='center')

def label_models(ax, models5, models6, models_isca=None):
    if models_isca is None:
        ylocs = np.linspace(1.5, 9, 3+len(models5)+len(models6))[-1::-1]
    else:
        ylocs = np.linspace(1.5, 9, 5+len(models5)+len(models6)+len(models_isca))[-1::-1]
    xloc = 0.76
    cnt = 0
    ax.text(xloc, ylocs[cnt], 'CMIP5', ha='left', va='center', fontsize=10)
    i=-1
    for mod in models5:
        cnt += 1
        i += 1
        COLOR = 'C0'
        letters = string.ascii_lowercase
        ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70), mfc=COLOR, alpha=0.3, zorder=30)
        ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70),mec=COLOR, mfc='None', zorder=30)
        ax.text(xloc, ylocs[cnt], '    ' + letters[i] + ') ' + mod, ha='left', va='center', fontsize=8)

    cnt += 1
    cnt += 1
    ax.text(xloc, ylocs[cnt], 'CMIP6', ha='left', va='center', fontsize=10)
    for mod in models6:
        cnt += 1
        i += 1
        COLOR = 'C1'
        letters = string.ascii_uppercase
        ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70), mfc=COLOR, alpha=0.3, zorder=30)
        ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70),mec=COLOR, mfc='None', zorder=30)
        ax.text(xloc, ylocs[cnt], '    ' + letters[i] + ') ' + mod, ha='left', va='center', fontsize=8)

    if models_isca is not None:
        cnt += 1
        cnt += 1
        ax.text(xloc, ylocs[cnt], 'Isca', ha='left', va='center', fontsize=10)
        for kk, mod in enumerate(models_isca):
            cnt += 1
            i += 1
            COLOR = 'C2'
            ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70), mfc=COLOR, alpha=0.3, zorder=30)
            ax.plot(xloc, ylocs[cnt], ls='', marker=MARK[mod], ms=np.sqrt(70),mec=COLOR, mfc='None', zorder=30)
            ax.text(xloc, ylocs[cnt], '    ' + str(kk+1) + ') ' + mod, ha='left', va='center', fontsize=8)

    ax.set_xlim(0.7, 1.5)
    ax.set_ylim(1, 9.5) #(1.5, 9.5)
    ax.axis("off")

def plot_expert(ax):
    (expert_cld_fbks, err_expert_cld_fbks, fbk_names) = get_expert_assessed_fbks()
    LN = len(fbk_names)

    yloc = np.arange(0, Nbar * LN, Nbar) - (Nbar / 2 + 0.5) * HEIGHT
    ax.errorbar(expert_cld_fbks[-1::-1], yloc, xerr=1.64 * err_expert_cld_fbks[-1::-1],
                fmt='kd', elinewidth=1.5, ecolor='k', ms=8, zorder=50, label='_nolegend_')
    ax.errorbar(expert_cld_fbks[-1::-1], yloc, xerr=0.95 * err_expert_cld_fbks[-1::-1],
                fmt='kd', elinewidth=2.5, capsize=7, capthick=3, ms=8, zorder=100,
                label='_nolegend_') #,label='WCRP Assessment')
    # Stick a straw-man TIE fighter in the white space to label it
    yloc = 9 # 5.5
    xloc_label = 0.7
    err_eg = 0.15
    ERR1 = 1.64 * err_eg # err_expert_cld_fbks[2]
    ERR2 = 0.95 * err_eg # err_expert_cld_fbks[2]
    ax.errorbar(xloc_label, yloc, xerr=ERR1, fmt='kd',
        elinewidth=1.5, ecolor='k',  ms=8, zorder=50, label='_nolegend_')
    ax.errorbar(xloc_label, yloc, xerr=ERR2, fmt='kd',
        elinewidth=2.5, capsize=7, capthick=3, ms=8, zorder=100, label='_nolegend_')
    dx_arr = [0, -ERR1, ERR1, -ERR2, ERR2]
    xstr_arr = ['Central', '5%', '95%', '17%', '83%']
    for dx, xstr in zip(dx_arr, xstr_arr):
        #pl.text(xloc_label + dx, yloc + 0.1, xstr, ha='left', va='bottom', rotation=45)
        pl.text(xloc_label + dx, yloc + 0.4, xstr, ha='center', va='bottom', rotation=0, fontsize=12)
    ax.text(xloc_label, yloc - 0.6, 'WCRP Assessment', ha='center', va='top', fontsize=12)
    return(fbk_names)

def scatter_label(x, y, models, models5, models_isca, dolabel=False, dolims=False):
    m, b, r, p, std_err = stats.linregress(x.compressed(), y.compressed())
    dummyx = np.linspace(x.min(),x.max(),100)
    if p<0.05:
        LABEL = 'r = '+f'{r:.2f}'+'*'
    else:
        LABEL = 'r = '+f'{r:.2f}'
    pl.plot(dummyx,m*dummyx+b,label=LABEL, lw=3, color='k')
    pl.legend(handletextpad=0.4,frameon=0)
    pl.xticks(fontsize=14)
    pl.yticks(fontsize=14)
    yloc = np.linspace(y.min(), y.max(),len(models))[-1::-1]
    cnt = -1
    ind_isca = 0
    for im,model in enumerate(models):
        cnt += 1
        if np.ma.is_masked(x[im]):
            continue
        if model in models5:
            LETTER, COLOR = string.ascii_lowercase[im], 'C0'
        elif model in models_isca:
            LETTER, COLOR = str(int(ind_isca+1)), 'C2'
            ind_isca += 1
        else:
            LETTER, COLOR = string.ascii_uppercase[im], 'C1'
        pl.text(x[im], y[im],LETTER, ha='center', va='center', color=COLOR,fontweight='bold', fontsize=16)
        if dolabel:
            pl.text(pl.xlim()[-1]+0.05*x.ptp(),yloc[cnt], ' '+LETTER+')  '+model, ha='left', va='center', color=COLOR)
            #pl.text(x.max()+0.05*x.ptp(),yloc[cnt], ' '+LETTER+')  '+model, ha='left', va='center', color=COLOR)
    if dolims:
        pl.xlim(x.min()-0.05*x.ptp(), x.max()+0.05*x.ptp())
        pl.ylim(y.min()-0.05*y.ptp(), y.max()+0.05*y.ptp())

def static_plot(assessed, ecs, models, fbk_names, gen, fig, gs):
    LN = assessed.shape[1]
    if gen=='isca':
        yloc = np.arange(0, Nbar*LN, Nbar) - (Nbar / 2 - 0.5) * HEIGHT
        FACE = 'C2'
        EDGE = 'C2'
    if gen=='5':
        yloc = np.arange(0, Nbar*LN, Nbar) + (Nbar / 2 - 0.5) * HEIGHT
        FACE = 'C0'
        EDGE = 'C0'
    if gen=='6':
        yloc = np.arange(0, Nbar*LN, Nbar) 
        FACE = 'C1'
        EDGE = 'C1'

    avg_assessed = np.ma.average(assessed,0)
    err_assessed = np.ma.std(assessed,0)
    N,T = assessed.shape #[models,fbk_types]
    if gen != 'isca':
        pl.barh(yloc, avg_assessed[-1::-1], height=HEIGHT, align='center', color=FACE,
            alpha=0.3, zorder=10, label='CMIP'+gen+' Mean [n='+str(N)+']')
    else:
        pl.barh(yloc, avg_assessed[-1::-1], height=HEIGHT, align='center', color=FACE,
            alpha=0.3, zorder=10, label=gen+' Mean [n='+str(N)+']')
    pl.barh(yloc, avg_assessed[-1::-1], height=HEIGHT, align='center', color='None',
        ec=EDGE, zorder=10) #,label='CMIP'+gen+' Mean [n='+str(N)+']')
    x = assessed[:, -1::-1]
    y = np.tile(yloc, (N,1))
    z = np.tile(ecs, (T,1)).T
    for ind,model in enumerate(models):
        pl.scatter(x[ind], y[ind],s=70,c=z[ind], marker=MARK[model],edgecolors='k', zorder=20,cmap=CMAP, norm=NORM)
        pl.plot(x[ind], y[ind], ls='', marker=MARK[model], ms=np.sqrt(70),mec='k', mfc='None', zorder=30)#,label=model)

    if gen=='6':
        for Y in np.arange(Nbar/2, Nbar*LN + (Nbar/2+0.5)*HEIGHT, Nbar):
            pl.axhline(y=Y, color='gray', ls='--', lw=0.5)
        #pl.yticks(yloc,fbk_names[-1::-1], fontsize=14)
        pl.yticks(yloc, ['\n'.join(wrap(fbk_nm.capitalize(), 15)) for fbk_nm in fbk_names[-1::-1]], va='center', fontsize=14)
        pl.xticks(fontsize=14)
        pl.axvline(x=0.0, color='k', ls='-')
        pl.xlabel('Cloud feedback parameter (Wm$^{-2}$K$^{-1}$)', fontsize=14)
        pl.ylim(-Nbar/2, Y - HEIGHT / 2)
        if assessed.max()<0.6:
            pl.xlim(-0.3, 0.3)
        else:
            pl.xlim(-0.6, 1.1) #1.2)
        pl.legend(loc=1, fontsize=10, fancybox=True, framealpha=1)
        pl.grid(False)
        pl.minorticks_off()

        # create a second axes for the colorbar
        #ax2 = pl.subplot(gs[-1:, 5:10])
        ax2 = fig.add_axes([0.53, 0.12, 0.012, 0.2])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=CMAP, norm=NORM, extend='both',
                 spacing='proportional', ticks=BOUNDS, boundaries=BOUNDS, orientation='vertical')
        cb.ax.tick_params(labelsize=12)
        cb_tick_labels = []
        cb_tick_ranges = [5, 17, 83, 95] # '(5%)', '(17%)', '(83%)', '(95%)'
        for bound, cb_tick_range in zip(BOUNDS[1:], cb_tick_ranges):
            cb_tick_labels.append(str(bound) + ' ('+ str(int(cb_tick_range)) + '%)' )
        cb.ax.set_yticklabels(cb_tick_labels)
        cb.ax.minorticks_off()
        # ax2.set_ylabel('ECS (K)', size=12)
        ax2.set_title('ECS (K)', size=12)

def vert_shade(fbk, err,ylabloc=False):
    dummyy = np.linspace(-1000,1000,100)
    xmin1 = fbk - 1.64 * err
    xmax1 = fbk + 1.64 * err
    xmin2 = fbk - 0.95 * err
    xmax2 = fbk + 0.95 * err
    pl.fill_betweenx(dummyy,xmin1,xmax1, color='k', alpha=0.1)
    pl.fill_betweenx(dummyy,xmin2,xmax2, color='k', alpha=0.1)
    pl.axvline(fbk, ls='--', color='k', lw=2)
    if ylabloc:
        pl.text(xmin1,ylabloc, ' 5%', fontsize=9, ha='center', va='center')
        pl.text(xmin2,ylabloc, ' 17%', fontsize=9, ha='center', va='center')
        pl.text(xmax2,ylabloc, ' 83%', fontsize=9, ha='center', va='center')
        pl.text(xmax1,ylabloc, ' 95%', fontsize=9, ha='center', va='center')

#def make_all_figs(cld_fbks6,cld_errs6,newmod):
def make_all_figs(exp_grps, exp_markers, isca_cld_fbk_fn, isca_cld_err_fn, isca_forcing_ecs_fn):
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
    for exp_grp, exp_marker in zip(exp_grps, exp_markers):
        MARK[exp_grp] = exp_marker

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
    with open(isca_forcing_ecs_fn, 'r') as f:
        ecs_dict_isca = json.load(f)['isca']

    # Get the assessed and unassessed feedbacks:
    (assessed5, unassessed5, ufbk_names5, ECS5, models5, ripfs5, E_TCA5,
        E_ctpt5, E_LW5, E_SW5, E_NET5) = get_fbks(cld_fbks5,cld_errs5,ecs_dict5)
    (assessed6, unassessed6, ufbk_names6, ECS6, models6, ripfs6, E_TCA6,
        E_ctpt6, E_LW6, E_SW6, E_NET6) = get_fbks(cld_fbks6,cld_errs6,ecs_dict6)
    (assessed_isca, unassessed_isca, ufbk_names_isca, ECS_isca, models_isca, 
        ripfs_isca, E_TCA_isca, E_ctpt_isca, E_LW_isca, E_SW_isca, 
        E_NET_isca) = get_fbks(cld_fbks_isca, cld_errs_isca, ecs_dict_isca)

    LN = assessed6.shape[1] # number of feedback categories
    #print(assessed6)

    ################################################################################################
    # BAR PLOT OF ECS ASSESSMENT CLOUD FEEDBACK COMPONENTS
    # 1) showing the 66% range (+/- 0.95sd) and 5-95% range (+/- 1.64 sd) 
    #    rather than 1sd (68%) and 2 sd (2.5-97.5%) ranges
    # 2) putting the assessment values beneath rather than within the CFMIP bars
    # 3) add something to the legend that makes it clear what the range bars are
    #    (1 sd / 2sd or 66% / 5-95%)
    ################################################################################################
    pl.close('all')
    fig = pl.figure(figsize=(15,12)) #figsize=(15,11))# figsize=(18,12)) #
    gs = gridspec.GridSpec(20, 20)
    ax = pl.subplot(gs[:, :10])
    fbk_names = plot_expert(ax)
    static_plot(assessed_isca, ECS_isca, models_isca, fbk_names, 'isca', fig, gs)
    static_plot(assessed5, ECS5, models5, fbk_names, '5', fig, gs)
    static_plot(assessed6, ECS6, models6, fbk_names, '6', fig, gs)
    '''
    # highlight your model
    m = models6.index(newmod)
    LABEL = newmod+' ['+str(np.round(ECS6[m],1))+' K]'
    yloc = np.arange(0,2*LN,2)-HEIGHT/2
    ax.plot(assessed6[m,-1::-1],yloc, ls='', marker=MARK[newmod],
            ms=12, color='m', zorder=200,label=LABEL)
    '''
    ax.legend(loc=1, fontsize=10, fancybox=True, framealpha=1)
    # ax.set_title('Assessed Cloud Feedback Values [amip-p4K]', fontsize=16)
    ax.set_title('Comparison of Assessed Cloud Feedbacks', fontsize=16)
    # new axis for labeling all models
    ax = pl.subplot(gs[:13, 10:12]) # gs[:10, 10:12]
    label_models(ax, models5, models6, models_isca)
    pl.savefig(fig_dir / 'WCRP_assessed_cld_fbks_amip-p4K.pdf', bbox_inches='tight')
    pl.savefig(fig_dir / 'WCRP_assessed_cld_fbks_amip-p4K.png', bbox_inches='tight', transparent=False, dpi=200)

    ################################################################################################
    # BAR PLOT OF UNASSESSED CLOUD FEEDBACK COMPONENTS
    ################################################################################################
    fig = pl.figure(figsize=(15,11)) #figsize=(18, 12))
    gs = gridspec.GridSpec(20, 20)
    ax = pl.subplot(gs[:, :10])
    static_plot(unassessed_isca, ECS_isca, models_isca, ufbk_names_isca, 'isca', fig, gs)
    static_plot(unassessed5, ECS5, models5, ufbk_names5, '5', fig, gs)
    static_plot(unassessed6, ECS6, models6, ufbk_names6, '6', fig, gs)
    '''
    # highlight your model
    m = models6.index(newmod)
    LABEL = newmod+' ['+str(np.round(ECS6[m],1))+' K]'
    yloc = np.arange(0,2*LN,2)-HEIGHT/2
    ax.plot(unassessed6[m,-1::-1],yloc, ls='-', marker=MARK[newmod], 
            ms=12, color='m', zorder=200,label=LABEL)
    '''
    ax.legend(loc=1, fontsize=10, fancybox=True, framealpha=1)
    #ax.set_title('Unassessed Cloud Feedback Values [amip-p4K]', fontsize=16)
    ax.set_title('Comparison of Unassessed Cloud Feedbacks', fontsize=16)
    # new axis for labeling all models
    ax = pl.subplot(gs[:13, 10:12]) # gs[:10, 10:12]
    label_models(ax, models5, models6, models_isca)
    pl.savefig(fig_dir / 'WCRP_unassessed_cld_fbks_amip-p4K.pdf', bbox_inches='tight')
    pl.savefig(fig_dir / 'WCRP_unassessed_cld_fbks_amip-p4K.png', bbox_inches='tight', transparent=False, dpi=200)

    ################################################################################################
    # ERROR METRIC OF MODEL AGREEMENT WITH INDIVIDUAL CLOUD FEEDBACKS
    ################################################################################################
    expert_cld_fbks, err_expert_cld_fbks, fbk_names = get_expert_assessed_fbks()
    serr = (assessed5 - expert_cld_fbks)**2
    # average taken over all but last 2 feedbacks, which are sum_assessed, true_total
    RMSE5 = np.sqrt(np.average(serr[:,:-2],1)) 
    serr = (assessed6 - expert_cld_fbks)**2
    RMSE6 = np.sqrt(np.average(serr[:,:-2],1))
    serr = (assessed_isca - expert_cld_fbks)**2
    RMSE_isca = np.sqrt(np.average(serr[:,:-2],1))

    assessed56 = np.array(list(assessed5) + list(assessed6) + list(assessed_isca))
    unassessed56 = np.array(list(unassessed5) + list(unassessed6) + list(unassessed_isca))
    E_TCA56 = np.array(list(E_TCA5) + list(E_TCA6) + list(E_TCA_isca))
    E_ctpt56 = np.array(list(E_ctpt5) + list(E_ctpt6) + list(E_ctpt_isca))
    E_LW56 = np.array(list(E_LW5) + list(E_LW6) + list(E_LW_isca))
    E_SW56 = np.array(list(E_SW5) + list(E_SW6) + list(E_SW_isca))
    E_NET56 = np.array(list(E_NET5) + list(E_NET6) + list(E_NET_isca))
    RMSE56 = np.array(list(RMSE5) + list(RMSE6) + list(RMSE_isca))
    ECS56 = np.array(list(ECS5) + list(ECS6) + list(ECS_isca))
    models56 = np.array(list(models5) + list(models6) + list(models_isca))

    # print('E_TCA_isca:', E_TCA_isca)
    # print('E_LW_isca:', E_LW_isca)
    # print('E_SW_isca:', E_SW_isca)
    # print('E_NET_isca:', E_NET_isca)

    # assessed56 = np.append(assessed5,assessed6,axis=0)
    # unassessed56 = np.append(unassessed5,unassessed6,axis=0)
    # E_TCA56 = np.append(E_TCA5,E_TCA6,axis=0)
    # E_ctpt56 = np.append(E_ctpt5,E_ctpt6,axis=0)
    # E_LW56 = np.append(E_LW5,E_LW6,axis=0)
    # E_SW56 = np.append(E_SW5,E_SW6,axis=0)
    # E_NET56 = np.append(E_NET5,E_NET6,axis=0)
    # RMSE56 = np.append(RMSE5,RMSE6)
    # ECS56 = np.append(ECS5,ECS6)
    # models56 = np.append(models5,models6)

    inds = np.argsort(RMSE56)

    ######################################################
    # Plot RMSE vs total cloud feedback:
    ######################################################
    pl.figure(figsize=(18,12))
    gs = gridspec.GridSpec(10, 24)

    # Color-code by ECS
    ax = pl.subplot(gs[:4, :9])
    pl.scatter(RMSE5, assessed5[:,-1], s=200, c=ECS5, marker='D',
                zorder=10, cmap=CMAP0, norm=NORM0)
    pl.scatter(RMSE6, assessed6[:,-1], s=275, c=ECS6, marker='o',
                zorder=10, cmap=CMAP0, norm=NORM0)
    pl.scatter(RMSE_isca, assessed_isca[:,-1], s=200, c=ECS_isca, marker='>',
                zorder=10, cmap=CMAP0, norm=NORM0)

    # plot again with no fill color so all symbols are present
    pl.plot(RMSE5, assessed5[:,-1], 'D', ms=np.sqrt(200),
            mec='k', mfc='None', zorder=20, label='CMIP5')
    pl.plot(RMSE6, assessed6[:,-1], 'o', ms=np.sqrt(275),
            mec='k', mfc='None', zorder=20,label='CMIP6')
    pl.plot(RMSE_isca, assessed_isca[:,-1], '>', ms=np.sqrt(200),
            mec='k', mfc='None', zorder=20, label='Isca')
    pl.legend(loc=3,ncol=2,handletextpad=0.4,frameon=0)
    pl.xlabel('Cloud Feedback RMSE [Wm$^{-2}$K$^{-1}$]', fontsize=14)
    pl.ylabel('Total Cloud Feedback [Wm$^{-2}$K$^{-1}$]', fontsize=14)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=12)
    xlim = (np.min([np.min(RMSE5), np.min(RMSE6), np.min(RMSE_isca)])-0.05,
            np.max([np.max(RMSE5), np.max(RMSE6), np.max(RMSE_isca)])+0.05)

    # put horizontal shading for assessed total cloud feedback
    horiz_shade(expert_cld_fbks[-1], err_expert_cld_fbks[-1], xlim[1]-0.02) # 0.16)
    pl.xlim(xlim) #(0.05, 0.17)
    pl.ylim(-0.35, 1.25)
    pl.title('a', fontsize=16,loc='left')

    # Label each model:
    x = RMSE56
    y = assessed56[:,-1]
    cnt = 0
    ind_isca = 0
    for im,model in enumerate(models56):
        cnt += 1
        if np.ma.is_masked(x[im]):
            continue
        if model in models5:
            LETTER = string.ascii_lowercase[im]
        elif model in models_isca:
            LETTER = str(int(ind_isca + 1))
            ind_isca += 1
        else:
            LETTER = string.ascii_uppercase[im]

        COLOR='k'
        if ECS56[im]<3.5 or ECS56[im]>5:
            COLOR='w'
        ax.text(x[im], y[im], LETTER, ha='center', va='center', 
            color=COLOR, fontsize=10,fontweight='bold', zorder=30)
    # create a second axes for the colorbar
    ax2 = pl.subplot(gs[:4, 9:10])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=CMAP0, norm=NORM0,
            spacing='proportional', ticks=BOUNDS0, boundaries=BOUNDS0)
    cb.ax.tick_params(labelsize=14)
    ax2.set_ylabel('ECS [K]', size=14)

    # Color-code by E_NET
    KEM_CMAP = pl.cm.magma_r  # define the colormap
    KEM_BOUNDS = np.arange(0.7,1.9,0.15)
    KEM_NORM = mpl.colors.BoundaryNorm(KEM_BOUNDS, KEM_CMAP.N)

    ax = pl.subplot(gs[:4, 12:21])
    pl.scatter(RMSE5,assessed5[:,-1],s=200,c=E_NET5[:,-1], marker='D',
                zorder=10,cmap=KEM_CMAP, norm=KEM_NORM)
    pl.scatter(RMSE6,assessed6[:,-1],s=275,c=E_NET6[:,-1], marker='o',
                zorder=10,cmap=KEM_CMAP, norm=KEM_NORM)
    pl.scatter(RMSE_isca,assessed_isca[:,-1],s=200,c=E_NET_isca[:,-1], marker='>',
                zorder=10,cmap=KEM_CMAP, norm=KEM_NORM)
    # plot again with no fill color so all symbols are present
    pl.plot(RMSE5, assessed5[:,-1], 'D', ms=np.sqrt(200), mec='k',
            mfc='None', zorder=20, label='CMIP5')
    pl.plot(RMSE6, assessed6[:,-1], 'o', ms=np.sqrt(275), mec='k',
            mfc='None', zorder=20, label='CMIP6')
    pl.plot(RMSE_isca, assessed_isca[:,-1], '>', ms=np.sqrt(200), mec='k',
            mfc='None', zorder=20, label='Isca')
    pl.legend(loc=3,ncol=2,handletextpad=0.4,frameon=0)
    pl.xlabel('Cloud Feedback RMSE [Wm$^{-2}$K$^{-1}$]', fontsize=14)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=12)
    xlim = (np.min([np.min(RMSE5), np.min(RMSE6), np.min(RMSE_isca)])-0.05,
            np.max([np.max(RMSE5), np.max(RMSE6), np.max(RMSE_isca)])+0.05)
    ylim = (np.min([np.min(assessed5[:,-1]), np.min(assessed6[:,-1]), np.min(assessed_isca[:,-1])]),
            np.max([np.max(assessed5[:,-1]), np.max(assessed6[:,-1]), np.max(assessed_isca[:,-1])]))
    # put horizontal shading for assessed total cloud feedback
    horiz_shade(expert_cld_fbks[-1], err_expert_cld_fbks[-1], xlim[1]-0.02) #0.16)
    pl.xlim(xlim) # 0.05, 0.17)
    pl.ylim(-0.35, 1.25)
    pl.title('b', fontsize=16,loc='left')
    
    # Label each model:
    x = RMSE56
    y = assessed56[:,-1]
    cnt = 0
    ind_isca = 0
    for im,model in enumerate(models56):
        cnt += 1
        if np.ma.is_masked(x[im]):
            continue
        if model in models5:
            LETTER = string.ascii_lowercase[im]
        elif model in models_isca:
            LETTER = str(int(ind_isca + 1))
            ind_isca += 1
        else:
            LETTER = string.ascii_uppercase[im]
        COLOR='k'
        if E_NET56[im,-1]>1.25:
            COLOR='w'
        ax.text(x[im], y[im],LETTER, ha='center', va='center',
            color=COLOR, fontsize=10, fontweight='bold', zorder=30)
    # create a second axes for the colorbar
    ax2 = pl.subplot(gs[:4, 21:22])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=KEM_CMAP, norm=KEM_NORM,
            spacing='proportional', ticks=KEM_BOUNDS, boundaries=KEM_BOUNDS)
    cb.ax.tick_params(labelsize=14)
    ax2.set_ylabel('$\mathrm{E_{NET}}$', size=14)
    pl.savefig(fig_dir / 'WCRP_assessed_RMSE_v_cldfbk2_amip-p4K.pdf', bbox_inches='tight')

    '''
    ######################################################
    # Plot Klein error metrics vs cloud feedback & RMSE:
    ######################################################
    # Plot E_NET vs total cloud feedback:
    pl.figure(figsize=(18,12))
    gs = gridspec.GridSpec(10, 24)
    # E_NET vs total cloud feedback
    ax = pl.subplot(gs[:4, :9])
    # X = np.append(E_NET5[:,-1],E_NET6[:,-1],axis=0)
    X = np.array(list(E_NET5[:,-1]) + list(E_NET6[:,-1]) + list(E_NET_isca[:,-1]))
    x = np.ma.masked_invalid(X)
    Y = assessed56[:,-1]
    y = np.ma.masked_where(x.mask, Y)
    scatter_label(x, y, models56, models5, models_isca, False, True)
    pl.ylabel(fbk_names[-1]+' [Wm$^{-2}$K$^{-1}$]', fontsize=14)
    pl.xlabel('$\mathrm{E_{NET}}$', fontsize=14)
    pl.title('a', fontsize=16,loc='left')
    # put horizontal shading for assessed total cloud feedback
    horiz_shade(expert_cld_fbks[-1], err_expert_cld_fbks[-1],x.max())
    # pl.ylim(-0.4,1.4)
    # pl.xlim(0.60,1.65)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    xlim = (x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ylim = (y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    # Plot E_NET vs cloud feedback RMSE:
    ax = pl.subplot(gs[:4, 12:21])
    #X = np.append(E_NET5[:,-1],E_NET6[:,-1],axis=0)
    X = np.array(list(E_NET5[:,-1]) + list(E_NET6[:,-1]) + list(E_NET_isca[:,-1]))
    x = np.ma.masked_invalid(X)
    Y = RMSE56
    y = np.ma.masked_where(x.mask, Y)
    scatter_label(x, y, models56, models5, models_isca, False, True)
    pl.ylabel('Cloud Feedback RMSE [Wm$^{-2}$K$^{-1}$]', fontsize=14)
    pl.xlabel('$\mathrm{E_{NET}}$', fontsize=14)
    pl.title('b', fontsize=16,loc='left')

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    xlim = (x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ylim = (y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

    pl.ylim(ylim) #(0.05,0.17)
    pl.xlim(xlim) #(0.60,1.65)
    pl.savefig(fig_dir / 'WCRP_totcldfbks2_v_E_NET_amip-p4K.pdf', bbox_inches='tight')
    '''


    #######################################################
    # PRINT OUT THE TABLE OF EVERYTHING:
    #######################################################

    # Determine min and max of the likely (66%) and 
    #  very likely (90%) range of expert judgement
    expert_min66 = expert_cld_fbks - 0.95 * err_expert_cld_fbks
    expert_max66 = expert_cld_fbks + 0.95 * err_expert_cld_fbks
    expert_min90 = expert_cld_fbks - 1.64 * err_expert_cld_fbks
    expert_max90 = expert_cld_fbks + 1.64 * err_expert_cld_fbks

    # ASSESSED
    fbk_table = fig_dir / 'CMIP56_assessed_cld_fbks_WCRP.txt'
    dash = '-' * 170
    with open(fbk_table, "w") as f:
        print(dash, file=f)
        print('amip-p4K vs WCRP Assessed Cloud Feedbacks', file=f)
        print(dash, file=f)
        print('Date modified: '+str(date.today()), file=f)
        print('Contact: Mark D. Zelinka [zelinka1@llnl.gov]', file=f)
        print(dash, file=f)
        data=['Model', 'Variant', 'Hi.Alt', 'Marine.Lo', 'Anvil', 'Land.Amt', 'Midlat.Amt',
              'Hilat.Tau', 'Unassessed', 'Assessed', 'Total', 'RMSE', 'E_NET', 'ECS']
        print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],
                   data[8],data[9],data[10],data[11],data[12],data[13]), file=f)
        print(dash, file=f)

        cnt = -1
        for gen in ['5', '6']: #, 'isca']: # For Isca, also need to change the string for CMIP+gen
            # if 'isca' in gen.lower():
            #     letters = [str(int(ii+1)) for ii in range(0,len(models_isca))]
            #     assessed = assessed_isca
            #     models = models_isca
            #     ripfs = ripfs_isca
            #     RMSE = RMSE_isca
            #     ECS = ECS_isca
            #     E_NET = E_NET_isca[:,-1]
            if gen=='5':
                letters = string.ascii_lowercase
                assessed = assessed5
                models = models5
                ripfs = ripfs5
                RMSE = RMSE5
                ECS = ECS5
                E_NET = E_NET5[:,-1]
            else:
                letters = string.ascii_uppercase
                assessed = assessed6
                models = models6
                ripfs = ripfs6
                RMSE = RMSE6
                ECS = ECS6
                E_NET = E_NET6[:,-1]
            E_NET = np.ma.masked_where(np.isnan(E_NET),E_NET)

            for ic in range(len(models)):
                cnt += 1
                if 'isca' in gen:
                    LM = letters[ic]+') ' + models[ic]
                else:
                    LM = letters[cnt]+') ' + models[ic]
                data=[LM, ripfs[ic]]
                for fb in range(len(fbk_names)):
                    this = assessed[ic,fb]
                    if this<expert_min90[fb] or this>expert_max90[fb]:
                        data.append('%0.2f'%(this)+'**')
                    elif this<expert_min66[fb] or this>expert_max66[fb]:
                        data.append('%0.2f'%(this)+'*')
                    else:
                        data.append('%0.2f'%(this))
                data.append('%0.2f'%(RMSE[ic]))
                data.append('%0.2f'%(E_NET[ic]))
                if ECS[ic]>4.7:
                    data.append('%0.2f'%(ECS[ic])+'**')
                elif ECS[ic]>3.9:
                    data.append('%0.2f'%(ECS[ic])+'*')
                else:
                    data.append('%0.2f'%(ECS[ic]))
                print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
                    format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],
                           data[7],data[8],data[9],data[10],data[11],data[12],data[13]), file=f)

            # multi-model mean
            print(dash, file=f)
            data = ['CMIP'+gen+' Average', '', '%0.2f'%(np.ma.average(assessed[:,0])),
                '%0.2f'%(np.ma.average(assessed[:,1])), '%0.2f'%(np.ma.average(assessed[:,2])),
                '%0.2f'%(np.ma.average(assessed[:,3])), '%0.2f'%(np.ma.average(assessed[:,4])),
                '%0.2f'%(np.ma.average(assessed[:,5])), '%0.2f'%(np.ma.average(assessed[:,6])),
                '%0.2f'%(np.ma.average(assessed[:,7])), '%0.2f'%(np.ma.average(assessed[:,8])),
                '%0.2f'%(np.ma.average(RMSE)), '%0.2f'%(np.ma.average(E_NET)), '%0.2f'%(np.ma.average(ECS))]
            print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
                format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],
                       data[8],data[9],data[10],data[11],data[12],data[13]), file=f)
            #print(dash, file=f)
            data = ['CMIP'+gen+' Stdev', '', '%0.2f'%(np.ma.std(assessed[:,0])),
                '%0.2f'%(np.ma.std(assessed[:,1])), '%0.2f'%(np.ma.std(assessed[:,2])),
                '%0.2f'%(np.ma.std(assessed[:,3])), '%0.2f'%(np.ma.std(assessed[:,4])),
                '%0.2f'%(np.ma.std(assessed[:,5])), '%0.2f'%(np.ma.std(assessed[:,6])),
                '%0.2f'%(np.ma.std(assessed[:,7])), '%0.2f'%(np.ma.std(assessed[:,8])),
                '%0.2f'%(np.ma.std(RMSE)), '%0.2f'%(np.ma.std(E_NET)), '%0.2f'%(np.ma.std(ECS))]
            print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
                format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],
                       data[8],data[9],data[10],data[11],data[12],data[13]), file=f)
            print(dash, file=f)

        # multi-model mean for combined CMIP5+6 models
        assessed = assessed56
        RMSE = RMSE56
        ECS = ECS56
        E_NET = E_NET56[:,-1]
        E_NET = np.ma.masked_where(np.isnan(E_NET),E_NET)

        #print(dash, file=f)
        data=['CMIP5/6 Average', '', '%0.2f'%(np.ma.average(assessed[:,0])),
            '%0.2f'%(np.ma.average(assessed[:,1])), '%0.2f'%(np.ma.average(assessed[:,2])),
            '%0.2f'%(np.ma.average(assessed[:,3])), '%0.2f'%(np.ma.average(assessed[:,4])),
            '%0.2f'%(np.ma.average(assessed[:,5])), '%0.2f'%(np.ma.average(assessed[:,6])),
            '%0.2f'%(np.ma.average(assessed[:,7])), '%0.2f'%(np.ma.average(assessed[:,8])),
            '%0.2f'%(np.ma.average(RMSE)), '%0.2f'%(np.ma.average(E_NET)), '%0.2f'%(np.ma.average(ECS))]
        print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],
                   data[8],data[9],data[10],data[11],data[12],data[13]), file=f)
        #print(dash, file=f)
        data=['CMIP5/6 Stdev', '', '%0.2f'%(np.ma.std(assessed[:,0])),
            '%0.2f'%(np.ma.std(assessed[:,1])), '%0.2f'%(np.ma.std(assessed[:,2])),
            '%0.2f'%(np.ma.std(assessed[:,3])), '%0.2f'%(np.ma.std(assessed[:,4])),
            '%0.2f'%(np.ma.std(assessed[:,5])), '%0.2f'%(np.ma.std(assessed[:,6])),
            '%0.2f'%(np.ma.std(assessed[:,7])), '%0.2f'%(np.ma.std(assessed[:,8])),
            '%0.2f'%(np.ma.std(RMSE)), '%0.2f'%(np.ma.std(E_NET)), '%0.2f'%(np.ma.std(ECS))]
        print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],
                   data[8],data[9],data[10],data[11],data[12],data[13]), file=f)
        print(dash, file=f)
        data=['WCRP Central', '']
        for fb in range(len(fbk_names)):
            data.append(str(expert_cld_fbks[fb]))
        print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],
                   data[6],data[7],data[8],data[9],data[10]), file=f)
        #print(dash, file=f)
        data=['WCRP Stdev', '']
        for fb in range(len(fbk_names)):
            data.append(str('%0.2f'%(err_expert_cld_fbks[fb])))
        print('{:<21s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}{:<11s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],
                   data[6],data[7],data[8],data[9],data[10]), file=f)
        print(dash, file=f)

    # UNASSESSED
    fbk_table = fig_dir / 'CMIP56_unassessed_cld_fbks_WCRP.txt'
    dash = '-' * 170
    with open(fbk_table, "w") as f:
        print(dash, file=f)
        print('amip-p4K vs WCRP Unassessed Cloud Feedbacks', file=f)
        print(dash, file=f)
        print('Date modified: '+str(date.today()), file=f)
        print('Contact: Mark D. Zelinka [zelinka1@llnl.gov]', file=f)
        print(dash, file=f)
        data=['Model', 'Variant', 'Lo Alt', 'Marine Hi', 'Land Tau', '60-90 Amt',
             'Extr Lo Tau', 'Extr Hi Tau', 'Extr Hi Amt', 'Z13 Resid', 'Sum Unassessed']
        print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],
                   data[6],data[7],data[8],data[9],data[10]), file=f)
        print(dash, file=f)

        for gen in ['5', '6']:
            if gen=='5':
                unassessed = unassessed5
                models = models5
                ripfs = ripfs5
            else:
                unassessed = unassessed6
                models = models6
                ripfs = ripfs6

            for ic in range(len(models)):
                data=[models[ic],ripfs[ic]]
                for fb in range(len(ufbk_names6)):
                    this = unassessed[ic,fb]
                    data.append('%0.2f'%(this))
                print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
                    format(data[0],data[1],data[2],data[3],data[4],data[5],
                           data[6],data[7],data[8],data[9],data[10]), file=f)

            # multi-model mean
            print(dash, file=f)
            data = ['CMIP'+gen+' Average', '', '%0.2f'%(np.ma.average(unassessed[:,0])),
                '%0.2f'%(np.ma.average(unassessed[:,1])), '%0.2f'%(np.ma.average(unassessed[:,2])),
                '%0.2f'%(np.ma.average(unassessed[:,3])), '%0.2f'%(np.ma.average(unassessed[:,4])), 
                '%0.2f'%(np.ma.average(unassessed[:,5])), '%0.2f'%(np.ma.average(unassessed[:,6])),
                '%0.2f'%(np.ma.average(unassessed[:,7])), '%0.2f'%(np.ma.average(unassessed[:,8]))]
            print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
                format(data[0],data[1],data[2],data[3],data[4],data[5],
                       data[6],data[7],data[8],data[9],data[10]), file=f)
            #print(dash, file=f)
            data = ['CMIP'+gen+' Stdev', '', '%0.2f'%(np.ma.std(unassessed[:,0])), 
                '%0.2f'%(np.ma.std(unassessed[:,1])), '%0.2f'%(np.ma.std(unassessed[:,2])),
                '%0.2f'%(np.ma.std(unassessed[:,3])), '%0.2f'%(np.ma.std(unassessed[:,4])),
                '%0.2f'%(np.ma.std(unassessed[:,5])), '%0.2f'%(np.ma.std(unassessed[:,6])),
                '%0.2f'%(np.ma.std(unassessed[:,7])), '%0.2f'%(np.ma.std(unassessed[:,8]))]
            print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
                format(data[0],data[1],data[2],data[3],data[4],data[5],
                       data[6],data[7],data[8],data[9],data[10]), file=f)
            print(dash, file=f)

        # multi-model mean for combined CMIP5+6 models
        unassessed = unassessed56
        #print(dash, file=f)
        data = ['CMIP5/6 Average', '', '%0.2f'%(np.ma.average(unassessed[:,0])), 
            '%0.2f'%(np.ma.average(unassessed[:,1])), '%0.2f'%(np.ma.average(unassessed[:,2])),
            '%0.2f'%(np.ma.average(unassessed[:,3])), '%0.2f'%(np.ma.average(unassessed[:,4])),
            '%0.2f'%(np.ma.average(unassessed[:,5])), '%0.2f'%(np.ma.average(unassessed[:,6])),
            '%0.2f'%(np.ma.average(unassessed[:,7])), '%0.2f'%(np.ma.average(unassessed[:,8]))]
        print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],
                   data[6],data[7],data[8],data[9],data[10]), file=f)
        #print(dash, file=f)
        data = ['CMIP5/6 Stdev', '', '%0.2f'%(np.ma.std(unassessed[:,0])), 
            '%0.2f'%(np.ma.std(unassessed[:,1])), '%0.2f'%(np.ma.std(unassessed[:,2])),
            '%0.2f'%(np.ma.std(unassessed[:,3])), '%0.2f'%(np.ma.std(unassessed[:,4])),
            '%0.2f'%(np.ma.std(unassessed[:,5])), '%0.2f'%(np.ma.std(unassessed[:,6])),
            '%0.2f'%(np.ma.std(unassessed[:,7])), '%0.2f'%(np.ma.std(unassessed[:,8]))]
        print('{:<21s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}{:<13s}'.\
            format(data[0],data[1],data[2],data[3],data[4],data[5],
                   data[6],data[7],data[8],data[9],data[10]), file=f)
        print(dash, file=f)

if __name__ == '__main__':
    #newmod = 'GFDL-CM4'

    # Add markers for Isca runs
    exp_tbl = pd.read_csv('isca_qflux_exps_for_plots.csv', header=0)
    exp_grps = list(exp_tbl.iloc[:, 0])
    exp_markers = list(exp_tbl.iloc[:, 2])
    for exp_grp, exp_marker in zip(exp_grps, exp_markers):
        MARK[exp_grp] = exp_marker

    #make_all_figs()
