import numpy as np

def map_SWkern_to_lon(Ksw, albcsmap):
    from scipy.interpolate import interp1d
    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,kernel_lats,3
    # albcsmap is size A,kernel_lats,kernel_lons
    
    albcs = np.arange(0.0, 1.5, 0.5) 
    A = albcsmap.shape[0]
    TT = Ksw.shape[1]
    PP = Ksw.shape[2]
    nlat = Ksw.shape[3]
    nlon = albcsmap.shape[2]
    SWkernel_map = np.ones((A, TT, PP, nlat, nlon)) * np.nan

    for M in range(A):
        MM = M
        while MM > 11:
            MM = MM - 12
        for i_lat in range(nlat):
            alon = albcsmap[M, i_lat, :] 
            # interp1d can't handle mask but it can deal with NaN (?)
            #try:
            #    alon2 = np.ma.masked_array(alon, np.isnan(alon)) 
            #except:
            #    alon2 = alon
            alon2 = alon
            if np.ma.count(alon2) > 1: # at least 1 unmasked value
                if np.nansum(Ksw[MM,:,:,i_lat,:] > 0) == 0:
                    SWkernel_map[M,:,:,i_lat,:] = 0
                else:
                    f = interp1d(albcs, Ksw[MM,:,:,i_lat,:], axis=2)
                    ynew = f(alon2)
                    #ynew= np.ma.masked_array(alon2, np.isnan(alon2))
                    SWkernel_map[M,:,:,i_lat,:] = ynew
            else:
                continue

    return SWkernel_map

def KT_decomposition_4D(c1, c2, Klw, Ksw):
    # this function takes in a (tau,CTP,lat,lon) matrix and performs the 
    # decomposition of Zelinka et al 2013 doi:10.1175/JCLI-D-12-00555.1

    # reshape to be (CTP,tau,lat,lon)
    # This is inefficient but done purely because Mark can't think in tau,CTP space
    c1  = np.transpose(c1,  axes=(1,0,2,3)) # control cloud fraction histogram
    c2  = np.transpose(c2,  axes=(1,0,2,3)) # perturbed cloud fraction histogram
    Klw = np.transpose(Klw, axes=(1,0,2,3)) # LW Kernel histogram
    Ksw = np.transpose(Ksw, axes=(1,0,2,3)) # SW Kernel histogram

    P = c1.shape[0]
    T = c1.shape[1]

    c = c1
    sum_c = np.tile(np.nansum(c, axis=(0,1)), (P,T,1,1))                       # Eq. B2
    dc = c2 - c1
    sum_dc = np.tile(np.nansum(dc, axis=(0,1)), (P,T,1,1))
    dc_prop = c * (sum_dc / sum_c)
    dc_star = dc - dc_prop                                                      # Eq. B1

    # LW components
    Klw0 = np.tile(np.nansum(Klw * c / sum_c, axis=(0,1)), (P,T,1,1))           # Eq. B4
    Klw_prime = Klw - Klw0                                                      # Eq. B3
    this = np.nansum(Klw_prime * np.tile(np.nansum(c / sum_c, 0), (P,1,1,1)), 1)  # Eq. B7a
    Klw_p_prime = np.tile(np.tile(this, (1,1,1,1)), (T,1,1,1))                  # Eq. B7b
    Klw_p_prime = np.transpose(Klw_p_prime, axes=[1,0,2,3])
    
    that = np.tile(np.tile(np.nansum(c / sum_c, 1), (1,1,1,1)), (T,1,1,1))     # Eq. B8a
    that = np.transpose(that, axes=[1,0,2,3])
    Klw_t_prime = np.tile(np.nansum(Klw_prime * that, 0), (P,1,1,1))           # Eq. B8b
    Klw_resid_prime = Klw_prime - Klw_p_prime - Klw_t_prime                    # Eq. B9
    dRlw_true = np.nansum(Klw * dc, axis=(0,1))                                # LW total

    dRlw_prop = Klw0[0,0,:,:] * sum_dc[0,0,:,:]                                 # LW amount component
    dRlw_dctp = np.nansum(Klw_p_prime * dc_star, axis=(0,1))                    # LW altitude component
    dRlw_dtau = np.nansum(Klw_t_prime * dc_star, axis=(0,1))                    # LW optical depth component
    dRlw_resid = np.nansum(Klw_resid_prime * dc_star, axis=(0,1))               # LW residual
    dRlw_sum = dRlw_prop + dRlw_dctp + dRlw_dtau + dRlw_resid                   # sum of LW components -- should equal LW total

    # SW components
    Ksw0 = np.tile(np.nansum(Ksw * c / sum_c, axis=(0,1)), (P,T,1,1))          # Eq. B4
    Ksw_prime = Ksw - Ksw0                                                     # Eq. B3
    this = np.nansum(Ksw_prime * np.tile(np.nansum(c / sum_c, 0), (P,1,1,1)), 1)  # Eq. B7a 
    Ksw_p_prime = np.tile(np.tile(this, (1,1,1,1)), (T,1,1,1))                  # Eq. B7b  
    Ksw_p_prime = np.transpose(Ksw_p_prime, axes=[1,0,2,3])
    that = np.tile(np.tile(np.nansum(c / sum_c, 1), (1,1,1,1)), (T,1,1,1))      # Eq. B8a
    that = np.transpose(that, axes=[1,0,2,3])
    Ksw_t_prime = np.tile(np.nansum(Ksw_prime * that, 0), (P,1,1,1))            # Eq. B8b
    Ksw_resid_prime = Ksw_prime - Ksw_p_prime - Ksw_t_prime                     # Eq. B9
    dRsw_true = np.nansum(Ksw * dc, axis=(0,1))                                 # SW total

    dRsw_prop = Ksw0[0,0,:,:] * sum_dc[0,0,:,:]                                 # SW amount component
    dRsw_dctp = np.nansum(Ksw_p_prime * dc_star, axis=(0,1))                    # SW altitude component
    dRsw_dtau = np.nansum(Ksw_t_prime * dc_star, axis=(0,1))                    # SW optical depth component
    dRsw_resid = np.nansum(Ksw_resid_prime * dc_star, axis=(0,1))               # SW residual
    dRsw_sum = dRsw_prop + dRsw_dctp + dRsw_dtau + dRsw_resid                   # sum of SW components -- should equal SW total

    dc_star = np.transpose(dc_star, (1,0,2,3)) 
    dc_prop = np.transpose(dc_prop, (1,0,2,3))

    return (dRlw_true, dRlw_prop, dRlw_dctp, dRlw_dtau, dRlw_resid, 
            dRsw_true, dRsw_prop, dRsw_dctp, dRsw_dtau, dRsw_resid,
            dc_star, dc_prop)

def KT_decomposition_general(c1, c2, Klw, Ksw):
    """
    this function takes in a (month,TAU,CTP,lat,lon) matrix and performs the 
    decomposition of Zelinka et al 2013 doi:10.1175/JCLI-D-12-00555.1
    """

    # To help with broadcasting, move month axis to the end so that TAU,CTP are first
    c1 = np.array(np.moveaxis(np.array(c1),0,-1))
    c2 = np.array(np.moveaxis(np.array(c2),0,-1))
    Klw = np.moveaxis(np.array(Klw),0,-1)
    Ksw = np.moveaxis(np.array(Ksw),0,-1)
    
    sum_c = np.nansum(np.nansum(c1,0),0)                            # Eq. B2
    dc = c2 - c1 
    sum_dc = np.nansum(np.nansum(dc,0),0)
    dc_prop = c1*(sum_dc/sum_c)
    dc_star = dc - dc_prop                                          # Eq. B1

    # LW components
    Klw0 = np.nansum(np.nansum(Klw*c1/sum_c,0),0)                   # Eq. B4
    Klw_prime = Klw - Klw0                                          # Eq. B3
    B7a = np.nansum(c1/sum_c,1,keepdims=True)                       # need to keep this as [TAU,1,...]
    Klw_p_prime = np.nansum(Klw_prime*B7a,0)                        # Eq. B7
    Klw_t_prime = np.nansum(Klw_prime*np.nansum(c1/sum_c,0),1)      # Eq. B8   
    Klw_resid_prime = Klw_prime - np.expand_dims(Klw_p_prime,0) - np.expand_dims(Klw_t_prime,1)        # Eq. B9
    dRlw_true = np.nansum(np.nansum(Klw*dc,1),0)                    # LW total
    dRlw_prop = Klw0*sum_dc                                         # LW amount component
    dRlw_dctp = np.nansum(Klw_p_prime*np.nansum(dc_star,0),0)       # LW altitude component
    dRlw_dtau = np.nansum(Klw_t_prime*np.nansum(dc_star,1),0)       # LW optical depth component
    dRlw_resid = np.nansum(np.nansum(Klw_resid_prime*dc_star,1),0)  # LW residual
    dRlw_sum = dRlw_prop + dRlw_dctp + dRlw_dtau + dRlw_resid       # sum of LW components -- should equal LW total

    # SW components
    Ksw0 = np.nansum(np.nansum(Ksw*c1/sum_c,0),0)                   # Eq. B4
    Ksw_prime = Ksw - Ksw0                                          # Eq. B3
    B7a = np.nansum(c1/sum_c,1,keepdims=True)                       # need to keep this as [TAU,1,...]
    Ksw_p_prime = np.nansum(Ksw_prime*B7a,0)                        # Eq. B7
    Ksw_t_prime = np.nansum(Ksw_prime*np.nansum(c1/sum_c,0),1)      # Eq. B8  
    Ksw_resid_prime = Ksw_prime - np.expand_dims(Ksw_p_prime,0) - np.expand_dims(Ksw_t_prime,1)        # Eq. B9 
    dRsw_true = np.nansum(np.nansum(Ksw*dc,1),0)                    # SW total
    dRsw_prop = Ksw0*sum_dc                                         # SW amount component
    dRsw_dctp = np.nansum(Ksw_p_prime*np.nansum(dc_star,0),0)       # SW altitude component
    dRsw_dtau = np.nansum(Ksw_t_prime*np.nansum(dc_star,1),0)       # SW optical depth component
    dRsw_resid = np.nansum(np.nansum(Ksw_resid_prime*dc_star,1),0)  # SW residual
    dRsw_sum = dRsw_prop + dRsw_dctp + dRsw_dtau + dRsw_resid       # sum of SW components -- should equal SW total

    # Set SW fields to zero where the sun is down
    RR = np.isnan(Ksw0) #Ksw0.mask
    dRsw_true = np.where(RR,0,dRsw_true)
    dRsw_prop = np.where(RR,0,dRsw_prop)
    dRsw_dctp = np.where(RR,0,dRsw_dctp)
    dRsw_dtau = np.where(RR,0,dRsw_dtau)
    dRsw_resid = np.where(RR,0,dRsw_resid)

    # Move month axis back to the beginning 
    dRlw_true = np.array(np.moveaxis(dRlw_true,-1,0))
    dRlw_prop = np.array(np.moveaxis(dRlw_prop,-1,0))
    dRlw_dctp = np.array(np.moveaxis(dRlw_dctp,-1,0))
    dRlw_dtau = np.array(np.moveaxis(dRlw_dtau,-1,0))
    dRlw_resid = np.array(np.moveaxis(dRlw_resid,-1,0))
    dRsw_true = np.array(np.moveaxis(dRsw_true,-1,0))
    dRsw_prop = np.array(np.moveaxis(dRsw_prop,-1,0))
    dRsw_dctp = np.array(np.moveaxis(dRsw_dctp,-1,0))
    dRsw_dtau = np.array(np.moveaxis(dRsw_dtau,-1,0))
    dRsw_resid = np.array(np.moveaxis(dRsw_resid,-1,0))
    dc_star = np.array(np.moveaxis(dc_star,-1,0))
    dc_prop = np.array(np.moveaxis(dc_prop,-1,0))

    return (dRlw_true, dRlw_prop, dRlw_dctp, dRlw_dtau, dRlw_resid, 
            dRsw_true, dRsw_prop, dRsw_dctp, dRsw_dtau, dRsw_resid,
            dc_star, dc_prop)
