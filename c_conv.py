# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de
# climate_convert

import pandas as pd
import numpy as np
import xarray as xr


def absT(T):
    '''
    converts absolute temperature (K) into temperature in degC
    '''
    return T-273.15

def vFlux_simp(fl):
    '''
    converts mass flux to volume flux of water.
    assumes a density of 1kg/L
    input: fl (in kg m-2 s-1)
    '''
    if (type(fl)==xr.core.dataset.Dataset) | (type(fl)==xr.core.dataarray.DataArray):
        tix = np.float(np.mean(np.diff(fl.time)/86400000000000.))
    else:
        tix = np.float(np.mean(np.diff(fl.index)/86400000000000.))
    return fl*86400.*tix

def vFlux(fl,Tc):
    '''
    converts mass flux to volume flux of water.
    input: fl (in kg m-2 s-1), Tc (in degC)
    '''

    def wd(T):
        # T needs to be given in °C
        # returns rho in kg/L
        a1 = -3.983035  # °C
        a2 = 301.797  # °C
        a3 = 522528.9  # °C2
        a4 = 69.34881  # °C
        a5 = 999974.950  # g/m3
        return (a5 * (1 - ((T + a1) ** 2 * (T + a2)) / (a3 * (T + a4)))) / 1000000.

    if (type(fl)==xr.core.dataset.Dataset) | (type(fl)==xr.core.dataarray.DataArray):
        tix = np.float(np.mean(np.diff(fl.time) / 86400000000000.))
    else:
        tix = np.float(np.mean(np.diff(fl.index) / 86400000000000.))

    return (fl/wd(Tc)) * 86400. * tix

def crad(ra):
    '''
    converts radiation (W m-2) into kJh m-2 day-1
    '''
    return ra * 86.400

def RHe(RH,T):
    '''
    converts relative humidity to vapor pressure
    input: RH (-), T (in K)
    based on Clausius-Clapeyron
    '''
    return 0.001 * (RH * (611. * np.exp((17.67 * (T - 273.15)) / (T - 29.65))))

def sh2rh(sh,tc,ap=1013.25,boundcontrol=False):
    '''Convert specific humidity to relative humidity
    from Bolton 1980 The computation of Equivalent Potential Temperature
    http://www.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    sh :: specific humidity (kg/kg)
    tc :: air temperature (C)
    ap :: air pressure (hPa)

    returns relative humidity
    '''

    es = 6.112 * np.exp((17.67 * tc)/(tc + 243.5))
    e = sh * ap / (0.378 * sh + 0.622)
    rh = e / es
    if boundcontrol:
        rh[rh > 1.] = 1.
        rh[rh < 0.] = 0.

    return rh

