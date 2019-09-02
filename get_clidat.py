# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de

import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import timedelta

from contrail.security.onlineca.client import OnlineCaClient
from pyesgf.search import SearchConnection
from pyesgf.logon import LogonManager

from pydap.client import open_url
from pydap.cas.esgf import setup_session

import get_climate_data.esgf_credentials as cred

def ESGFquery(project='CORDEX', experiment='rcp85', time_frequency='day', domain='EUR-11', variable='pr', search_conn='http://esgf-data.dkrz.de/esg-search'):
    '''function to connect with ESGF server and to query for data
    all parameters must be defined. returns pyesgf.search.context object for further usage
    to switch off variable give it False as parameter'''

    #login
    lm = LogonManager()

    lm.logon_with_openid(openid=cred.OPENID, password=cred.PWD)
    #lm.is_logged_on()

    #query ESGF
    conn = SearchConnection(search_conn, distrib=False)

    if (project == 'CORDEX') | (project == 'reklies-index') | (project == 'CORDEX-Reklies'):
        ctx = conn.new_context(project=project, experiment=experiment, time_frequency=time_frequency, domain=domain, variable=variable)
    else:
        if variable:
            ctx = conn.new_context(project=project, experiment=experiment, time_frequency=time_frequency, variable=variable)
        else:
            ctx = conn.new_context(project=project, experiment=experiment, time_frequency=time_frequency)

    lm.logoff()
    return ctx

def ESGFgetid(ctx):
    ''' return pd.Series of dataset_ids from pyesgf.search.context'''
    dids = np.repeat('qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwtzuiopqwetzuiopqwertzuiopqwertzuiop', len(ctx.search()))
    for i in np.arange(len(ctx.search())):
        dids[i] = ctx.search()[i].dataset_id
    dids = pd.DataFrame(dids,columns=['dataset_id'])

    dids['model'] ='qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiop'
    dids['variable'] = 'qwertzuiopqwert'
    dids['ensemble'] = 'qwertzuiopqwert'
    dids['RCM'] = 'qwertzuiopqwert'
    dids['experiment'] = 'qwertzuiopqwert'

    for i in dids.index:
        if dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CORDEX':
            dids.loc[i,'model'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[4]
            dids.loc[i,'variable'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[-2]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[6]
            dids.loc[i, 'RCM'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[7]
            dids.loc[i, 'experiment'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[5]
        #elif dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CMIP5':
        else:
            dids.loc[i,'model'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[3]
            dids.loc[i,'variable'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[6]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[-2]
            dids.loc[i, 'RCM'] = 'none'
            dids.loc[i, 'experiment'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[4]

    return dids

def ESGFfiles(ctx,dix,vari=None):
    '''return list of files from pyesgf.search.context and ONE specific dataset_id line
    dix is something like dids.loc[10]
    '''
    files = ctx.search()[dix.name].file_context().search()
    fix = np.repeat('qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwtzuiopqwetzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwtzuiopqwetzuiopqwertzuiopqwertzuiop', len(files))
    i = 0
    for file in files:
        if file.opendap_url == None:
            fix[i] = file.urls['HTTPServer'][0][0]
        else:
            fix[i] = file.opendap_url
        i += 1

    return fix


def ESGF_getdata(url,lat_min=53.1,lat_max=53.9,lon_min=6.8,lon_max=8.4):
    '''Download data from ESGF repository of one nc file for given spatial extent.
    Standard is the extent of Eastern Frisia at the German North Sea coast.
    Returns xarray with data.'''

    if 'ceda' in cred.OPENID:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url, username=cred.cedaUSR)
    else:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url)
    dataset = open_url(url, session=session)

    lats = dataset.lat[:].data[0]
    lons = dataset.lon[:].data[0]
    attr_n = list(dataset.keys())[-1]

    frisia_sub = ((lats > lat_min) & (lats < lat_max) & (lons > lon_min) & (lons < lon_max))

    # box
    fsu1 = min(np.where(frisia_sub)[0])
    fsu2 = max(np.where(frisia_sub)[0])
    fsu3 = min(np.where(frisia_sub)[1])
    fsu4 = max(np.where(frisia_sub)[1])

    # data
    dummy = dataset[attr_n][:, fsu1:fsu2, fsu3:fsu4].data[0]

    # timestamp
    ti = np.repeat(pd.to_datetime('1949-12-01 00:00:00'), dataset.time.shape[0])
    dt = dataset.time[:].data[:]
    for i in np.arange(dataset.time.shape[0]):
        ti[i] = ti[i] + timedelta(days=dt[i])

    # build xarray
    dsx = xr.Dataset({attr_n: (['time', 'x', 'y'], dummy)},
                     coords={'lon': (['x', 'y'], lons[fsu1:fsu2, fsu3:fsu4]),
                             'lat': (['x', 'y'], lats[fsu1:fsu2, fsu3:fsu4]),
                             'time': ti, 'reference_time': ti[0]})

    dsx.attrs['var_name'] = attr_n
    if (attr_n == 'pr') | (attr_n[:2] == 'ev'):
        dsx.attrs['units'] = 'kg m-2 s-1'
    elif (attr_n == 'tauu') | (attr_n == 'tauv') | (attr_n == 'ps') | (attr_n == 'psl'):
        dsx.attrs['units'] = 'Pa'
    elif (attr_n[:2] == 'ta') | (attr_n[:2] == 'ts'):
        dsx.attrs['units'] = 'K'
    elif (attr_n[-2:] == 'ds'):
        dsx.attrs['units'] = 'W m-2'

    return dsx

def ESGF_getstack(ctx,dix,lat_min=53.1,lat_max=53.9,lon_min=6.8,lon_max=8.4,storex=True,data_out=True):
    '''Download data from ESGF repository of one dataset for given spatial extent.
        Standard is the extent of Eastern Frisia at the German North Sea coast.
        Returns xarray with data.
        This takes about 0.5 h per stack - due to the slow response of the servers.'''

    import progressbar

    #files:
    fix = ESGFfiles(ctx, dix)

    counti = 0
    countfail = 0
    with progressbar.ProgressBar(max_value=len(fix)) as bar:
        for url in fix:
            try:
                dsx_i = ESGF_getdata(url,lat_min,lat_max,lon_min,lon_max)
                countfail = 0
            except:
                countfail += 1
                try:
                    dsx_i = ESGF_getdata(url, lat_min, lat_max, lon_min, lon_max)
                    countfail = 0
                except:
                    pass

            if (counti == 0) & (countfail == 0):
                dsm = dsx_i
            elif (counti == 0) & (countfail > 0):
                print('failed to load data ... sorry.')
                if data_out:
                    return xr.DataArray(['failed to load data ... sorry.'])
                else:
                    return False
            elif countfail == 0:
                dsm = xr.concat([dsm, dsx_i], dim='time')
            elif countfail > 4:
                print('failed to load data 4 times ... sorry.')
                if data_out:
                    return xr.DataArray(['failed to load data 4 times ... sorry.'])
                else:
                    return False
            counti += 1
            bar.update(counti)

    dsm.attrs['dataset_id'] = dix.dataset_id

    if storex:
        dsm.sortby('time').to_netcdf(dix.dataset_id.split('|')[0] + '.frisia.nc')

    print('success. check output and dataset '+dix.dataset_id.split('|')[0])
    if data_out:
        return dsm
    else:
        return True

def ESGF_splot(dsx,ti='2030-11-30'):
    '''plot map of one time frame (tix) in xarray of climate data'''
    from cartopy import config
    import cartopy.crs as ccrs
    import matplotlib as plt

    dummy = dsx.sel(time=ti)[dsx.attrs['var_name']].data[0]
    dummy[dummy > 1e15] = np.nan #remove nan values

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines('10m')
    plt.contourf(dsx.lon, dsx.lat, dummy, 60, transform=ccrs.PlateCarree())
    plt.colorbar()
    plt.title(dsx.attrs['var_name']+' @ '+str(pd.to_datetime(dsx.time[tix].data).date()))
    return

def ESGF_tplot(dsx,x=2,y=4):
    '''plot tiem series of xarray of climate data'''
    dsx.sel(x=x, y=y)[dsx.attrs['var_name']].plot()
    return
