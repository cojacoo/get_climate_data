# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de

import pandas as pd
import numpy as np
import xarray as xr
import numba

import pyeto
from climate_indices import indices
from get_climate_data.waterdensity import waterdensity as wd

from netCDF4 import Dataset
from datetime import timedelta
import dateutil

try:
    from contrail.security.onlineca.client import OnlineCaClient
except:
    try:
        from contrail.security.onlineca.client import OnlineCaClient
    except:
        print('- - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - -')
        print('WARNING: CEDA online authentification client could not be loaded.')
        print('- - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - - ! - - -')

from pyesgf.search import SearchConnection
from pyesgf.logon import LogonManager

from pydap.client import open_url
from pydap.cas.esgf import setup_session

try:
    import get_climate_data.esgf_credentials as cred
except:
    print('Hello first-time user!')
    print('Please rename the esgf_credential_dummy.py file to esgf_credential.py after entering your information.')
    print('Make sure not to include the file in the git repository to protect your login.')
    raise

def ESGFquery(project='CORDEX', experiment='rcp85', time_frequency='day', domain='EUR-11', variable='pr', search_conni=0, ov=True):
    '''function to connect with ESGF server and to query for data
    all parameters must be defined. returns pyesgf.search.context object for further usage
    to switch off variable give it False as parameter'''

    #login
    lm = LogonManager()

    if 'ceda' in cred.OPENID:
        lm.logon_with_openid(openid=cred.OPENID, password=cred.PWD, username=cred.cedaUSR)
    else:
        lm.logon_with_openid(openid=cred.OPENID, password=cred.PWD)
    #lm.is_logged_on()

    search_conn = np.array(['http://esgf-data.dkrz.de/esg-search','http://esgf-index1.ceda.ac.uk/esg-search', 'http://esgf-node.llnl.gov/esg-search'])
    #query ESGF
    conn = SearchConnection(search_conn[search_conni], distrib=False)

    if (project == 'CORDEX') | (project == 'reklies-index') | (project == 'CORDEX-Reklies'):
        ctx = conn.new_context(project=project, experiment=experiment, time_frequency=time_frequency, domain=domain, variable=variable)
    elif (project == 'CMIP6'):
        if ov:
            ctx = conn.new_context(project='CMIP6', variable=variable, frequency=time_frequency, experiment_id=experiment, member_id='r1i1p1f1')
        else:
            ctx = conn.new_context(project='CMIP6', variable=variable, frequency=time_frequency, experiment_id=experiment)
    else:
        if (variable=='atmos') | (variable=='land') | (variable=='ocean'):
            ctx = conn.new_context(project=project, experiment=experiment, time_frequency=time_frequency, realm=variable)
        elif variable:
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
        if (dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CORDEX') | (dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'cordex') | (dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'cordex-reklies'):
            dids.loc[i,'model'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[4]
            dids.loc[i,'variable'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[-2]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[6]
            dids.loc[i, 'RCM'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[7]
            dids.loc[i, 'experiment'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[5]
        elif dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CMIP5':
            dids.loc[i, 'model'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[3]
            dids.loc[i, 'variable'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[6]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[8]
            dids.loc[i, 'RCM'] = 'none'
            dids.loc[i, 'experiment'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[4]
        elif dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CMIP6':
            dids.loc[i, 'model'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[3]
            dids.loc[i, 'variable'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[7]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[5]
            dids.loc[i, 'RCM'] = 'none'
            dids.loc[i, 'experiment'] = dids.loc[i, 'dataset_id'].split('|')[0].split('.')[4]
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
    if type(dix)==pd.Series:
        files = ctx.search()[dix.name].file_context().search()
    else:
        files = ctx.search()[dix.index[0]].file_context().search()
    fix = np.repeat('qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwtzuiopqwetzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwtzuiopqwetzuiopqwertzuiopqwertzuiop', len(files))
    i = 0
    for file in files:
        if file.opendap_url == None:
            fix[i] = file.urls['HTTPServer'][0][0]
        else:
            fix[i] = file.opendap_url
        i += 1

    return fix

def ESGF_prepquick(url,lat_min=53.1,lat_max=53.9,lon_min=6.8,lon_max=8.4):
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

    region_sub = ((lats > lat_min) & (lats < lat_max) & (lons > lon_min) & (lons < lon_max))

    # box
    subidx = [min(np.where(region_sub)[0]),max(np.where(region_sub)[0]),min(np.where(region_sub)[1]),max(np.where(region_sub)[1])]
    lonsx = lons[subidx[0]:subidx[1], subidx[2]:subidx[3]]
    latsx = lats[subidx[0]:subidx[1], subidx[2]:subidx[3]]

    return subidx, lonsx, latsx

def ESGF_quickNSC_CORDEX(url,vari,subidx, lonsx, latsx):
    if 'ceda' in cred.OPENID:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url, username=cred.cedaUSR)
    else:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url)
    dataset = open_url(url, session=session)

    dummy = dataset[vari][:,subidx[0]:subidx[1], subidx[2]:subidx[3]].data[0]

    # timestamp
    ti = np.repeat(pd.to_datetime('1949-12-01 00:00:00'), dataset.time.shape[0])
    dt = dataset.time[:].data[:]
    for i in np.arange(dataset.time.shape[0]):
        ti[i] = ti[i] + timedelta(days=dt[i])

    # build xarray
    dsx = xr.Dataset({vari: (['time', 'x', 'y'], dummy)},
                     coords={'lon': (['x', 'y'], lonsx),
                             'lat': (['x', 'y'], latsx),
                             'time': ti})
    return dsx

def ESGF_getdata(url,lat_min=53.1,lat_max=53.9,lon_min=6.8,lon_max=8.4):
    '''Download data from ESGF repository of one nc file for given spatial extent.
    Standard is the extent of Eastern Frisia at the German North Sea coast.
    Returns xarray with data.'''

    if 'ceda' in cred.OPENID:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url, username=cred.cedaUSR)
    else:
        session = setup_session(cred.OPENID, cred.PWD, check_url=url)
    dataset = open_url(url, session=session)

    ky = list(dataset.keys())
    vy = ['tas', 'tasmax', 'tasmin', 'pr', 'prhmax', 'ps', 'huss', 'hurs', 'hus','sfcWind', 'sfcWindmax', 'rsds', 'evspsbl', 'mrso', 'prc', 'rhs', 'psl', 'dtr']

    for a in (set(vy) & set(ky)):
        attr_n = a
        break

    if ('mpi-ge' in url) | ('cmip5' in url) | ('cmip6' in url):
        lats = dataset.lat[:].data[:]
        lons = dataset.lon[:].data[:]

        #grand ensemble requires different treatment
        region_sub_lat = ((lats > lat_min) & (lats < lat_max))
        region_sub_lon = ((lons > lon_min) & (lons < lon_max))

        printi = False
        if sum(region_sub_lat)<4:
            while sum(region_sub_lat)<4:
                lat_min -= 0.2
                lat_max += 0.2
                region_sub_lat = ((lats > lat_min) & (lats < lat_max))
                printi = True

        if sum(region_sub_lon)<4:
            while sum(region_sub_lon)<4:
                lon_min -= 0.2
                lon_max += 0.2
                region_sub_lon = ((lons > lon_min) & (lons < lon_max))
                printi = True

        if printi:
            print('expanded lats to between ' + str(lat_min) + ' and ' + str(lat_max))
            print('expanded lons to between ' + str(lon_min) + ' and ' + str(lon_max))

        subidx = [min(np.where(region_sub_lat)[0]), max(np.where(region_sub_lat)[0]), min(np.where(region_sub_lon)[0]), max(np.where(region_sub_lon)[0])]
        try:
            data = dataset[attr_n][:, subidx[0]:subidx[1] + 1, subidx[2]:subidx[3] + 1].data[:]
            dummy = data[0]
            latx = data[2]
            lonx = data[3]
        except:
            try:
                dummy = dataset[attr_n].data[0][:, subidx[0]:subidx[1] + 1, subidx[2]:subidx[3] + 1]
                #latx = dataset['lat'].data[subidx[0]:subidx[1] + 1]
                latx = np.tile(dataset['lat'].data[subidx[0]:subidx[1] + 1],(subidx[3]-subidx[2])+1).reshape(((subidx[3]-subidx[2])+1,(subidx[1]-subidx[0]) + 1)).T
                #lonx = dataset['lon'].data[subidx[2]:subidx[3] + 1]
                lonx = np.repeat(dataset['lon'].data[subidx[2]:subidx[3] + 1],(subidx[1]-subidx[0])+1).reshape(((subidx[3]-subidx[2])+1,(subidx[1]-subidx[0]) + 1)).T
            except:
                dummy = dataset['.'.join([attr_n,attr_n])].data[0][:, subidx[0]:subidx[1] + 1, subidx[2]:subidx[3] + 1]
                #latx = dataset['.'.join([attr_n,'lat'])].data[subidx[0]:subidx[1] + 1]
                latx = np.tile(dataset['.'.join([attr_n,'lat'])].data[subidx[0]:subidx[1] + 1], (subidx[3] - subidx[2]) + 1).reshape(
                    ((subidx[3] - subidx[2]) + 1, (subidx[1] - subidx[0]) + 1)).T
                #lonx = dataset['.'.join([attr_n,'lon'])].data[subidx[0]:subidx[1] + 1]
                lonx = np.repeat(dataset['.'.join([attr_n,'lon'])].data[subidx[2]:subidx[3] + 1], (subidx[1] - subidx[0]) + 1).reshape(
                    ((subidx[3] - subidx[2]) + 1, (subidx[1] - subidx[0]) + 1)).T

        if np.shape(latx) != np.shape(dummy)[1:]:
            latx = np.repeat(latx, np.shape(dummy)[1:][1]).reshape(np.shape(dummy)[1:])
        if np.shape(lonx) != np.shape(dummy)[1:]:
            lonx = np.tile(lonx, np.shape(dummy)[1:][0]).reshape(np.shape(dummy)[1:])

        #timestamp
        if ('cmip5' in url):
            start_date = pd.to_datetime(url.split('_')[-1].split('.')[0].split('-')[0], format='%Y%m%d')
            ti = start_date + (dataset.time.data[:]-np.floor(dataset.time.data[0][0]))*timedelta(days=1)
        elif 'days since' in dataset.time.attributes['units']:
            if int(dataset.time.attributes['units'][11:][:4])>1770:
                start_date = pd.to_datetime(dataset.time.attributes['units'][11:])
                ti = start_date + pd.to_timedelta(dataset.time.data[:],'days')
            else:
                sd_dummy = dataset.time.attributes['units'][11:]
                sd_dummy1 = '1900'+sd_dummy[4:]
                start_date = pd.to_datetime(sd_dummy1)
                sd_dummy_offset = (dateutil.parser.parse(sd_dummy1) - dateutil.parser.parse(sd_dummy)).days - (dateutil.parser.parse(url.split('_')[-1].split('.')[0].split('-')[0]) - (start_date + pd.to_timedelta(dataset.time.data[0]-(dateutil.parser.parse(sd_dummy1) - dateutil.parser.parse(sd_dummy)).days, 'days'))).days[0]
                ti = start_date + pd.to_timedelta(dataset.time.data[:]-sd_dummy_offset, 'days')

        else:
            start_date = pd.to_datetime('2005-01-01 00:00:00')
            ti = start_date + dataset.time.data[:] * timedelta(days=1)

    else:
        lats = dataset.lat[:].data[0]
        lons = dataset.lon[:].data[0]

        frisia_sub = ((lats > lat_min) & (lats < lat_max) & (lons > lon_min) & (lons < lon_max))
        # box
        fsu1 = min(np.where(frisia_sub)[0])
        fsu2 = max(np.where(frisia_sub)[0])
        fsu3 = min(np.where(frisia_sub)[1])
        fsu4 = max(np.where(frisia_sub)[1])

        # data
        dummy = dataset[attr_n].data[0]
        if len(np.shape(dummy)) == 3:
            dummy = np.squeeze(dummy[:,fsu1:fsu2, fsu3:fsu4])
        elif len(np.shape(dummy)) == 4:
            dummy = np.squeeze(dummy[:,0,fsu1:fsu2, fsu3:fsu4])
        else:
            print('Dataset has wrong dimensionality!')
            return

        lonx = lons[fsu1:fsu2, fsu3:fsu4]
        latx = lats[fsu1:fsu2, fsu3:fsu4]

        # timestamp
        ti = np.repeat(pd.to_datetime('1949-12-01 00:00:00'), dataset.time.shape[0])
        dt = dataset.time[:].data[:]
        for i in np.arange(dataset.time.shape[0]):
            ti[i] = ti[i] + timedelta(days=dt[i])

    # build xarray
    dsx = xr.Dataset({attr_n: (['time', 'x', 'y'], dummy)},
                     coords={'lon': (['x', 'y'], lonx),
                             'lat': (['x', 'y'], latx),
                             'time': ti})

    dsx.attrs['var_name'] = attr_n
    if (attr_n == 'pr') | (attr_n[:2] == 'ev'):
        dsx.attrs['units'] = 'kg m-2 s-1'
    elif (attr_n == 'tauu') | (attr_n == 'tauv') | (attr_n == 'ps') | (attr_n == 'psl'):
        dsx.attrs['units'] = 'Pa'
    elif (attr_n[:2] == 'ta') | (attr_n[:2] == 'ts'):
        dsx.attrs['units'] = 'K'
    elif (attr_n == 'huss') | (attr_n == 'hurs'):
        dsx.attrs['units'] = 'm s-2'
    elif (attr_n == 'sfcWind'):
        dsx.attrs['units'] = 'Pa'
    elif (attr_n[-2:] == 'ds'):
        dsx.attrs['units'] = 'W m-2'

    return dsx

def ESGF_getstack(ctx,dix,vry=None,lat_min=53.1,lat_max=53.9,lon_min=6.8,lon_max=8.4,storex=True,data_out=True,storext='.frisia.nc',wrkdir=''):
    '''Download data from ESGF repository of one dataset for given spatial extent.
        Standard is the extent of Eastern Frisia at the German North Sea coast.
        Returns xarray with data.
        This takes about 0.5 h per stack - due to the slow response of the servers.'''

    import progressbar
    #files:
    fix = ESGFfiles(ctx, dix)
    fiv = [x.split('/')[-1].split('_')[0] for x in fix]
    fix1 = pd.DataFrame([fix, fiv], index=['fix', 'vari']).T
    if vry==None:
        print('processing '+str(len(fix1))+' files.')
    else:
        fix1 = fix1.loc[fix1.vari == vry]
        print('processing ' + str(len(fix1)) + ' files.')

    if len(fix1.vari.unique())==1:
        counti = 0
        countfail = 0
        with progressbar.ProgressBar(max_value=len(fix1)) as bar:
            for url in fix1.iloc[:,0].values:
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
            if vry == None:
                dsm.sortby('time').to_netcdf(wrkdir + dix.dataset_id.split('|')[0] + storext)
            else:
                dsm.sortby('time').to_netcdf(wrkdir + dix.dataset_id.split('|')[0] + '_' + vry + storext)

    else:
        with progressbar.ProgressBar(max_value=len(fix)) as bar:
            vari_u = list(set(['tas', 'tasmax', 'tasmin', 'pr', 'ps', 'huss', 'sfcWind', 'rsds', 'evspsbl', 'rhs', 'psl']).intersection(list(fix1.vari.unique())))
            for varix in vari_u:
                for url in fix1.loc[fix1.vari==varix,'fix']:
                    counti = 0
                    countfail = 0
                    try:
                        dsx_i = ESGF_getdata(url, lat_min, lat_max, lon_min, lon_max)
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

                dsm.attrs['dataset_id'] = '_'.join(fix1.loc[fix1.vari==varix,'fix'].split('/')[-1].split('_')[:-1])

                if storex:
                    dsm.sortby('time').to_netcdf(wrkdir + '_'.join(fix1.loc[fix1.vari==varix,'fix'].split('/')[-1].split('_')[:-1]) + storext)

    print('success. check output and dataset '+dix.dataset_id.split('|')[0])
    if data_out:
        return dsm
    else:
        return True

def ESGF_splot(dsx,ti='2030-11-30',stepsx=10):
    '''plot map of one time frame (tix) in xarray of climate data'''
    from cartopy import config
    import cartopy.crs as ccrs
    import matplotlib.pyplot

    if len(dsx.sel(time=ti)[dsx.attrs['var_name']].data) == 0:
        dummy = dsx.sel(time=slice(ti, str((pd.to_datetime(ti) + timedelta(days=40)).date())))[dsx.attrs['var_name']].data[0]
    else:
        dummy = dsx.sel(time=ti)[dsx.attrs['var_name']].data[0]

    dummy[dummy > 1e15] = np.nan #remove nan values

    ax = axes(projection=ccrs.PlateCarree())
    ax.coastlines('10m')
    contourf(dsx.lon, dsx.lat, dummy, stepsx, transform=ccrs.PlateCarree())
    colorbar()
    title(dsx.attrs['var_name']+' @ '+ti)
    return

def ESGF_tplot(dsx,x=2,y=4):
    '''plot tiem series of xarray of climate data'''
    dsx.sel(x=x, y=y)[dsx.attrs['var_name']].plot()
    return

def ESGF_cordex(redmodel=False,vy=None,expi='rcp85',lat_min=52.5, lat_max=55., lon_min=6.7, lon_max=11.,storext='.NSc.nc',wrkdir='',search_conni=0):
    import os

    try:
        if vy == None:
            # get all
            # vy = ['tas', 'tasmax', 'tasmin', 'pr', 'ps', 'huss', 'hurs', 'sfcWind', 'rsds']
            vy = ['tas', 'tasmax', 'tasmin', 'pr', 'prhmax', 'ps', 'huss', 'hurs', 'sfcWind', 'sfcWindmax', 'rsds', 'evspsbl', 'mrso', 'prc']
    except:
        pass

    for vari in vy:
        # querry ESGF data
        ctx_m = ESGFquery(project='CORDEX', experiment=expi, time_frequency='day', domain='EUR-11', variable=vari, search_conni=search_conni)
        di_m = ESGFgetid(ctx_m)
        if redmodel:
            di_m1 = di_m.loc[(di_m.model == redmodel) & ((di_m.variable == 'r1i1p1') | (di_m.ensemble == 'r1i1p1'))]
            print('ready to get ' + str(len(di_m1)) + ' model results for ' + model + ' and ' + vari + ' in ' + expi)
        else:
            di_m1 = di_m
            print('ready to get ' + str(len(di_m1)) + ' model results for ' + vari + ' in ' + expi)


        for j in di_m1.index:
            if ~os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0]+storext):
                try:
                    dsm = ESGF_getstack(ctx_m, di_m1.loc[j], lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                    print('processed ' + di_m1.loc[j,'dataset_id'])
                except:
                    print(di_m1.loc[j, 'dataset_id']+' failed processing')
                    print('Restart manually.')
            else:
                print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext+' already exists.')
                print(di_m1.loc[j, 'dataset_id'] + ' not processed again.')

    return


def ESGF_reklies(expi='rcp85',lat_min=52.5, lat_max=55., lon_min=6.7, lon_max=11.,storext='.NSc.nc',wrkdir=''):
    import os
    vy = ['tas', 'tasmax', 'tasmin', 'pr', 'ps', 'huss', 'hurs', 'sfcWind', 'rsds']

    for vari in vy:
        # querry ESGF data
        ctx_m = ESGFquery(project='CORDEX-Reklies', experiment=expi, time_frequency='day', domain='EUR-11', variable=vari)
        di_m = ESGFgetid(ctx_m)
        di_m1 = di_m.loc[di_m.ensemble == 'r1i1p1'] #only r1i1p1 results

        print('ready to get '+str(len(di_m1))+' model results for '+vari+' in '+expi)
        for j in di_m1.index:
            #apparently 
            if os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0]+storext):
                print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext + ' already exists.')
                print(di_m1.loc[j, 'dataset_id'] + ' not processed again.')
            else:
                try:
                    print('started to process ' + di_m1.loc[j, 'dataset_id'])
                    dsm = ESGF_getstack(ctx_m, di_m1.loc[j], lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                    print('processed ' + di_m1.loc[j,'dataset_id'])
                except:
                    print(di_m1.loc[j, 'dataset_id']+' failed processing')
                    print('Restart manually.')

    return

def ESGF_mpi_ge(expi='rcp85', vy = ['tas', 'pr', 'ps', 'hurs', 'sfcWind', 'rsdt', 'evspsbl','prc'], lat_min=50.5, lat_max=57., lon_min=5.5, lon_max=12.2,storext='.NScl.nc',wrkdir=''):
    import os


    for vari in vy:
        # querry ESGF data
        ctx_m = ESGFquery(project='MPI-GE', experiment=expi, time_frequency='mon', variable=vari)
        di_m = ESGFgetid(ctx_m)
        di_m1 = di_m.loc[di_m.variable == 'atmos'] #only atmos realm

        print('ready to get '+str(len(di_m1))+' model results for '+vari+' in '+expi)
        for j in di_m1.index:
            if os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0]+storext):
                print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext + ' already exists. not processed again.')
            else:
                try:
                    print('started to process ' + di_m1.loc[j, 'dataset_id'])
                    dsm = ESGF_getstack(ctx_m, di_m1.loc[j], lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                    print('processed ' + di_m1.loc[j,'dataset_id'])
                except:
                    print(di_m1.loc[j, 'dataset_id']+' failed processing')
                    print('Restart manually.')

    return

def ESGF_cmip_mon(expi='rcp85', vy = ['tas','tasmax','tasmin', 'pr', 'ps', 'huss', 'sfcWind', 'rsds', 'evspsbl'], lat_min=50.5, lat_max=57., lon_min=5.5, lon_max=12.2,storext='.NScl.nc',wrkdir=''):
    import os


    for vari in vy:
        # querry ESGF data
        ctx_m = ESGFquery(project='CMIP5', experiment=expi, time_frequency='mon', variable=vari)
        di_m = ESGFgetid(ctx_m)
        di_m1 = di_m.loc[di_m.variable == 'atmos'] #only atmos realm

        print('ready to get '+str(len(di_m1))+' model results for '+vari+' in '+expi)
        for j in di_m1.index:
            if os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0]+storext):
                print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext + ' already exists. not processed again.')
            else:
                try:
                    print('started to process ' + di_m1.loc[j, 'dataset_id'])
                    dsm = ESGF_getstack(ctx_m, di_m1.loc[j], lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                    print('processed ' + di_m1.loc[j,'dataset_id'])
                except:
                    print(di_m1.loc[j, 'dataset_id']+' failed processing')
                    print('Restart manually.')

    return

def ESGF_cmip_day(expi='rcp85', vy = ['tas','tasmax','tasmin', 'pr', 'psl', 'hus', 'sfcWind', 'rsds', 'rhs'], lat_min=50.5, lat_max=57., lon_min=5.5, lon_max=12.2,storext='.NScl.nc',wrkdir='',sci=0):
    import os

    for vari in vy:
        # querry ESGF data
        ctx_m = ESGFquery(project='CMIP5', experiment=expi, time_frequency='day', variable=vari, search_conni=sci)
        di_m = ESGFgetid(ctx_m)
        di_m1 = di_m.loc[di_m.variable == 'atmos'] #only atmos realm

        print('ready to get '+str(len(di_m1))+' model results for '+vari+' in '+expi)
        for j in di_m1.index:
            if os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0] + '_' + vari +storext):
                print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext + ' already exists. not processed again.')
            else:
                try:
                    print('started to process ' + di_m1.loc[j, 'dataset_id'])
                    dsm = ESGF_getstack(ctx_m, di_m1.loc[j], vry=vari, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                    print('processed ' + di_m1.loc[j,'dataset_id'])
                except:
                    print(di_m1.loc[j, 'dataset_id']+' failed processing')
                    print('Restart manually.')

    return

def ESGF_cmip6_day(exps= 'all', vy = 'all', lat_min=50.5, lat_max=57., lon_min=5.5, lon_max=12.2, storext='.NScl.nc',wrkdir='',sci=0, ov=True):
    import os
    try:
        if exps == 'all':
            exps= ['historical','ssp119','ssp126','ssp245','ssp370','ssp434','ssp460','ssp585']
    except:
        pass
    try:
        if vy == 'all':
            vy = ['tas','tasmax','tasmin', 'pr', 'psl', 'hus', 'sfcWind', 'rsds','sfcWindmax']
    except:
        pass

    for expi in exps:
        for vari in vy:
            # querry ESGF data
            ctx_m = ESGFquery(project='CMIP6', experiment=expi, time_frequency='day', variable=vari, search_conni=sci, ov=ov)
            di_m1 = ESGFgetid(ctx_m)

            print('ready to get '+str(len(di_m1))+' model results for '+vari+' in '+expi)
            for j in di_m1.index:
                if os.path.isfile(wrkdir+di_m1.loc[j,'dataset_id'].split('|')[0] + '_' + vari +storext):
                    print(di_m1.loc[j, 'dataset_id'].split('|')[0] + storext + ' already exists. not processed again.')
                else:
                    try:
                        print('started to process ' + di_m1.loc[j, 'dataset_id'])
                        dsm = ESGF_getstack(ctx_m, di_m1.loc[j], vry=vari, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,storext=storext,wrkdir=wrkdir)
                        print('processed ' + di_m1.loc[j,'dataset_id'])
                    except:
                        print(di_m1.loc[j, 'dataset_id']+' failed processing')
                        print('Restart manually.')

    return


def nc_inventory(files):
    '''
    build table of attributes from file list from ESGF downloads
    The input can be generated by line magic like this:
    files = !ls /Volumes/ConradSSD/climate/MPI_GE/*nc

    returns dataframe with most attributes and a model-result-code
    '''


    dids = pd.DataFrame(files, columns=['file'])
    dids['model'] = 'qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiop'
    dids['code'] = 'qwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiopqwertzuiop'
    dids['institute'] = 'qwertzuiopqwert'
    dids['variable'] = 'qwertzuiopqwert'
    dids['ensemble'] = 'qwertzuiopqwert'
    dids['RCM'] = 'qwertzuiopqwert'
    dids['experiment'] = 'qwertzuiopqwert'

    for i in dids.index:
        if dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'cordex':
            dids.loc[i, 'institute'] = dids.loc[i, 'file'].split('/')[-1].split('.')[3]
            dids.loc[i, 'model'] = '-'.join(dids.loc[i, 'file'].split('/')[-1].split('.')[4].split('-')[1:])
            dids.loc[i, 'variable'] = dids.loc[i, 'file'].split('/')[-1].split('.')[10]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'file'].split('/')[-1].split('.')[6]
            dids.loc[i, 'RCM'] = dids.loc[i, 'file'].split('/')[-1].split('.')[7]
            dids.loc[i, 'experiment'] = dids.loc[i, 'file'].split('/')[-1].split('.')[5]
            dids.loc[i, 'code'] = '.'.join(
                [dids.loc[i, 'institute'], dids.loc[i, 'model'], dids.loc[i, 'RCM'], dids.loc[i, 'ensemble'], dids.loc[i, 'experiment']])
        # elif dids.loc[i, 'dataset_id'].split('|')[0].split('.')[0] == 'CMIP5':
        elif dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'mpi-ge':
            dids.loc[i, 'institute'] = 'MPI_GE'
            dids.loc[i, 'model'] = dids.loc[i, 'file'].split('/')[-1].split('.')[3]
            dids.loc[i, 'variable'] = dids.loc[i, 'file'].split('/')[-1].split('.')[7]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'file'].split('/')[-1].split('.')[8]
            dids.loc[i, 'RCM'] = dids.loc[i, 'file'].split('/')[-1].split('.')[6]
            dids.loc[i, 'experiment'] = dids.loc[i, 'file'].split('/')[-1].split('.')[4]
            dids.loc[i, 'code'] = '.'.join(
                [dids.loc[i, 'institute'], dids.loc[i, 'model'], dids.loc[i, 'ensemble'], dids.loc[i, 'experiment']])
        elif (dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'cordex-reklies') | (dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'reklies-index'):
            dids.loc[i, 'institute'] = dids.loc[i, 'file'].split('/')[-1].split('.')[3]
            dids.loc[i, 'model'] = '-'.join(dids.loc[i, 'file'].split('/')[-1].split('.')[4].split('-')[1:])
            dids.loc[i, 'variable'] = dids.loc[i, 'file'].split('/')[-1].split('.')[10]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'file'].split('/')[-1].split('.')[8]
            dids.loc[i, 'RCM'] = dids.loc[i, 'file'].split('/')[-1].split('.')[7]
            dids.loc[i, 'experiment'] = dids.loc[i, 'file'].split('/')[-1].split('.')[5]
            dids.loc[i, 'code'] = '.'.join(
                [dids.loc[i, 'model'], dids.loc[i, 'ensemble'], dids.loc[i, 'RCM'], dids.loc[i, 'experiment']])
        elif dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'cmip5':
            dids.loc[i, 'institute'] = dids.loc[i, 'file'].split('/')[-1].split('.')[2]
            dids.loc[i, 'model'] = dids.loc[i, 'file'].split('/')[-1].split('.')[3]
            dids.loc[i, 'variable'] = dids.loc[i, 'file'].split('/')[-1].split('.')[9].split('_')[1]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'file'].split('/')[-1].split('.')[8]
            dids.loc[i, 'RCM'] = dids.loc[i, 'file'].split('/')[-1].split('.')[9].split('_')[0]
            dids.loc[i, 'experiment'] = dids.loc[i, 'file'].split('/')[-1].split('.')[4]
            dids.loc[i, 'code'] = '.'.join(
                [dids.loc[i, 'institute'], dids.loc[i, 'model'], dids.loc[i, 'ensemble'], dids.loc[i, 'RCM'], dids.loc[i, 'experiment']])
        elif dids.loc[i, 'file'].split('/')[-1].split('.')[0] == 'CMIP6':
            dids.loc[i, 'institute'] = dids.loc[i, 'file'].split('/')[-1].split('.')[2]
            dids.loc[i, 'model'] = dids.loc[i, 'file'].split('/')[-1].split('.')[3]
            dids.loc[i, 'variable'] = dids.loc[i, 'file'].split('/')[-1].split('.')[7]
            dids.loc[i, 'ensemble'] = dids.loc[i, 'file'].split('/')[-1].split('.')[5]
            dids.loc[i, 'RCM'] = dids.loc[i, 'file'].split('/')[-1].split('.')[9].split('_')[0]
            dids.loc[i, 'experiment'] = dids.loc[i, 'file'].split('/')[-1].split('.')[4]
            dids.loc[i, 'code'] = '.'.join(
                [dids.loc[i, 'institute'], dids.loc[i, 'model'], dids.loc[i, 'ensemble'], dids.loc[i, 'RCM'], dids.loc[i, 'experiment']])

    return dids

def read_CM(fix):
    '''
    read and merge xarrays of different variables of one model output
    fix is a list of files.

    e.g. new_xarray = read_CM(NSc.loc[NSc.code == NSc.code.unique()[5],'file'])

    returns new_xarray
    '''
    if type(fix) == pd.Series:
        fix = fix.values

    dummy = xr.open_dataset(fix[0])
    for fi in fix[1:]:
        dummy = xr.merge([dummy, xr.open_dataset(fi)])

    return dummy

def readconv_CM(NSc,simp=True):
    import get_climate_data.c_conv as cc
    dummy = read_CM(NSc.file)

    dummy['Tc'] = cc.absT(dummy.tas)
    for vari in NSc.variable:
        if vari == 'pr':
            if simp:
                dummy['prec'] = cc.vFlux_simp(dummy.pr)
            else:
                dummy['prec'] = cc.vFlux(dummy.pr,dummy.Tc)
        elif vari=='tas':
            pass
        elif vari == 'tasmin':
            dummy['Tmnc'] = cc.absT(dummy.tasmin)
        elif vari == 'tasmax':
            dummy['Tmxc'] = cc.absT(dummy.tasmax)
        elif vari == 'rsds':
            dummy['rad'] = cc.crad(dummy.rsds)
        elif vari == 'evspsbl':
                if simp:
                    dummy['etp'] = cc.vFlux_simp(dummy.evspsbl)
                else:
                    dummy['etp'] = cc.vFlux(dummy.evspsbl, dummy.Tc)
        if vari == 'prhmax':
            if simp:
                dummy['prec_hmax'] = cc.vFlux_simp(dummy.prhmax)
            else:
                dummy['prec_hmax'] = cc.vFlux(dummy.prhmax,dummy.Tc)
        elif vari == 'huss':
            dummy['RH'] = cc.sh2rh(dummy.huss, dummy.Tc, dummy.ps/100.)

    return dummy

# wrapper for climate model data to weather file units and expected columns
@numba.jit
def Te_opt(T_e, gammax, vabar):
    maxdeltaT_e = 1.
    maxit = 9999999
    itc = 0
    while maxdeltaT_e < 0.01:
        v_e = 0.6108 * np.exp(17.27 * T_e / (T_e + 237.3))  # saturated vapour pressure at T_e (S2.5)
        T_enew = gammax * (v_e - vabar)  # rearranged from S8.8
        deltaT_e = T_enew - T_e
        T_e = T_enew
        maxdeltaT_e = np.abs(np.max(deltaT_e))
        if itc > maxit:
            break
        itc += 1
    return T_e


def ET_SzilagyiJozsa(data, Elev, lat, windfunction_ver='1948', alpha=0.23, zerocorr=True):
    # Taken from R package Evapotranspiration >> Danlu Guo <danlu.guo@adelaide.edu.au>
    # Daily Actual Evapotranspiration after Szilagyi, J. 2007, doi:10.1029/2006GL028708
    # data is assumed to be a pandas data frame with at least:
    # T or Tmin/max - daily temperature in degree Celcius,
    # RH or RHmin/max - daily  relative humidity in percentage,
    # u2 - daily wind speed in meters per second
    # Rs - daily solar radiation in Megajoule per square meter.
    # Result in [mm/day] for EToSJ, EToPM, and EToPT reference ET

    # update alphaPT according to Szilagyi and Jozsa (2008)
    alphaPT = 1.31
    lat_rad = lat * np.pi / 180.
    alphaPT = 1.31
    sigma = 4.903e-09
    Gsc = 0.082
    lambdax = 2.45

    # add julian Days
    J = data.index.dayofyear

    # Calculating mean temperature
    if ('Ta' in data.columns):
        Ta = data['Ta']
    elif ('T' in data.columns):
        Ta = data['T']
    else:
        Ta = (data['Tmax'] + data[
            'Tmin']) / 2  # Equation S2.1 in Tom McMahon's HESS 2013 paper, which in turn was based on Equation 9 in Allen et al, 1998.

    # Saturated vapour pressure
    vs_Tmax = 0.6108 * np.exp(17.27 * data['Tmax'] / (data['Tmax'] + 237.3))  # Equation S2.5
    vs_Tmin = 0.6108 * np.exp(17.27 * data['Tmin'] / (data['Tmin'] + 237.3))  # Equation S2.5
    vas = (vs_Tmax + vs_Tmin) / 2.  # Equation S2.6

    # Vapour pressure
    if 'RHmax' in data.columns:
        vabar = (vs_Tmin * data['RHmax'] / 100. + vs_Tmax * data['RHmin'] / 100.) / 2.  # Equation S2.7
        # mean relative humidity
        RHmean = (data['RHmax'] + data['RHmin']) / 2

    else:
        vabar = (vs_Tmin + vs_Tmax) / 2 * data['RH'] / 100.

    if 'Rs' in data.columns:
        R_s = data['Rs']
    else:
        print('Radiation data missing')
        return

    # Calculations from data and constants for Penman
    if 'aP' in data.columns:
        P = data['aP']
    else:
        P = 101.3 * ((293. - 0.0065 * Elev) / 293.) ** 5.26  # atmospheric pressure (S2.10)

    delta = 4098 * (0.6108 * np.exp((17.27 * Ta) / (Ta + 237.3))) / (
            (Ta + 237.3) ** 2)  # slope of vapour pressure curve (S2.4)
    gamma = 0.00163 * P / lambdax  # psychrometric constant (S2.9)
    d_r2 = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)  # dr is the inverse relative distance Earth-Sun (S3.6)
    delta2 = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)  # solar dedication (S3.7)
    w_s = np.arccos(-1. * np.tan(lat_rad) * np.tan(delta2))  # sunset hour angle (S3.8)
    N = 24 / np.pi * w_s  # calculating daily values
    R_a = (1440 / np.pi) * d_r2 * Gsc * (
            w_s * np.sin(lat_rad) * np.sin(delta2) + np.cos(lat_rad) * np.cos(delta2) * np.sin(
        w_s))  # extraterristrial radiation (S3.5)
    R_so = (0.75 + (2 * 10 ** - 5) * Elev) * R_a  # clear sky radiation (S3.4)

    R_nl = sigma * (0.34 - 0.14 * np.sqrt(vabar)) * ((data['Tmax'] + 273.2) ** 4 + (data['Tmin'] + 273.2) ** 4) / 2 * (
            1.35 * R_s / R_so - 0.35)  # estimated net outgoing longwave radiation (S3.3)
    # For vegetated surface
    R_nsg = (1 - alpha) * R_s  # net incoming shortwave radiation (S3.2)
    R_ng = R_nsg - R_nl  # net radiation (S3.1)

    if 'u2' in data.columns:
        u2 = data['u2']
        if windfunction_ver == "1948":
            f_u = 2.626 + 1.381 * u2  # wind function Penman 1948 (S4.11)
        elif windfunction_ver == "1956":
            f_u = 1.313 + 1.381 * u2  # wind function Penman 1956 (S4.3)

        Ea = f_u * (vas - vabar)  # (S4.2)

        Epenman_Daily = delta / (delta + gamma) * (R_ng / lambdax) + gamma / (
                delta + gamma) * Ea  # Penman open-water evaporation (S4.1)

    else:

        Epenman_Daily = 0.047 * R_s * np.sqrt(Ta + 9.5) - 2.4 * (R_s / R_a) ** 2 + 0.09 * (Ta + 20) * (
                1 - RHmean / 100)  # Penman open-water evaporation without wind data by Valiantzas (2006) (S4.12)

    # Iteration for equilibrium temperature T_e
    T_e = Ta
    gammax = Ta - 1 / gamma * (1 - R_ng / (lambdax * Epenman_Daily))

    T_e = Te_opt(T_e.values, gammax.values, vabar.values)
    T_e = pd.Series(T_e, index=Ta.index)

    deltaT_e = 4098 * (0.6108 * np.exp((17.27 * T_e) / (T_e + 237.3))) / (
            (T_e + 237.3) ** 2)  # slope of vapour pressure curve (S2.4)
    E_PT_T_e = alphaPT * (deltaT_e / (deltaT_e + gamma) * R_ng / lambdax)  # Priestley-Taylor evapotranspiration at T_e
    E_SJ_Act_Daily = 2 * E_PT_T_e - Epenman_Daily  # actual evapotranspiration by Szilagyi and Jozsa (2008) (S8.7)

    if zerocorr:
        ET_Daily = np.fmax(E_SJ_Act_Daily, 0.)
    else:
        ET_Daily = E_SJ_Act_Daily

    return ET_Daily, Epenman_Daily, E_PT_T_e, vabar

def qair2rh(qair, temp, press = 1013.25):
    es = 6.112 * np.exp((17.67 * temp)/(temp + 243.5))
    e = qair * press / (0.378 * qair + 0.622)
    rh = e / es
    rh[rh > 1] = 1
    rh[rh < 0] = 0
    return(rh*100.)

def get_coord(coords,dummy):
    '''
    get indexer in xarray from given lat/lon coords
    :param coords: list with lat, lon
    :param dummy: xarray with lat lon coordinates
    :return: [x,y] indexes
    '''
    dev_mx = abs(dummy.lon.values - coords[1]) + abs(dummy.lat.values - coords[0]) #deviation matrix
    xi,yi = np.where(dev_mx == np.min(dev_mx))
    return [xi[0],yi[0]]

def CMIP5_ts(NSc, modi, mask, coords=[53.43, 7.09]):
    '''
    wrapper for climate model data to weather file units and expected columns
    :param NSc: dataframe output from gc.nc_inventory
    :param modi: index for code.unique()
    :param coords: lat/lon of center point
    :return: dataframe with daily data in the following columns [tas, tasmax, tasmin, pr, huss, rsds, ps, sfcWind, et_harg]
    '''

    if len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) > 10:
        k = 0
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]
        NScx = NScx.loc[NScx.RCM == NScx.RCM.unique()[k]]
        while len(NScx) < 8:
            k += 1
            NScx = NScx.loc[NScx.RCM == NScx.RCM.unique()[k]]
            if k == len(NScx.RCM.unique()) - 1:
                break

    elif len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) < 4:
        print('many missing variables!')
        return []
    else:
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]

    v = 'tas'
    i = NScx.loc[NScx.variable == v].index[0]
    dummy = xr.open_dataset(NScx.loc[i, 'file'])
    dlat = coords[0]
    tas = ESGF_m(dummy, mask)
    if tas.mean() > 200.:
        tas = tas - 273.15
    dummyp = pd.DataFrame(tas)
    apnan = True

    for v in ['tasmax', 'tasmin', 'ps','psl', 'pr', 'hurs', 'huss', 'hus', 'rsds', 'sfcWind', 'dtr']:
        try:
            i = NScx.loc[NScx.variable == v].index[0]
            dummy = xr.open_dataset(NScx.loc[i, 'file'])
            if NScx.loc[i, 'variable'] == 'tasmin':
                tasmin = ESGF_m(dummy, mask)
                if tasmin.mean() > 200.:
                    tasmin = tasmin - 273.15
                dummyp = pd.concat([dummyp, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'tasmax':
                tasmax = ESGF_m(dummy, mask)
                if tasmax.mean() > 200.:
                    tasmax = tasmax - 273.15
                dummyp = pd.concat([dummyp, tasmax], axis=1)
            elif (NScx.loc[i, 'variable'] == 'dtr') & ~('tasmax' in NScx.variable.unique()):
                dtr = ESGF_m(dummy, mask)
                tasmax = tas + dtr / 2.
                tasmin = tas - dtr / 2.
                dummyp = pd.concat([dummyp, tasmax, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'pr':
                prx = ESGF_m(dummy, mask)
                if prx.resample('1Y').sum().mean() < 400.:
                    prx *= 86400.
                if 'ps' in NScx.variable.values:
                    precd = pd.concat([tas, ps, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values, precd.ps.values) / 1000000.)
                else:
                    precd = pd.concat([tas, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values) / 1000000.)
                # prec.columns = ['pr']
                dummyp = pd.concat([dummyp, prec], axis=1)
                # prec = (wd(pd.concat([tas,ps],axis=1).tas.values,pd.concat([tas,ps],axis=1).ps.values)*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400)/1000.
                # prec = 999847.0655858772*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400/1000.
            elif (NScx.loc[i, 'variable'] == 'huss') | (NScx.loc[i, 'variable'] == 'hus'):
                # hu = (dummy.sel(x=xi, y=yi).to_dataframe().huss * (611. * np.exp((17.67 * (tas)) / (tas + 273.15 - 29.65))))
                if 'ps' in NScx.variable.values:
                    hu = qair2rh(ESGF_m(dummy, mask), tas, ps / 100.)
                else:
                    hu = qair2rh(ESGF_m(dummy, mask), tas)
                if 'hu' in dummyp.columns:
                    hu.name = 'hus'
                elif 'hus' in dummyp.columns:
                    hu.name = 'hus2'
                else:
                    hu.name = 'hu'
                if np.mean(hu) < 1.:
                    hu = 100. * hu
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'hurs':
                hu = ESGF_m(dummy, mask)
                hu.name = 'hu'
                if np.mean(hu) < 1.:
                    hu = 100. * hu
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'rsds':
                rsds = 0.086400 * ESGF_m(dummy, mask)  # W/m2 >> MJ/m2day
                dummyp = pd.concat([dummyp, rsds], axis=1)
            elif (NScx.loc[i, 'variable'] == 'ps') | (NScx.loc[i, 'variable'] == 'psl'):
                ps = ESGF_m(dummy, mask)
                if np.mean(ps) > 2000.:
                    ps = ps / 100.
                dummyp = pd.concat([dummyp, ps], axis=1)
                apnan = False
            elif NScx.loc[i, 'variable'] == 'sfcWind':
                sfcWind = ESGF_m(dummy, mask)
                dummyp = pd.concat([dummyp, sfcWind], axis=1)
        except:
            print('try failed with ' + v)
            pass

    dummyp['et_harg'] = np.nan
    lat = pyeto.deg2rad(dlat)
    for i in dummyp.index:
        sol_dec = pyeto.sol_dec(i.dayofyear)
        sha = pyeto.sunset_hour_angle(lat, sol_dec)
        ird = pyeto.inv_rel_dist_earth_sun(i.dayofyear)
        et_rad = pyeto.et_rad(lat, sol_dec, sha, ird)
        dummyp.loc[i, 'et_harg'] = pyeto.hargreaves(dummyp.loc[i, 'tasmin'], dummyp.loc[i, 'tasmax'],
                                                    dummyp.loc[i, 'tas'], et_rad)

    if 'sfcWind' in dummyp.columns:
        sfcWinddummy = False
    else:
        dummyp['sfcWind'] = 4.3
        sfcWinddummy = True

    if 'ps' in dummyp.columns:
        psdummy = False
    else:
        dummyp['ps'] = 1013.13
        psdummy = True

    try:
        if apnan:
            dummyx = dummyp[['tas', 'tasmax', 'tasmin', 'pr', 'hu', 'rsds', 'sfcWind']].copy()
            dummyx.columns = ['Ta', 'Tmax', 'Tmin', 'Prec', 'RH', 'Rs', 'u2']
            print('No atmospheric pressue data given. Using standard reference instead.')
        else:
            dummyx = dummyp[['tas', 'tasmax', 'tasmin', 'pr', 'hu', 'rsds', 'ps', 'sfcWind']].copy()
            dummyx.columns = ['Ta', 'Tmax', 'Tmin', 'Prec', 'RH', 'Rs', 'aP', 'u2']
            dummyx.aP = dummyx.aP * 0.1

        EToSJ, EToPM2, EToPT, vabar = ET_SzilagyiJozsa(dummyx, 3., 53.4, zerocorr=True)
        dummyp['vabar'] = vabar
        dummyp['EToSJ'] = EToSJ
        dummyp['EToPM'] = EToPM2
        dummyp['EToPT'] = EToPT
        if apnan:
            dummyp['aP'] = np.nan

    except:
        print('Failed to calculate EToSJ.')
        dummyp['vabar'] = np.nan
        dummyp['EToSJ'] = np.nan
        dummyp['EToPM'] = np.nan
        dummyp['EToPT'] = np.nan
        if apnan:
            dummyp['aP'] = np.nan

    dummyp['EToPM1'] = np.nan
    if apnan & ('rsds' in dummyp.columns):
        EToPM = pyeto.fao56_penman_monteith(dummyp.rsds.values, dummyp['tas'].values + 273.15, dummyp.sfcWind.values,
                                            pyeto.svp_from_t(dummyp['tas'].values), dummyp.vabar,
                                            pyeto.delta_svp(dummyp['tas'].values),
                                            pyeto.psy_const(1013.13 * 0.1))
        EToPM = pd.Series(EToPM)
        EToPM.index = dummyp.index
        dummyp['EToPM1'] = EToPM
    elif ('rsds' in dummyp.columns):
        EToPM = pyeto.fao56_penman_monteith(dummyp.rsds.values, dummyp['tas'].values + 273.15, dummyp.sfcWind.values,
                                            pyeto.svp_from_t(dummyp['tas'].values), dummyp.vabar,
                                            pyeto.delta_svp(dummyp['tas'].values),
                                            pyeto.psy_const(dummyp.ps.values * 0.1))
        EToPM = pd.Series(EToPM)
        EToPM.index = dummyp.index
        dummyp['EToPM1'] = EToPM
    else:
        dummyp['rsds'] = np.nan

    if sfcWinddummy:
        dummyp['sfcWind'] = np.nan
    if psdummy:
        dummyp['ps'] = np.nan

    dummyp['EToHG1'] = np.nan
    EToHG = pyeto.hargreaves(dummyp.tasmin.values, dummyp.tasmax.values, dummyp.tas.values,
                             pyeto.et_rad(52. * np.pi / 180., pyeto.sol_dec(dummyp.index.dayofyear.values),
                                          pyeto.sunset_hour_angle(52. * np.pi / 180.,
                                                                  pyeto.sol_dec(dummyp.index.dayofyear.values)),
                                          pyeto.inv_rel_dist_earth_sun(dummyp.index.dayofyear.values)))
    EToHG = pd.Series(EToHG)
    EToHG.index = dummyp.index
    dummyp['EToHG1'] = EToHG

    return dummyp

def cordex_ts(NSc, modi, coords=[53.43, 7.09]):
    '''
    wrapper for climate model data to weather file units and expected columns
    :param NSc: dataframe output from gc.nc_inventory
    :param modi: index for code.unique()
    :param coords: lat/lon of center point
    :return: dataframe with daily data in the following columns [tas, tasmax, tasmin, pr, huss, rsds, ps, sfcWind, et_harg]
    '''

    if len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) > 8:
        k = 0
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]
        NScx = NScx.loc[NScx.ensemble == NScx.ensemble.unique()[k]]
        while len(NScx) < 8:
            k += 1
            NScx = NScx.loc[NScx.ensemble == NScx.ensemble.unique()[k]]
            if k == len(NScx.ensemble.unique()) - 1:
                break

    elif len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) < 4:
        print('many missing variables!')
    else:
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]

    v = 'tas'
    i = NScx.loc[NScx.variable == v].index[0]
    dummy = xr.open_dataset(NScx.loc[i, 'file'])
    xi, yi = get_coord(coords, dummy)
    dlat = coords[0]
    tas = dummy.sel(x=xi, y=yi).to_dataframe().tas
    if tas.mean() > 200.:
        tas = tas - 273.15
    dummyp = tas.copy()

    for v in ['tasmax', 'tasmin', 'ps', 'pr', 'huss', 'hurs', 'rsds', 'sfcWind', 'dtr']:
        try:
            i = NScx.loc[NScx.variable == v].index[0]
            dummy = xr.open_dataset(NScx.loc[i, 'file'])
            if NScx.loc[i, 'variable'] == 'tasmin':
                tasmin = dummy.sel(x=xi, y=yi).to_dataframe().tasmin
                if tasmin.mean() > 200.:
                    tasmin = tasmin - 273.15
                dummyp = pd.concat([dummyp, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'tasmax':
                tasmax = dummy.sel(x=xi, y=yi).to_dataframe().tasmax
                if tasmax.mean() > 200.:
                    tasmax = tasmax - 273.15
                dummyp = pd.concat([dummyp, tasmax], axis=1)
            elif (NScx.loc[i, 'variable'] == 'dtr') & ~('tasmax' in NScx.variable.unique()):
                dtr = dummy.sel(x=xi, y=yi).to_dataframe()['dtr']
                tasmax = tas + dtr / 2.
                tasmin = tas - dtr / 2.
                dummyp = pd.concat([dummyp, tasmax, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'pr':
                prx = dummy.sel(x=xi, y=yi).to_dataframe().pr
                if prx.resample('1Y').sum().mean() < 400.:
                    prx *= 86400.
                if 'ps' in NScx.variable.values:
                    precd = pd.concat([tas, ps, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values, precd.ps.values) / 1000000.)
                else:
                    precd = pd.concat([tas, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values) / 1000000.)
                # prec.columns = ['pr']
                dummyp = pd.concat([dummyp, prec], axis=1)
                # prec = (wd(pd.concat([tas,ps],axis=1).tas.values,pd.concat([tas,ps],axis=1).ps.values)*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400)/1000.
                # prec = 999847.0655858772*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400/1000.
            elif NScx.loc[i, 'variable'] == 'huss':
                #hu = (dummy.sel(x=xi, y=yi).to_dataframe().huss * (611. * np.exp((17.67 * (tas)) / (tas + 273.15 - 29.65))))
                if 'ps' in NScx.variable.values:
                    hu = qair2rh(dummy.sel(x=xi, y=yi).to_dataframe().huss, tas, ps/100.)
                else:
                    hu = qair2rh(dummy.sel(x=xi, y=yi).to_dataframe().huss, tas)
                hu.name = 'hu'
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'hurs':
                hu = dummy.sel(x=xi, y=yi).to_dataframe().hurs
                hu.name = 'hu'
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'rsds':
                rsds = 0.086400 * dummy.sel(x=xi, y=yi).to_dataframe().rsds  # W/m2 >> MJ/m2day
                dummyp = pd.concat([dummyp, rsds], axis=1)
            elif NScx.loc[i, 'variable'] == 'ps':
                ps = dummy.sel(x=xi, y=yi).to_dataframe().ps
                dummyp = pd.concat([dummyp, ps], axis=1)
            elif NScx.loc[i, 'variable'] == 'sfcWind':
                sfcWind = dummy.sel(x=xi, y=yi).to_dataframe().sfcWind
                dummyp = pd.concat([dummyp, sfcWind], axis=1)
        except:
            print('try failed with ' + v)
            pass

    dummyp['et_harg'] = np.nan
    lat = pyeto.deg2rad(dlat)
    for i in dummyp.index:
        sol_dec = pyeto.sol_dec(i.dayofyear)
        sha = pyeto.sunset_hour_angle(lat, sol_dec)
        ird = pyeto.inv_rel_dist_earth_sun(i.dayofyear)
        et_rad = pyeto.et_rad(lat, sol_dec, sha, ird)
        dummyp.loc[i, 'et_harg'] = pyeto.hargreaves(dummyp.loc[i, 'tasmin'], dummyp.loc[i, 'tasmax'],
                                                    dummyp.loc[i, 'tas'], et_rad)

    return dummyp


def cordex_ts_m(NSc, modi, mask, coords=[53.43, 7.09]):
    '''
    wrapper for climate model data to weather file units and expected columns
    :param NSc: dataframe output from gc.nc_inventory
    :param modi: index for code.unique()
    :param mask: 2D array for spatial mask
    :param coords: lat/lon of center point
    :return: dataframe with daily data in the following columns [tas, tasmax, tasmin, pr, huss, rsds, ps, sfcWind, et_harg]
    '''

    if len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) > 8:
        k = 0
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]
        NScx = NScx.loc[NScx.ensemble == NScx.ensemble.unique()[k]]
        while len(NScx) < 8:
            k += 1
            NScx = NScx.loc[NScx.ensemble == NScx.ensemble.unique()[k]]
            if k == len(NScx.ensemble.unique()) - 1:
                break

    elif len(NSc.loc[NSc.code == NSc.code.unique()[modi]]) < 4:
        print('many missing variables!')
    else:
        NScx = NSc.loc[(NSc.code == NSc.code.unique()[modi])]

    v = 'tas'
    i = NScx.loc[NScx.variable == v].index[0]
    dummy = xr.open_dataset(NScx.loc[i, 'file'])
    dlat = coords[0]
    tas = ESGF_m(dummy, mask)
    if tas.mean() > 200.:
        tas = tas - 273.15
    dummyp = pd.DataFrame(tas)
    apnan = True

    for v in ['tasmax', 'tasmin', 'ps', 'pr', 'hurs', 'huss', 'rsds', 'sfcWind', 'dtr']:
        try:
            i = NScx.loc[NScx.variable == v].index[0]
            dummy = xr.open_dataset(NScx.loc[i, 'file'])
            if NScx.loc[i, 'variable'] == 'tasmin':
                tasmin = ESGF_m(dummy, mask)
                if tasmin.mean() > 200.:
                    tasmin = tasmin - 273.15
                dummyp = pd.concat([dummyp, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'tasmax':
                tasmax = ESGF_m(dummy, mask)
                if tasmax.mean() > 200.:
                    tasmax = tasmax - 273.15
                dummyp = pd.concat([dummyp, tasmax], axis=1)
            elif (NScx.loc[i, 'variable'] == 'dtr') & ~('tasmax' in NScx.variable.unique()):
                dtr = ESGF_m(dummy, mask)
                tasmax = tas + dtr / 2.
                tasmin = tas - dtr / 2.
                dummyp = pd.concat([dummyp, tasmax, tasmin], axis=1)
            elif NScx.loc[i, 'variable'] == 'pr':
                prx = ESGF_m(dummy, mask)
                if prx.resample('1Y').sum().mean() < 400.:
                    prx *= 86400.
                if 'ps' in NScx.variable.values:
                    precd = pd.concat([tas, ps, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values, precd.ps.values) / 1000000.)
                else:
                    precd = pd.concat([tas, prx], axis=1)
                    prec = precd.pr * (wd(precd.tas.values) / 1000000.)
                # prec.columns = ['pr']
                dummyp = pd.concat([dummyp, prec], axis=1)
                # prec = (wd(pd.concat([tas,ps],axis=1).tas.values,pd.concat([tas,ps],axis=1).ps.values)*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400)/1000.
                # prec = 999847.0655858772*(dummy.sel(x=xi,y=yi).to_dataframe().pr)*86.400/1000.
            elif NScx.loc[i, 'variable'] == 'huss':
                # hu = (dummy.sel(x=xi, y=yi).to_dataframe().huss * (611. * np.exp((17.67 * (tas)) / (tas + 273.15 - 29.65))))
                if 'ps' in NScx.variable.values:
                    hu = qair2rh(ESGF_m(dummy, mask), tas, ps / 100.)
                else:
                    hu = qair2rh(ESGF_m(dummy, mask), tas)
                if 'hu' in dummyp.columns:
                    hu.name = 'hus'
                else:
                    hu.name = 'hu'
                if np.mean(hu) < 1.:
                    hu = 100. * hu
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'hurs':
                hu = ESGF_m(dummy, mask)
                hu.name = 'hu'
                if np.mean(hu) < 1.:
                    hu = 100. * hu
                dummyp = pd.concat([dummyp, hu], axis=1)
            elif NScx.loc[i, 'variable'] == 'rsds':
                rsds = 0.086400 * ESGF_m(dummy, mask)  # W/m2 >> MJ/m2day
                dummyp = pd.concat([dummyp, rsds], axis=1)
            elif NScx.loc[i, 'variable'] == 'ps':
                ps = ESGF_m(dummy, mask)
                if np.mean(ps) > 2000.:
                    ps = ps / 100.
                dummyp = pd.concat([dummyp, ps], axis=1)
                apnan = False
            elif NScx.loc[i, 'variable'] == 'sfcWind':
                sfcWind = ESGF_m(dummy, mask)
                dummyp = pd.concat([dummyp, sfcWind], axis=1)
        except:
            print('try failed with ' + v)
            pass

    dummyp['et_harg'] = np.nan
    lat = pyeto.deg2rad(dlat)
    for i in dummyp.index:
        sol_dec = pyeto.sol_dec(i.dayofyear)
        sha = pyeto.sunset_hour_angle(lat, sol_dec)
        ird = pyeto.inv_rel_dist_earth_sun(i.dayofyear)
        et_rad = pyeto.et_rad(lat, sol_dec, sha, ird)
        dummyp.loc[i, 'et_harg'] = pyeto.hargreaves(dummyp.loc[i, 'tasmin'], dummyp.loc[i, 'tasmax'],
                                                    dummyp.loc[i, 'tas'], et_rad)

    try:
        if apnan:
            dummyx = dummyp[['tas', 'tasmax', 'tasmin', 'pr', 'hu', 'rsds', 'sfcWind']].copy()
            dummyx.columns = ['Ta', 'Tmax', 'Tmin', 'Prec', 'RH', 'Rs', 'u2']
            print('No atmospheric pressue data given. Using standard reference instead.')
        else:
            dummyx = dummyp[['tas', 'tasmax', 'tasmin', 'pr', 'hu', 'rsds', 'ps', 'sfcWind']].copy()
            dummyx.columns = ['Ta', 'Tmax', 'Tmin', 'Prec', 'RH', 'Rs', 'aP', 'u2']
            dummyx.aP = dummyx.aP * 0.1

        EToSJ, EToPM2, EToPT, vabar = ET_SzilagyiJozsa(dummyx, 3., 53.4, zerocorr=True)
        dummyp['vabar'] = vabar
        dummyp['EToSJ'] = EToSJ
        dummyp['EToPM'] = EToPM2
        dummyp['EToPT'] = EToPT
        if apnan:
            dummyp['aP'] = np.nan

    except:
        print('Failed to calculate EToSJ.')
        dummyp['vabar'] = np.nan
        dummyp['EToSJ'] = np.nan
        dummyp['EToPM'] = np.nan
        dummyp['EToPT'] = np.nan
        if apnan:
            dummyp['aP'] = np.nan

    dummyp['EToPM1'] = np.nan
    if 'sfcWind' in dummyp.columns:
        sfcWinddummy = False
    else:
        dummyp['sfcWind'] = 4.3
        sfcWinddummy = True

    if apnan & ('rsds' in dummyp.columns):
        EToPM = pyeto.fao56_penman_monteith(dummyp.rsds.values, dummyp['tas'].values + 273.15, dummyp.sfcWind.values,
                                            pyeto.svp_from_t(dummyp['tas'].values), dummyp.vabar,
                                            pyeto.delta_svp(dummyp['tas'].values),
                                            pyeto.psy_const(1013.13 * 0.1))
        EToPM = pd.Series(EToPM)
        EToPM.index = dummyp.index
        dummyp['EToPM1'] = EToPM
    elif ('rsds' in dummyp.columns):
        EToPM = pyeto.fao56_penman_monteith(dummyp.rsds.values, dummyp['tas'].values + 273.15, dummyp.sfcWind.values,
                                     pyeto.svp_from_t(dummyp['tas'].values), dummyp.vabar,
                                     pyeto.delta_svp(dummyp['tas'].values), pyeto.psy_const(dummyp.ps.values * 0.1))
        EToPM = pd.Series(EToPM)
        EToPM.index = dummyp.index
        dummyp['EToPM1'] = EToPM
    else:
        dummyp['rsds'] = np.nan

    if sfcWinddummy:
        dummyp['sfcWind'] = np.nan

    dummyp['EToHG1'] = np.nan
    EToHG = pyeto.hargreaves(dummyp.tasmin.values, dummyp.tasmax.values, dummyp.tas.values,
                          pyeto.et_rad(52. * np.pi / 180., pyeto.sol_dec(dummyp.index.dayofyear.values),
                                    pyeto.sunset_hour_angle(52. * np.pi / 180., pyeto.sol_dec(dummyp.index.dayofyear.values)),
                                    pyeto.inv_rel_dist_earth_sun(dummyp.index.dayofyear.values)))
    EToHG = pd.Series(EToHG)
    EToHG.index = dummyp.index
    dummyp['EToHG1'] = EToHG

    return dummyp

def ESGF_m(dsx, mask=[], var=None, agg='mean'):
    '''spatial aggregates
    dsx :: xarray to get spatial aggregate from
    mask :: bool 2D array masking the area of the xarray
    returns pandas Series of aggregated property
    '''
    if var == None:
        var = dsx.attrs['var_name']

    warni=True
    mask2 = np.ones((dsx.dims['x'], dsx.dims['y'])).astype(np.bool)
    if np.shape(mask) == np.shape(mask2):
        dummy = dsx[var].data[:, mask]
    elif mask == []:
        dummy = dsx[var].data[:, mask2]
    else:
        for i in np.arange(dsx.dims['x']):
            for j in np.arange(dsx.dims['y']):
                try:
                    mask2[i,j] = mask[i,j]
                except:
                    if warni:
                        print('mask shape bluntly adjusted for '+var+'.')
                        warni=False
                    else:
                        pass
        dummy = dsx[var].data[:, mask2]

    dummy[dummy > 1e15] = np.nan  # remove nan values

    if agg == 'mean':
        dummy = np.nanmean(dummy, axis=1)
    elif agg == 'median':
        dummy = np.nanmedian(dummy, axis=1)
    elif agg == 'var':
        dummy = np.nanvar(dummy, axis=1)
    elif type(agg) == float:
        dummy = np.nanpercentile(dummy, agg, axis=1)

    dsx1 = pd.Series(dummy, index=pd.to_datetime(dsx.time.data))
    dsx1.name = var
    return dsx1


def cli_weather_wrp(dummyp, dstart=None, dend=None):
    '''
    Wrapper for climate model output column names
    :param dummyp: pandas data frame from climate model output
    :param dstart: optional start date
    :param dend: optional end date
    :return: data frame with standard weather column names
    '''
    cols = dummyp.columns.values
    cols[cols == 'tas'] = 'T'
    cols[cols == 'tasmax'] = 'Tmax'
    cols[cols == 'tasmin'] = 'Tmin'
    cols[cols == 'ps'] = 'aP'
    cols[cols == 'pr'] = 'Prec'
    cols[cols == 'hu'] = 'RH'
    cols[cols == 'rsds'] = 'Rs'
    cols[cols == 'sfcWind'] = 'u2'
    cols[cols == 'et_harg'] = 'EToHG'

    if dstart == None:
        dstart = dummyp.index[0]
    if dend == None:
        dend = dummyp.index[-1]

    dummyp1 = dummyp.loc[dstart:dend].copy()
    dummyp1.columns = cols

    return dummyp1


# wrapper for self-calibrating PDSI calculation
def scPDSI(cts,awc=123,rs='1M',ETo='PM'):
    #this uses a conversion to inches! *0.0393701
    prcp = cts.Prec.resample(rs).sum().values * 0.0393701
    if ETo=='PM':
        pet = cts.EToPM.resample(rs).sum().values * 0.0393701
    else:
        pet = cts.EToHG.resample(rs).sum().values * 0.0393701
    awc = awc * 0.0393701
    start_year = cts.index[10].year
    end_year = cts.index[-10].year
    scpdsi, pdsi, phdi, pmdi, zindex = indices.scpdsi(prcp, pet, awc, start_year, start_year, end_year)
    dummy = pd.DataFrame([scpdsi, pdsi, phdi, pmdi, zindex]).T
    dummy.index = cts.Prec.resample(rs).sum().index
    dummy.columns = ['scpdsi', 'pdsi', 'phdi', 'pmdi', 'zindex']
    return dummy


def scPDSI1M(cts, awc=123, ETo='PM'):
    '''self-calibrating Palmer drought severity index
    :param cts: climate time series (pandas data frame with time index)
    :param awc: available soilwater content (float in mm)
    :param ETo: PM for Penman Monteith else uses Hargreaves

    Assumes about monthly input data.
    Returns 5 columns of PDSI estimators
    '''
    # this uses a conversion to inches! *0.0393701
    prcp = cts.Prec.values * 0.0393701
    if ETo == 'PM':
        pet = cts.EToPM.values * 0.0393701
    else:
        pet = cts.EToHG.values * 0.0393701
    awc = awc * 0.0393701
    start_year = cts.index[1].year
    end_year = cts.index[-1].year
    scpdsi, pdsi, phdi, pmdi, zindex = indices.scpdsi(prcp, pet, awc, start_year, start_year, end_year)
    dummy = pd.DataFrame([scpdsi, pdsi, phdi, pmdi, zindex]).T
    dummy.index = cts.index
    dummy.columns = ['scpdsi', 'pdsi', 'phdi', 'pmdi', 'zindex']
    return dummy

# climate data aggregation specification
def climate_tagg(x):
    '''
    Aggregation function wrapper.
    Use cli_weather_wrp(data).resample('1M').apply(climate_tagg)
    :param x: data frame with climate data
    :return: resampled version
    '''
    names = {
            'T': x['T'].mean(),
            'Tmin': x['Tmin'].min(),
            'Tmax': x['Tmax'].max(),
            'Prec': x['Prec'].sum(),
            'Rs': x['Rs'].sum(),
            'RH': x['RH'].mean(),
            'u2': x['u2'].mean(),
            'vabar': x['vabar'].mean(),
            'aP': x['aP'].mean(),
            'EToPM': x['EToPM'].sum(),
            'EToPM1': x['EToPM1'].sum(),
            'EToHG': x['EToHG'].sum(),
            'EToSJ': x['EToSJ'].sum(),
            'EToPT': x['EToPT'].sum()}

    return pd.Series(names, index=['T', 'Tmin', 'Tmax', 'Prec', 'Rs', 'RH', 'u2', 'vabar', 'aP', 'EToPM', 'EToPM1', 'EToHG', 'EToSJ', 'EToPT'])

def cli_wrp1M(NSc, modi, mask, tres):
    dummy = cordex_ts_m(NSc,modi,mask)
    dummyr = cli_weather_wrp(dummy).resample(tres).apply(climate_tagg)
    dummyr['scPDSIhg'] = scPDSI1M(dummyr,ETo='HG').scpdsi
    try:
        dummyr['scPDSIpm'] = scPDSI1M(dummyr,ETo='PM').scpdsi
    except:
        dummyr['scPDSIpm'] = np.nan
    return dummyr

def cli_wrp(NSc, modi, mask, tres='1D'):
    dummy = cordex_ts_m(NSc,modi,mask)
    dummyr = cli_weather_wrp(dummy).resample(tres).apply(climate_tagg)
    return dummyr

def cli_wrpC5(NSc, modi, mask, tres='1D'):
    dummy = CMIP5_ts(NSc,modi,mask)
    dummyr = cli_weather_wrp(dummy).resample(tres).apply(climate_tagg)
    return dummyr


def get_monthly_climate(NSc, mask, tres='1M', proj='CORDEX'):
    firstitem = True
    for i in np.arange(len(NSc.code.unique())):
        try:
            dummyx = cli_wrp1M(NSc, i, mask, tres)
            dummyc = NSc.loc[NSc.code == NSc.code.unique()[i]].iloc[0]
            if firstitem:
                dummyxr = xr.Dataset({dummyc.code: xr.DataArray(dummyx, dims=['time', 'vars'],
                                                                attrs={'model': dummyc.model,
                                                                       'RCM': dummyc.RCM,
                                                                       'RCP': dummyc.experiment,
                                                                       'ensemble': dummyc.ensemble,
                                                                       'Project': proj})})
                firstitem = False
            else:
                dummyxr[dummyc.code] = xr.DataArray(dummyx, dims=['time', 'vars'],
                                                    attrs={'model': dummyc.model,
                                                           'RCM': dummyc.RCM,
                                                           'RCP': dummyc.experiment,
                                                           'ensemble': dummyc.ensemble,
                                                           'Project': proj})
            print('++ added ' + NSc.code.unique()[i])
        except:
            print('!! could not load ' + NSc.code.unique()[i])

    return dummyxr

def append_monthly_climate(NSc, mask, dummyxr, tres='1M', proj='CORDEX'):
    for i in np.arange(len(NSc.code.unique())):
        try:
            dummyx = cli_wrp1M(NSc, i, mask, tres)
            dummyc = NSc.loc[NSc.code == NSc.code.unique()[i]].iloc[0]
            dummyxr[dummyc.code] = xr.DataArray(dummyx, dims=['time', 'vars'],
                                                attrs={'model': dummyc.model,
                                                       'RCM': dummyc.RCM,
                                                       'RCP': dummyc.experiment,
                                                       'ensemble': dummyc.ensemble,
                                                       'Project': proj})
            print('++ added ' + NSc.code.unique()[i])
        except:
            print('!! could not load ' + NSc.code.unique()[i])

    return dummyxr

def get_all_climate(NSc, mask, tres='1D', proj='CORDEX',appendxr=False):
    if type(appendxr)==bool:
        firstitem = True
    else:
        dummyxr = appendxr
    for i in np.arange(len(NSc.code.unique())):
        try:
            if (proj=='CORDEX') | (proj=='REKLIES'):
                dummyx = cli_wrp(NSc, i, mask, tres)
            elif proj=='CMIP5':
                dummyx = cli_wrpC5(NSc, i, mask, tres)
            dummyc = NSc.loc[NSc.code == NSc.code.unique()[i]].iloc[0]
            if firstitem:
                dummyxr = xr.Dataset({dummyc.code: xr.DataArray(dummyx, dims=['time', 'vars'],
                                                                attrs={'model': dummyc.model,
                                                                       'RCM': dummyc.RCM,
                                                                       'RCP': dummyc.experiment,
                                                                       'ensemble': dummyc.ensemble,
                                                                       'Project': proj})})
                firstitem = False
            else:
                dummyxr[dummyc.code] = xr.DataArray(dummyx, dims=['time', 'vars'],
                                                    attrs={'model': dummyc.model,
                                                           'RCM': dummyc.RCM,
                                                           'RCP': dummyc.experiment,
                                                           'ensemble': dummyc.ensemble,
                                                           'Project': proj})
            print('++ added ' + NSc.code.unique()[i])
        except:
            print('!! could not load ' + NSc.code.unique()[i])

    return dummyxr




