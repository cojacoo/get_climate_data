# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de

import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import timedelta

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

def ESGFquery(project='CORDEX', experiment='rcp85', time_frequency='day', domain='EUR-11', variable='pr', search_conni=0):
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
    vy = ['tas', 'tasmax', 'tasmin', 'pr', 'prhmax', 'ps', 'huss', 'hurs', 'sfcWind', 'sfcWindmax', 'rsds', 'evspsbl', 'mrso', 'prc', 'rhs', 'psl', 'dtr']

    for a in (set(vy) & set(ky)):
        attr_n = a
        break

    if ('mpi-ge' in url) | ('cmip5' in url):
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
        data = dataset[attr_n][:, subidx[0]:subidx[1] + 1, subidx[2]:subidx[3] + 1].data[:]
        dummy = data[0]
        latx = data[2]
        lonx = data[3]

        if np.shape(latx) != np.shape(dummy)[1:]:
            latx = np.repeat(latx, np.shape(dummy)[1:][1]).reshape(np.shape(dummy)[1:])
        if np.shape(lonx) != np.shape(dummy)[1:]:
            lonx = np.tile(lonx, np.shape(dummy)[1:][0]).reshape(np.shape(dummy)[1:])

        #timestamp
        if ('cmip5' in url):
            start_date = pd.to_datetime(url.split('_')[-1].split('.')[0].split('-')[0], format='%Y%m%d')
            ti = start_date + (dataset.time.data[:]-np.floor(dataset.time.data[0][0]))*timedelta(days=1)

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






