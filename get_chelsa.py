# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de

import pandas as pd
import numpy as np
import xarray as xr
from bs4 import BeautifulSoup
import requests
import progressbar
import os


def listFD(url, ext=''):
    '''List files in web directory of given url and extension'''
    page = requests.get(url).text
    #print(page)
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


def store_tempfi(url, fix='temp.tif'):
    '''Load a geotiff from given url.
    Deletes existing file before downloading.
    '''
    try:
        os.remove(fix)
    except:
        pass

    r = requests.get(url, allow_redirects=True)
    open(fix, 'wb').write(r.content)

def store_fi(url):
    '''Load a file from given url.
    Deletes existing file before downloading.
    '''
    fix = url.split('/')[-1]
    try:
        os.remove(fix)
    except:
        pass

    r = requests.get(url, allow_redirects=True)
    open(fix, 'wb').write(r.content)

    return fix


def get_data_box(raster_fi, flatdata=True, lon_min=6.2, lat_min=52.0, lon_max=11.0, lat_max=55.0, out_tif='clip_temp.tif'):
    '''Open GeoTiff and subsample at given coordinate box. Returns 2D array.'''
    import pycrs
    import rasterio as rio
    import cartopy.crs as ccrs
    from rasterio.warp import transform
    from rasterio.mask import mask
    from rasterio.plot import show
    from shapely.geometry import box
    from fiona.crs import from_epsg
    import json
    import geopandas as gpd

    data = rio.open(raster_fi)
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))

    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

    coords = getFeatures(geo)
    out_img, out_transform = rio.mask.mask(data, coords, crop=True)
    datax = out_img.astype(np.float)
    datax[datax <= -20000.] = np.nan

    if flatdata:
        return datax[0, :, :]
    else:
        out_meta = data.meta.copy()
        epsg_code = int(data.crs.data['init'][5:])
        out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
                         "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                        )

        with rio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)

        da = xr.open_rasterio(out_tif)

        # Compute the lon/lat coordinates with rasterio.warp.transform
        ny, nx = len(da['y']), len(da['x'])
        x, y = np.meshgrid(da['x'], da['y'])

        # Rasterio works with 1D arrays
        lon, lat = transform(da.crs, {'init': 'EPSG:4326'},
                             x.flatten(), y.flatten())
        lon = np.asarray(lon).reshape((ny, nx))
        lat = np.asarray(lat).reshape((ny, nx))
        da.coords['lon'] = (('y', 'x'), lon)
        da.coords['lat'] = (('y', 'x'), lat)
        return da


def get_chelsa1(fi, vari='bc', expi='rcp45', store_fi=False, savei=10, lon_min=6.2, lat_min=52.0, lon_max=11.0, lat_max=55.0, fix='temp.tif', cfix='ctemp.tif'):
    '''Download Chelsa geotif data in xarray.
    Expects a pd.Dataframe fi as input (see example below).
    Returns xarray with data in subset boy'''
    if (type(store_fi) == bool) & (store_fi == True):
        store_fi = 'chelsa.nc'
    # get time index
    ti_stridx = fi.loc[(fi.variable == vari) & (fi.experiment == expi), 'month'].values
    ti_idx = pd.to_datetime(ti_stridx, format='%Y_%m')

    # get first data
    store_tempfi(fi.loc[(fi.variable == vari) & (fi.experiment == expi) & (fi.month == ti_stridx[0]), 'address'].values[0],fix)
    dataxr = get_data_box(fix, False, lon_min, lat_min, lon_max, lat_max, cfix)

    # reduce to flat data again
    data = dataxr.data[0, :, :].astype(np.float)
    data[data < -20000.] = np.nan

    # construct xarray for all data
    datax = np.zeros((len(ti_idx), np.shape(data)[0], np.shape(data)[1])) * np.nan
    datax[0, :, :] = data

    # get remaining data, loop through file list
    safesave = 0
    safesavei = 0
    with progressbar.ProgressBar(max_value=len(ti_stridx)) as bar:
        for i in np.arange(len(ti_stridx))[1:]:
            try:
                store_tempfi(fi.loc[(fi.variable == vari) & (fi.experiment == expi) & (
                            fi.month == ti_stridx[i]), 'address'].values[0],fix)
                datax[i, :, :] = get_data_box(fix,True,lon_min, lat_min, lon_max, lat_max)
            except:
                return dsx

            safesave += 1
            if ((type(store_fi) == str) & (safesave == savei)):
                # save every savei steps
                if safesavei == 0:
                    # build xarray
                    dsx = xr.Dataset({vari: (['time', 'x', 'y'], datax[safesavei:safesavei + safesave])},
                                     coords={'lon': (['x', 'y'], dataxr.lon),
                                             'lat': (['x', 'y'], dataxr.lat),
                                             'time': ti_idx[safesavei:safesavei + safesave]})
                    safesavei = safesavei + safesave
                    safesave = 0
                else:
                    dsxi = xr.Dataset({vari: (['time', 'x', 'y'], datax[safesavei:safesavei + safesave])},
                                      coords={'lon': (['x', 'y'], dataxr.lon),
                                              'lat': (['x', 'y'], dataxr.lat),
                                              'time': ti_idx[safesavei:safesavei + safesave]})
                    dsx = xr.concat([dsx, dsxi], dim='time')
                    safesavei = safesavei + safesave
                    safesave = 0

                dsx.to_netcdf(store_fi)
            bar.update(i)

    #final xarray construction
    if ((type(store_fi) == str) & (i == len(ti_stridx) - 1)):
        dsxi = xr.Dataset({vari: (['time', 'x', 'y'], datax[safesavei:safesavei + safesave])},
                          coords={'lon': (['x', 'y'], dataxr.lon),
                                  'lat': (['x', 'y'], dataxr.lat),
                                  'time': ti_idx[safesavei:safesavei + safesave]})
        dsx = xr.concat([dsx, dsxi], dim='time')
        dsx.to_netcdf(store_fi)
    else:
        dsx = xr.Dataset({vari: (['time', 'x', 'y'], datax)},
                         coords={'lon': (['x', 'y'], dataxr.lon),
                                 'lat': (['x', 'y'], dataxr.lat),
                                 'time': ti_idx})
    return dsx

def plot_frame(dsx,tix,vari='bc'):
    '''Plot a map from xarray with lat lon'''
    import cartopy.crs as ccrs
    import matplotlib as plt
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines('10m')
    if vari == 'bc':
        plt.contourf(dsx.lon, dsx.lat, dsx.bc[tix], 60, transform=ccrs.PlateCarree())
    elif vari == 'tmax':
        plt.contourf(dsx.lon, dsx.lat, dsx.tmax[tix], 60, transform=ccrs.PlateCarree())
    elif vari == 'tmin':
        plt.contourf(dsx.lon, dsx.lat, dsx.tmin[tix], 60, transform=ccrs.PlateCarree())
    plt.colorbar()
    plt.title(str(pd.to_datetime(dsx.time[tix].data).date()))

def get_chelsa_models():
    url = 'https://www.wsl.ch/lud/chelsa/data/cmip5_ts/nsf-doe-ncar_cesm1-bgc_ar5/'
    ext = 'tif'

    model_dirs = listFD('https://www.wsl.ch/lud/chelsa/data/cmip5_ts/', '/')[1:]
    models = [x.split('//')[2][:-1] for x in model_dirs]
    chelsa_ar5 = pd.DataFrame([models, model_dirs]).T
    chelsa_ar5.columns = ['models', 'mdir']

    return chelsa_ar5

def get_all_chelsa_data(chelsa_ar5):
    for k in np.arange(len(chelsa_ar5)):
        fia = listFD(chelsa_ar5.iloc[k, 1], 'tif')
        fiv = [x.split('_')[-1].split('.')[0] for x in fia]
        fit = [x.split('_rcp')[1][3:10] for x in fia]
        fim = [x.split('_')[5] for x in fia]
        fi = pd.DataFrame([fia, fiv, fit, fim]).T
        fi.columns = ['address', 'variable', 'month', 'experiment']

        prec = get_chelsa1(fi, vari='bc', expi='rcp45', store_fi='prec_rcp45_'+chelsa_ar5.iloc[k, 0]+'.nc')
        tmax = get_chelsa1(fi, vari='tmax', expi='rcp45', store_fi='tmax_rcp45_' + chelsa_ar5.iloc[k, 0] + '.nc')
        tmin = get_chelsa1(fi, vari='tmin', expi='rcp45', store_fi='tmin_rcp45_' + chelsa_ar5.iloc[k, 0] + '.nc')

        prec1 = get_chelsa1(fi, vari='bc', expi='rcp85', store_fi='prec_rcp85_' + chelsa_ar5.iloc[k, 0] + '.nc')
        tmax1 = get_chelsa1(fi, vari='tmax', expi='rcp85', store_fi='tmax_rcp85_' + chelsa_ar5.iloc[k, 0] + '.nc')
        tmin1 = get_chelsa1(fi, vari='tmin', expi='rcp85', store_fi='tmin_rcp85_' + chelsa_ar5.iloc[k, 0] + '.nc')

        print('downloaded model '+chelsa_ar5.iloc[k, 0])
    return

def get_single_chelsa_data(chelsa_ar5,k=0,vari='bc', expi='rcp45'):
    fia = listFD(chelsa_ar5.iloc[k, 1], 'tif')
    fiv = [x.split('_')[-1].split('.')[0] for x in fia]
    fit = [x.split('_rcp')[1][3:10] for x in fia]
    fim = [x.split('_')[5] for x in fia]
    fi = pd.DataFrame([fia, fiv, fit, fim]).T
    fi.columns = ['address', 'variable', 'month', 'experiment']

    return get_chelsa1(fi, vari=vari, expi=expi, store_fi='single_' + chelsa_ar5.iloc[k, 0] + '.nc')
