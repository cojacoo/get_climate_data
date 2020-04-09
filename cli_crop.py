# Climate Crop Projection
# Wrapper to:
# 1. load climate model projections, observed weather, observed crop production
# 2. convert climate model data to applicable weather units
# 3. evaluate historical climate model data against observed weather to derive a weighing array
# 4. calibrate a crop model to the observed crop production based on historical climate model data
# 5. apply the calibrated crop model parameters to project future crop production

import numpy as np
import pandas as pd
import xarray as xr

import pyeto
from climate_indices import indices
from get_climate_data.waterdensity import waterdensity as wd

import SIMPLEcrop as SC
import spotpy

# wrapper for climate model data to weather file units and expected columns

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


def cli_weather_wrp(dummyp, dstart=None, dend=None):
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
def scPDSI(cts,awc=123,rs='1M'):
    #this uses a conversion to inches! *0.0393701
    prcp = cts.Prec.resample(rs).sum().values * 0.0393701
    pet = cts.EToHG.resample(rs).sum().values * 0.0393701
    awc = awc * 0.0393701
    start_year = cts.index[10].year
    end_year = cts.index[-10].year
    scpdsi, pdsi, phdi, pmdi, zindex = indices.scpdsi(prcp, pet, awc, start_year, start_year, end_year)
    dummy = pd.DataFrame([scpdsi, pdsi, phdi, pmdi, zindex]).T
    dummy.index = cts.Prec.resample(rs).sum().index
    dummy.columns = ['scpdsi', 'pdsi', 'phdi', 'pmdi', 'zindex']
    return dummy


# build weather weights
def weather_comp_weight(weather,climatemodel,sow_date='-10-01',stop_date='-09-01'):
    # build weight array based on agreement of PDSI in growing period
    # weather coherence assisted model evaluation
    obs_PDSI = scPDSI(weather).scpdsi
    clim_PDSI = scPDSI(climatemodel).scpdsi

    wPDSIo = (abs((obs_PDSI.resample('2M').mean() + 6.) - (clim_PDSI.resample('2M').mean() + 6.)) / 6.)
    wPDSI = np.zeros(len(weather.index.year.unique())-1)
    i = 0
    for yr in weather.index.year.unique()[1:]:
        sowing = pd.to_datetime(str(yr - 1) + sow_date)
        stopday = pd.to_datetime(str(yr) + stop_date)
        wPDSI[i] = wPDSIo.loc[sowing:stopday].mean()
        i += 1

    weights = pd.Series(1. - (0.5 * wPDSI), index = pd.to_datetime(weather.index.year.unique()[1:].astype(str) + stop_date))
    return weights


def get_yield(fi='../agri/agri_stats.xlsx', crop='Winterwheat_Y', distr_mean=True):
    agriyield1 = pd.read_excel(fi, skiprows=3, index_col='Year')
    if distr_mean:
        yield_kg = agriyield1.groupby(agriyield1.index)[crop].mean()*100.
    else:
        yield_kg = agriyield1.loc[agriyield1.District == 'Aurich', crop] * 100.
    return yield_kg


# define evaluation function for yields with applied weights
def eval_fu(evaluation, simulation, weights=1.):
    # update values with resepctive weights
    sim = simulation.copy() * weights
    eva = evaluation.copy() * weights

    # volume error
    ve = np.sum(sim - eva) / np.sum(eva)
    # agreement index
    ai = 1 - (np.sum((eva - sim) ** 2)) / (np.sum((np.abs(sim - np.mean(eva)) + np.abs(eva - np.mean(eva))) ** 2))

    return ai - ve


def start_spot(eval,cli,scheme='SCE',testflight=False,weights=1.,crop='wheat',rep=10000,dbname='calibdb',dynCO2=True):
    # set standard parameters
    # Crop Model Parameters
    if crop == 'wheat':
        para = {
            'AWC': 0.14,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 800.,
            'WUC': 0.096,
            's_water': 0.4,
            'maxT': 34.,
            'extremeT': 50.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.24,
            'HIp': 0.36,
            'Tsum': 1600.,
            'I50maxW': 25.,
            'I50maxH': 100.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    elif crop == 'meadow':
        para = {
            'AWC': 0.1,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 100.,
            'WUC': 0.096,
            's_water': 0.6,
            'maxT': 34.,
            'extremeT': 40.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.34,
            'HIp': 0.5,
            'Tsum': 1500.,
            'I50maxW': 25.,
            'I50maxH': 120.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    elif crop == 'maize':
        para = {
            'AWC': 0.14,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 800.,
            'WUC': 0.096,
            's_water': 0.4,
            'maxT': 34.,
            'extremeT': 50.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.24,
            'HIp': 0.5,
            'Tsum': 1600.,
            'I50maxW': 25.,
            'I50maxH': 100.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    else:
        print('Crop not yet foreseen. Sorry.')

    # load expected CO2 eq. concentration
    if dynCO2 == True:
        CO2 = pd.read_csv('RCP45_MIDYEAR_CONCENTRATIONS.DAT', skiprows=38, delim_whitespace=True, index_col=0).CO2EQ
    else:
        CO2 = eval.copy()*0.+400.

    class spotpy_setup(object):
        if crop == 'wheat':
            def __init__(self):
                self.params = [spotpy.parameter.Uniform('TBASE', low=0, high=2, optguess=0),
                               spotpy.parameter.Uniform('TOPT', low=12, high=17, optguess=15),
                               spotpy.parameter.Uniform('RUE', low=1.1, high=1.4, optguess=1.24),
                               spotpy.parameter.Uniform('I50maxH', low=85, high=115, optguess=100),
                               spotpy.parameter.Uniform('I50maxW', low=20, high=28, optguess=25),
                               spotpy.parameter.Uniform('MaxT', low=32, high=26, optguess=34),
                               spotpy.parameter.Uniform('CO2RUE', low=0.34, high=0.5, optguess=0.4),
                               spotpy.parameter.Uniform('Swater', low=0.3, high=0.6, optguess=0.4),
                               spotpy.parameter.Uniform('TSUM', low=1800, high=2300, optguess=2100),
                               spotpy.parameter.Uniform('AWC', low=0.1, high=0.3, optguess=0.14),
                               spotpy.parameter.Uniform('DDC', low=0.4, high=0.6, optguess=0.5),
                               spotpy.parameter.Uniform('RCN', low=50., high=75., optguess=66.),
                               spotpy.parameter.Uniform('RZD', low=300., high=800., optguess=800.)
                               ]
                self.evals = eval.values

        elif crop == 'meadow':
            def __init__(self):
                self.params = [spotpy.parameter.Uniform('TBASE', low=0., high=8., optguess=5.),
                               spotpy.parameter.Uniform('TOPT', low=10., high=20., optguess=15.),
                               spotpy.parameter.Uniform('RUE', low=1., high=1.8, optguess=1.24),
                               spotpy.parameter.Uniform('I50maxH', low=85, high=115, optguess=100),
                               spotpy.parameter.Uniform('I50maxW', low=10, high=30, optguess=25),
                               spotpy.parameter.Uniform('MaxT', low=32, high=39, optguess=34),
                               spotpy.parameter.Uniform('Swater', low=0.1, high=1.2, optguess=0.2),
                               spotpy.parameter.Uniform('TSUM', low=1350, high=1700, optguess=1500),
                               ]
                self.evals = eval.values

        elif crop == 'maize':
            def __init__(self):
                self.params = [spotpy.parameter.Uniform('TBASE', low=7, high=10, optguess=8),
                               spotpy.parameter.Uniform('TOPT', low=24, high=32, optguess=28),
                               spotpy.parameter.Uniform('RUE', low=1.95, high=2.4, optguess=2.1),
                               spotpy.parameter.Uniform('I50maxH', low=85, high=115, optguess=100),
                               spotpy.parameter.Uniform('I50maxW', low=10, high=14, optguess=12),
                               spotpy.parameter.Uniform('MaxT', low=32, high=39, optguess=34),
                               spotpy.parameter.Uniform('CO2RUE', low=0.001, high=0.1, optguess=0.01),
                               spotpy.parameter.Uniform('Swater', low=1.2, high=1.8, optguess=1.5),
                               spotpy.parameter.Uniform('TSUM', low=2000, high=2300, optguess=2050),
                               spotpy.parameter.Uniform('AWC', low=0.1, high=0.3, optguess=0.14),
                               spotpy.parameter.Uniform('DDC', low=0.4, high=0.6, optguess=0.5),
                               spotpy.parameter.Uniform('RCN', low=50., high=75., optguess=66.),
                               spotpy.parameter.Uniform('RZD', low=300., high=800., optguess=800.)
                               ]
                self.evals = eval.values

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        if crop == 'wheat':
            def simulation(self, x):
                para['Tbase'] = x[0]
                para['Topt'] = x[1]
                para['RUE'] = x[2]
                para['I50maxH'] = x[3]
                para['I50maxW'] = x[4]
                para['maxT'] = x[5]
                para['CO2_RUE'] = x[6]
                para['s_water'] = x[7]
                para['Tsum'] = x[8]
                para['AWC'] = x[9]
                para['DDC'] = x[10]
                para['RCN'] = x[11]
                para['RZD'] = x[12]

                yields = np.zeros(len(eval))
                i = 0
                for yr in eval.index:
                    sowing = pd.to_datetime(str(yr - 1) + '-10-20')
                    stopday = pd.to_datetime(str(yr) + '-09-01')
                    res = SC.SIMPLE(cli, sowing, stopday, para, CO2.loc[yr])
                    yields[i] = res.Yield.max()
                    i += 1

                return yields
        elif crop == 'meadow':
            def simulation(self, x):
                para['Tbase'] = x[0]
                para['Topt'] = x[1]
                para['RUE'] = x[2]
                para['I50maxH'] = x[3]
                para['I50maxW'] = x[4]
                para['maxT'] = x[5]
                para['s_water'] = x[6]
                para['Tsum'] = x[7]
                yields = np.zeros(len(eval))
                i = 0
                for yr in eval.index:
                    sowing = pd.to_datetime(str(yr) + '-03-10')
                    stopday = pd.to_datetime(str(yr) + '-10-15')
                    res = SC.SIMPLE(cli, sowing, stopday, para, CO2.loc[yr])
                    yields[i] = res.Yield.max()
                    i += 1
                return yields
        elif crop == 'maize':
            def simulation(self, x):
                para['Tbase'] = x[0]
                para['Topt'] = x[1]
                para['RUE'] = x[2]
                para['I50maxH'] = x[3]
                para['I50maxW'] = x[4]
                para['maxT'] = x[5]
                para['CO2_RUE'] = x[6]
                para['s_water'] = x[7]
                para['Tsum'] = x[8]
                para['AWC'] = x[9]
                para['DDC'] = x[10]
                para['RCN'] = x[11]
                para['RZD'] = x[12]
                yields = np.zeros(len(eval))
                i = 0
                for yr in eval.index:
                    sowing = pd.to_datetime(str(yr - 1) + '-03-15')
                    stopday = pd.to_datetime(str(yr) + '-10-15')
                    res = SC.SIMPLE(cli, sowing, stopday, para, CO2.loc[yr])
                    yields[i] = res.Yield.max()
                    i += 1
                return yields

        def evaluation(self):
            return self.evals

        if scheme == 'SCE':
            def objectivefunction(self, simulation, evaluation, params=None):
                like = 1. - eval_fu(evaluation, simulation, weights)
                return like
        elif scheme == 'DREAM':
            def objectivefunction(self, simulation, evaluation, params=None):
                like = eval_fu(evaluation, simulation, weights)
                return like
        else:
            print('scheme not defined.')

    spot_setup = spotpy_setup()

    if testflight:
        import matplotlib.pylab as plt
        import seaborn as sns
        sns.set_style('whitegrid', {'grid.linestyle': u'--'})

        x = spotpy_setup.parameters(spot_setup)['random']
        dummy = spotpy_setup.simulation(spotpy_setup, x)

        plt.figure(figsize=(15, 4))
        plt.subplot(121)
        plt.plot(dummy,label='simulated')
        plt.plot(eval.values,label='observed')
        plt.legend()
        plt.subplot(122)
        sns.distplot(dummy, label='simulated')
        sns.distplot(eval.values, label='observed')
        plt.legend()

        print(spotpy_setup.objectivefunction(spotpy_setup, dummy, eval.values))
        return

    else:
        if scheme == 'SCE':
            sampler_sce = spotpy.algorithms.sceua(spot_setup, dbname='SCE_'+dbname, dbformat='csv')
            sampler_sce.sample(rep, ngs=10, kstop=50, peps=0.1, pcento=0.05)

        elif scheme == 'DREAM':
            dream_sampler = spotpy.algorithms.dream(spot_setup, dbname='DREAM_'+dbname, dbformat='csv')
            dream_sampler.sample(rep, nChains=6, convergence_limit=0.05, runs_after_convergence=200)

    return


def calib_results(caldb,eval,ref='like',plotit=True):
    import hydroeval as he
    from scipy.stats import spearmanr
    from scipy.stats import linregress

    results = spotpy.analyser.load_csv_results(caldb)
    # Get fields with simulation data
    fields = [word for word in results.dtype.names if word.startswith('sim')]
    resDF = pd.DataFrame(results)

    def eval2(x, y):
        return pd.Series(np.concatenate((linregress(x, y)[:], spearmanr(x, y)[:])),
                         index=['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'spearman_corr', 'spearman_p'])

    cidx = np.where(resDF.columns.str.contains("simulat"))[0]
    resDF['KGE'] = np.nan
    resDF['spearman_corr'] = np.nan
    resDF['R2'] = np.nan
    for i in resDF.index:
        resDF.loc[i, 'spearman_corr'] = eval2(eval.values, resDF.iloc[i, cidx].values)[-2]
        resDF.loc[i, 'R2'] = eval2(eval.values, resDF.iloc[i, cidx].values)[-3] ** 2
        resDF.loc[i, 'KGE'] = he.kge(resDF.iloc[i, cidx].values, eval.values)[0][0]

    if plotit:
        import matplotlib.pylab as plt
        import seaborn as sns
        sns.set_style('whitegrid', {'grid.linestyle': u'--'})

        plt.figure(figsize=(15, 4))
        plt.subplot(121)
        for i in np.arange(len(resDF)):
            plt.plot(eval.index, resDF.iloc[i, cidx].values, ':', c='gray', alpha=0.006)

        plt.plot(eval.index, eval.values, label='Observation')
        plt.plot(eval.index, resDF.iloc[resDF['spearman_corr'].idxmax(), cidx].values, label='max spear')
        plt.plot(eval.index, resDF.iloc[resDF['KGE'].idxmax(), cidx].values, label='max KGE')
        if 'DREAM' in caldb:
            plt.plot(eval.index, resDF.iloc[resDF['like1'].idxmax(), cidx].values, label='max like')
        else:
            plt.plot(eval.index, resDF.iloc[resDF['like1'].idxmin(), cidx].values, label='max like')
        plt.legend()

        plt.subplot(122)
        sns.distplot(eval.values, label='observed')
        # sns.distplot(resDF.iloc[resDF['spearman_corr'].idxmax(),14:-4].values,label='max spear')
        # sns.distplot(resDF.iloc[resDF['KGE'].idxmax(),14:-4].values,label='max KGE')
        if ref == 'like':
            if 'DREAM' in caldb:
                sns.distplot(resDF.iloc[resDF['like1'].idxmax(), cidx].values, label='max like')
            else:
                sns.distplot(resDF.iloc[resDF['like1'].idxmin(), cidx].values, label='max like')
        elif ref == 'KGE':
            sns.distplot(resDF.iloc[resDF['KGE'].idxmax(), cidx].values, label='max KGE')
        plt.legend()

    return resDF


def cli_SC(data, x, CO2, crop='wheat', nme='dummy'):
    cli_dummy1 = cli_weather_wrp(data)

    if crop == 'wheat':
        para = {
            'AWC': 0.14,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 800.,
            'WUC': 0.096,
            's_water': 0.4,
            'maxT': 34.,
            'extremeT': 50.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.24,
            'HIp': 0.36,
            'Tsum': 1600.,
            'I50maxW': 25.,
            'I50maxH': 100.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    elif crop == 'meadow':
        para = {
            'AWC': 0.1,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 100.,
            'WUC': 0.096,
            's_water': 0.6,
            'maxT': 34.,
            'extremeT': 40.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.34,
            'HIp': 0.5,
            'Tsum': 1500.,
            'I50maxW': 25.,
            'I50maxH': 120.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    elif crop == 'maize':
        para = {
            'AWC': 0.14,
            'DDC': 0.5,
            'RCN': 66.,
            'RZD': 800.,
            'WUC': 0.096,
            's_water': 0.4,
            'maxT': 34.,
            'extremeT': 50.,
            'Tbase': 0.,
            'Topt': 15.,
            'InitialTT': 0.,
            'CO2_RUE': 0.08,
            'RUE': 1.24,
            'HIp': 0.5,
            'Tsum': 1600.,
            'I50maxW': 25.,
            'I50maxH': 100.,
            'IniBio': 1.,
            'Elev': 1.,
            'lat': 53.4
        }
    else:
        print('Crop not yet foreseen. Sorry.')
        return

    yields = np.zeros(93)
    i = 0

    if crop == 'wheat':
        para['Tbase'] = x[0]
        para['Topt'] = x[1]
        para['RUE'] = x[2]
        para['I50maxH'] = x[3]
        para['I50maxW'] = x[4]
        para['maxT'] = x[5]
        para['CO2_RUE'] = x[6]
        para['s_water'] = x[7]
        para['Tsum'] = x[8]
        para['AWC'] = x[9]
        para['DDC'] = x[10]
        para['RCN'] = x[11]
        para['RZD'] = x[12]

        sow_date = '-10-20'
        stop_date = '-09-01'

    elif crop == 'meadow':
        para['Tbase'] = x[0]
        para['Topt'] = x[1]
        para['RUE'] = x[2]
        para['I50maxH'] = x[3]
        para['I50maxW'] = x[4]
        para['maxT'] = x[5]
        para['s_water'] = x[6]
        para['Tsum'] = x[7]

        sow_date = '-03-10'
        stop_date = '-10-15'

    elif crop == 'maize':
        para['Tbase'] = x[0]
        para['Topt'] = x[1]
        para['RUE'] = x[2]
        para['I50maxH'] = x[3]
        para['I50maxW'] = x[4]
        para['maxT'] = x[5]
        para['CO2_RUE'] = x[6]
        para['s_water'] = x[7]
        para['Tsum'] = x[8]
        para['AWC'] = x[9]
        para['DDC'] = x[10]
        para['RCN'] = x[11]
        para['RZD'] = x[12]

        sow_date = '-03-15'
        stop_date = '-10-15'


    for yr in 2000 + np.arange(100)[7:]:
        sowing = pd.to_datetime(str(yr - 1) + sow_date)
        stopday = pd.to_datetime(str(yr) + stop_date)
        try:
            res = SC.SIMPLE(cli_dummy1, sowing, stopday, para, CO2.loc[yr])
            yields[i] = res.Yield.max()
        except:
            yields[i] = np.nan
        i += 1

    yields_cl = pd.DataFrame(yields, index=2000 + np.arange(100)[7:])
    yields_cl.columns = [nme]

    return yields_cl


