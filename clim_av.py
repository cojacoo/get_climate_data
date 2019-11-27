# python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de
# climate analyse and data visualisation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.pyplot import *
import xarray as xr


def seasoner(ts, season='w', collector='sum', seasons=2):
    '''
    resamples for winter/summer with time stamps at BEGINNING
    ts is a pandas time series
    season can be w (winter) or s (summer)
    collector can be sum, mean or median
    '''
    try:
        idx = ts.index[(ts.index.month == 10) & (ts.index.day == 1)][0]
    except:
        idx = ts.index[(ts.index.month == 10)][0]

    if seasons==2:
        if (collector == 'sum'):
            serr = ts.loc[idx:].resample('2QS').sum()
        elif (collector == 'mean'):
            serr = ts.loc[idx:].resample('2QS').mean()
        elif (collector == 'min'):
            serr = ts.loc[idx:].resample('2QS').min()
        elif (collector == 'max'):
            serr = ts.loc[idx:].resample('2QS').max()
        elif (collector == 'median'):
            serr = ts.loc[idx:].resample('2QS').median()

        if (season == 'w'):
            return serr[::2]
        elif (season == 's'):
            return serr[1::2]
    else:
        if (collector == 'sum'):
            serr = ts.loc[idx:].resample('QS').sum()
        elif (collector == 'mean'):
            serr = ts.loc[idx:].resample('QS').mean()
        elif (collector == 'min'):
            serr = ts.loc[idx:].resample('QS').min()
        elif (collector == 'max'):
            serr = ts.loc[idx:].resample('QS').max()
        elif (collector == 'median'):
            serr = ts.loc[idx:].resample('QS').median()

        if (season == 'w'):
            return serr[::4]
        elif (season == 'spring'):
            return serr[1::4]
        elif (season == 's'):
            return serr[2::4]
        elif (season == 'autum'):
            return serr[3::4]


def climate_iniday(ts):
    '''
    Calculate climate indicator days for one year.
    Input ONE year of meteorological data with daily time step

    It is intended to be used like this:
    meteodata.groupby(meteodata.index.year).apply(climate_iniday).frost_day.plot()
    '''

    if (ts.index[-1] - ts.index[0]).days > 366:
        print('more than one year given!')

    ice_day = sum(ts.tasmax < 0.)
    frost_day = sum(ts.tasmin < 0.)
    hot_day = sum(ts.tasmax >= 30.)
    summer_day = sum(ts.tasmax >= 25.)
    tropic_night = sum(ts.tasmin >= 20.)
    rainy_day = sum(ts.pr >= 5.)

    return pd.DataFrame([ice_day, frost_day, summer_day, hot_day, tropic_night, rainy_day],
                        index=['ice_day', 'frost_day', 'summer_day', 'hot_day', 'tropic_night', 'rainy_day']).T


def scPDSI(cts,awc=123,rs='1M'):
    '''
    Wrapper for the self-calibrating Palmer Drought Severity Index etc.
    Input cts as pd.DataFrame of a climate time series,
    awc is the available water capacity (in mm),
    rs is the resampling frequency.
    '''

    from climate_indices import indices
    #this uses a conversion to inches! *0.0393701
    prcp = cts.pr.resample(rs).sum().values * 0.0393701
    pet = cts.et_harg.resample(rs).sum().values * 0.0393701
    awc = awc * 0.0393701
    start_year = cts.index[10].year
    end_year = cts.index[-10].year
    scpdsi, pdsi, phdi, pmdi, zindex = indices.scpdsi(prcp, pet, awc, start_year, start_year, end_year)
    dummy = pd.DataFrame([scpdsi, pdsi, phdi, pmdi, zindex]).T
    dummy.index = cts.pr.resample(rs).sum().index
    dummy.columns = ['scpdsi', 'pdsi', 'phdi', 'pmdi', 'zindex']

    return dummy


def kde(data, split_ts=1, cplot=True):
    '''
    An extended KDE plot returning the kde too.
    data is assumed to be a pd.Series with a timestamp index
    split_ts is the number of splitted KDEs to plot
    cplot indicates if a plot is created

    Example use:
        figsize(10,2)
        bla = kde(ts,3)
        title('Norderney Annual Maximum Air Temperature')
        xlabel('T (°C)')
        ylabel('Kernel Density')
        savefig('Tmax_Norderney.pdf',bbox_inches='tight')
    '''
    import matplotlib.pylab as plt
    from sklearn.neighbors import KernelDensity

    x_d = np.linspace(np.min(data) * 0.9, np.max(data) * 1.1, len(data))

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(data[:, None])

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])

    if cplot & (split_ts == 1):
        plt.fill_between(x_d, np.exp(logprob), alpha=0.4, facecolor='grey')

    lp = np.exp(logprob)
    xd = x_d

    if split_ts > 1:
        spliti = np.linspace(0, len(data), split_ts + 1).astype(int)
        cxx = ['#E69F00', '#009E73', '#0072B2', '#D55E00', '#CC79A7']
        for i in np.arange(split_ts):
            datax = data.iloc[spliti[i]:spliti[i + 1]]
            x_d = np.linspace(np.min(datax) * 0.9, np.max(datax) * 1.1, len(datax))

            # instantiate and fit the KDE model
            kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
            kde.fit(datax[:, None])

            # score_samples returns the log of the probability density
            logprob = kde.score_samples(x_d[:, None])

            if cplot:
                plt.fill_between(x_d, np.exp(logprob), alpha=0.4, facecolor=cxx[i],
                             label='-'.join([str(datax.index.year.min()), str(datax.index.year.max())]))
        plt.legend()

    if cplot:
        cmap = plt.cm.get_cmap('cividis_r')
        colors = plt.cm.cividis_r(np.linspace(0, 1, len(data)))
        colorsx = cmap(np.arange(cmap.N))

        for i in np.arange(len(data)):
            plt.plot([data.iloc[i], data.iloc[i]], [0, np.max(lp) * 1.1], c=colors[i])

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Year', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.ax.set_yticklabels(
            np.round(np.linspace(data.index.year.min(), data.index.year.max(), 6)).astype(int).astype(
                str))  # vertically oriented colorbar

    return [xd, lp]


def monthlyx(dy, dyx=1, ylab='T (°C)', clab1='Monthly Mean in Year', clab2='Monthly Max in Year', pls='cividis_r'):
    '''
    Monthly plot of time series with scatter
    dy and dyx can be two pandas time series (second one can be dropped if 1)
    ylab is the ylabel
    clab1 is the label of the first colorbar
    clab2 is the label of the first colorbar
    pls is the palette of the scatter
    '''
    cmap = cm.get_cmap(pls)
    colors = cmap(np.linspace(0, 1, len(dy.index.year.unique())))
    colorsx = cmap(np.arange(cmap.N))

    cmap1 = cm.get_cmap('gist_heat_r')
    colors1 = cm.gist_heat_r(np.linspace(0, 1, len(dy.index.year.unique())))
    colorsx1 = cmap1(np.arange(cmap1.N))

    for i in dy.index:
        plot((i.month + (np.random.rand(1) - 1.5))[0], dy[i], '.', c=colors[int(i.year - 2006)])
        if type(dyx) == int:
            pass
        else:
            plot((i.month + (np.random.rand(1) - 1.5))[0], dyx[i], '.', c=colors1[int(i.year - 2006)])

    cbar = colorbar(cm.ScalarMappable(cmap=cmap), label=clab1, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.set_yticklabels(np.round(np.linspace(dy.index.year.min(), dy.index.year.max(), 6)).astype(int).astype(str))

    if type(dyx) == int:
        pass
    else:
        cbar1 = colorbar(cm.ScalarMappable(cmap=cmap1), label=clab2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar1.ax.set_yticklabels(
            np.round(np.linspace(dy.index.year.min(), dy.index.year.max(), 6)).astype(int).astype(str))

    xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ylabel(ylab)
    return


def monthly_vio(ts,yrsep=2050,yl='Precip (mm)',cl='Monthly Sum in Year',inx='quartile',plv='RdGy',pls='cividis_r'):
    '''
    Monthly plot of time series with scatter and violins
    ts time series
    yrsep is the separation year for the two violine halfs as int or two sections with 4 int in a list or array
    yl is the ylabel
    cl is the label of the colorbar
    inx can be 'quartile' or 'stick' for the violin interior
    plv is the palette of violins
    pls is the palette of the scatter
    '''
    df1 = pd.DataFrame(ts)
    vari = df1.columns[0]
    df1['Month'] = df1.index.month
    if type(yrsep)==int:
        df1['Period'] = '-'.join([str(df1.index.year.min()),str(yrsep)])
        df1.loc[df1.index.year>yrsep,'Period'] = '-'.join([str(yrsep+1),str(df1.index.year.max())])
        df1x = df1
    else:
        df1['Period'] = '-'.join([str(yrsep[0]), str(yrsep[1])])
        df1.loc[df1.index.year >= yrsep[2], 'Period'] = '-'.join([str(yrsep[2]), str(yrsep[3])])
        df1x = df1.loc[((df1.index.year>=yrsep[0]) & (df1.index.year<=yrsep[1])) | ((df1.index.year>=yrsep[2]) & (df1.index.year<=yrsep[3]))]


    sns.violinplot(x='Month', y=vari, hue='Period', data=df1x, alpha=0.1, split=True,
               palette=plv, scale="count", inner=inx,scale_hue=False, bw=.4)

    monthlyx(df1[vari],1,yl,cl,pls)
    return


# Warming stripes motivated by the plots of Ed Hawkins and Alexander Radtke @ warmingstripes.com

def stripe_plot(ts, plv='bwr',lwx=9):
    cmap = cm.get_cmap(plv)

    tsn = ts.min().round(1)
    tsm = ts.max().round(1)
    clookup = np.round(np.linspace(tsn, tsm, int(np.round(1 + 10 * (tsm - tsn)))), 1)
    colors = cmap(np.linspace(0, 1, len(clookup)))

    for i in ts.index:
        plot([i, i], [0, 1], c=colors[np.where((clookup >= ts.loc[i] - 0.05) & (clookup <= ts.loc[i] + 0.05))[0][0]],
             lw=lwx)

    cbar = colorbar(cm.ScalarMappable(cmap=cmap), label='T (°C)', ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(np.round(np.linspace(tsn, tsm, 5)).astype(int).astype(str))



# splitting stripes plot
def sigm(x):
    return 1 / (1 + np.exp(-x))

def yshifter(ts,breakpoint=2020,yshift=0.5):
    yrs = ts.index.year

    dummy_y = np.zeros(len(yrs))
    dummy_y[yrs>breakpoint+9] = yshift
    dummy_y[(yrs>=breakpoint-10) & (yrs<=breakpoint+10)] = sigm(np.linspace(-10,10,21))*yshift

    return pd.Series(dummy_y,index=ts.index)

def div_stripe(ts, breakpoint=2020, plv='bwr', lwx=9):
    cmap = cm.get_cmap(plv)

    tsn = ts.min(axis=0).min().round(1)
    tsm = ts.max(axis=0).max().round(1)
    coli = ts.columns

    clookup = np.round(np.linspace(tsn, tsm, int(np.round(1 + 10 * (tsm - tsn)))), 1)
    colors = cmap(np.linspace(0, 1, len(clookup)))

    y_baseline = yshifter(ts, breakpoint, 0.1)

    for i in ts.index:
        plot([i, i], [y_baseline[i], y_baseline[i] + 1],
             c=colors[np.where((clookup >= ts.loc[i, coli[0]] - 0.05) & (clookup <= ts.loc[i, coli[0]] + 0.05))[0][0]],
             lw=lwx, solid_capstyle='projecting')
        plot([i, i], [(-1 * y_baseline[i]), (-1 * y_baseline[i]) - 1],
             c=colors[np.where((clookup >= ts.loc[i, coli[1]] - 0.05) & (clookup <= ts.loc[i, coli[1]] + 0.05))[0][0]],
             lw=lwx, solid_capstyle='projecting')

    cbar = colorbar(cm.ScalarMappable(cmap=cmap), label='T (°C)', ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels(np.round(np.linspace(tsn, tsm, 5)).astype(int).astype(str))

