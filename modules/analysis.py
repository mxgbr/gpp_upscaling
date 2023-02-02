from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
import itertools
import glob
import os
from sklearn.linear_model import LinearRegression
from datetime import date

import modules.utils as utils

# plotting styles
# color pallette
lc_colors = {
    'WET': '#3185FC',
    'BAR': '#FFBA08',
    'SNO': '#7F7F7F',
    'CRO': '#D00000',
    'ENF': '#00CDC8',
    'EBF': '#1B998B',
    'DBF': '#8FE388',
    'MF': '#D5E002',
    'GRA': '#9B68CE',
    'SH': '#E28826',
    'SAV': '#FF9B85',
    'WAT': '#335BA3',
    'REST': '#46237A'
}
sns.set_palette(sns.color_palette(list(lc_colors.values())))

cmap_gpp_1 = matplotlib.colors.LinearSegmentedColormap.from_list('gpp_1', [
    (0, '#3185FC'),
    (0.4, '#1B998B'),
    (0.9, '#FFBA08'),
    (1, '#FFBA08')
])

cmap_gpp_2 = matplotlib.colors.LinearSegmentedColormap.from_list('gpp_2', [
    (0, '#FFBA08'),
    (0.5, '#FFFFFF'),
    (1, '#1B998B')
])

cmap_gpp_3 = matplotlib.colors.LinearSegmentedColormap.from_list('gpp_3', [
    (0, '#FFFFFF'),
    (0.9, '#D00000'),
    (1, '#D00000')
])

# default chart layout
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

def eval_metrics(exp_id, exp_dir='experiments/', out_path=None, min_months=0):
    '''Evaluates repeated CVs

    Reads experiments and combines folds to sensitivity analysis.

    Args:
        exp_id (int): Experiment ID
        exp_dir (str): Directory of experiments
        out_path (str): Saving directory

    Returns:
        Dataframe with repetitions as rows and metrics as columns
    '''
    # get all array_ids for experiment
    repetitions = glob.glob(os.path.join(exp_dir, str(exp_id) + '_*'))

    final_df = []
    for rep in repetitions:
        _, y, _, _, _, test_idx, y_pred = utils.Experiment.load(rep)
        if len(y_pred) == 0:
            print('Experiment ' + str(rep) + ' couln\'t be loaded.')
            continue
        y.name = 'GT'

        test_idx = list(itertools.chain(*test_idx))
        y_pred = pd.concat(y_pred)
        y_pred.name = 'Pred'
        y_eval = pd.concat([y.iloc[test_idx], y_pred], axis=1)

        y_eval = y_eval[(y_eval.GT.groupby('SITE_ID').transform(lambda x: x.count()) >= min_months)]

        r2_overall = sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values)
        r2_trend = sklearn.metrics.r2_score(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values)
        r2_anomalies = sklearn.metrics.r2_score(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred, detrend=True).values)
        r2_sites = sklearn.metrics.r2_score(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values)
        r2_msc = sklearn.metrics.r2_score(msc(y_eval.GT).values, msc(y_eval.Pred).values)

        rmse_overall = sklearn.metrics.mean_squared_error(y_eval.GT.values, y_eval.Pred.values, squared=False)
        rmse_trend = sklearn.metrics.mean_squared_error(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values, squared=False)
        rmse_anomalies = sklearn.metrics.mean_squared_error(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred).values, squared=False)
        rmse_sites = sklearn.metrics.mean_squared_error(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values, squared=False)
        rmse_msc = sklearn.metrics.mean_squared_error(msc(y_eval.GT).values, msc(y_eval.Pred).values, squared=False)

        final_df.append([rep.split('/')[-1], r2_overall, r2_trend, r2_anomalies, r2_sites, r2_msc, rmse_overall, rmse_trend, rmse_anomalies, rmse_sites, rmse_msc])

    final_df = pd.DataFrame(final_df, columns=['exp_id', 'r2_overall', 'r2_trend', 'r2_anomalies', 'r2_sites', 'r2_msc', 'rmse_overall', 'rmse_trend', 'rmse_anomalies', 'rmse_sites', 'rmse_msc']).set_index('exp_id')
    
    if out_path is not None:
        os.makedirs(os.path.join(out_path, exp_id), exist_ok=True)
        final_df.to_csv(os.path.join(out_path, exp_id, 'metrics.csv'))

    return final_df

def plt_model_comparison(data, out_dir, var_set, model, metric, ylims=[], **kwargs):
    '''Creates violin plot for models and variables

    Args:
        data (pd.DataFrame): Data frame of modeling results
        out_dir (str): Path to output directory
        var_set (str): Name of var set column
        model (str): Name of model column
        metric (str): Name of metric column
        ylims (list): List of tuples for y limits (ymin, ymax), set none if auto
    '''
    ax = sns.violinplot(data=data, hue=var_set, x=model, y=metric, showfliers=True, inner="quartile", **kwargs)
    #sns.swarmplot(data=data, hue=var_set, x=model, y=metric, dodge=True, palette='dark:black', legend=False)

    if len(ylims) > 0:
        if ylims[0] is not None:
            ax.set_ylim(*ylims[0])

    ax.set_ylabel('$r^2$')
    ax.set_title('Overall')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'benchmark_r2_overall.pdf'))

    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    sns.violinplot(data=data, hue=var_set, x=model, y='r2_trend', ax=ax[0,0], inner="quartile", **kwargs)
    #sns.swarmplot(data=data, hue=var_set, x=model, y='r2_trend', dodge=True, palette='dark:black', ax=ax[0,0], legend=False)

    sns.violinplot(data=data, hue=var_set, x=model, y='r2_sites', ax=ax[0,1], inner="quartile", **kwargs)
    #sns.swarmplot(data=data, hue=var_set, x=model, y='r2_sites', dodge=True, palette='dark:black', ax=ax[0,1], legend=False)

    sns.violinplot(data=data, hue=var_set, x=model, y='r2_msc', ax=ax[1,0], inner="quartile", **kwargs)
    #sns.swarmplot(data=data, hue=var_set, x=model, y='r2_msc', dodge=True, palette='dark:black', ax=ax[1,0], legend=False)

    sns.violinplot(data=data, hue=var_set, x=model, y='r2_anomalies', ax=ax[1,1], inner="quartile", **kwargs)
    #sns.swarmplot(data=data, hue=var_set, x=model, y='r2_anomalies', dodge=True, palette='dark:black', ax=ax[1,1], legend=False)

    if len(ylims) > 1:
        for idx, ax_item in enumerate(ax.flatten()):
            if ylims[idx+1] is not None:
                ax_item.set_ylim(*ylims[idx+1])

    ax[0, 0].set_title('Trend')
    ax[0, 1].set_title('Across-site Variability')
    ax[1, 0].set_title('Mean Seasonal Cycle')
    ax[1, 1].set_title('Anomalies')

    ax[0, 0].set_ylabel('$r^2$')
    ax[1, 0].set_ylabel('$r^2$')
    ax[0, 1].set_ylabel('$r^2$')
    ax[1, 1].set_ylabel('$r^2$')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'benchmark_r2_decomp.pdf'))

def eval_lc(exp_id, exp_dir, site_df, out_path=None, min_months=0):
    '''Evaluated performance per LC class

    Args:
        exp_id (int): Experiment ID
        exp_dir (str): Directory of experiments
        site_df (pd.DataFrame): Frame including IGBP and koppen_main columns with SITE_ID as index
        out_path (str): Saving directory

    Returns:
        Dataframe with repetitions as rows and metrics as columns
    '''
    # get all array_ids for experiment
    repetitions = glob.glob(os.path.join(exp_dir, str(exp_id) + '_*'))

    final_df = []
    for rep in repetitions:
        _, y, _, _, _, test_idx, y_pred = utils.Experiment.load(rep)
        y.name = 'GT'

        test_idx = list(itertools.chain(*test_idx))
        y_pred = pd.concat(y_pred)
        y_pred.name = 'Pred'
        y_eval = pd.concat([y.iloc[test_idx], y_pred], axis=1)

        y_eval = y_eval[(y_eval.GT.groupby('SITE_ID').transform(lambda x: x.count()) >= min_months)]

        y_eval = y_eval.merge(site_df[['IGBP', 'koppen_main']], left_on='SITE_ID', right_index=True)

        r2_overall = y_eval.groupby(['SITE_ID', 'IGBP']).apply(lambda x: sklearn.metrics.r2_score(x.GT, x.Pred))
        r2_overall.name = 'r2_overall'
        r2_msc = y_eval.groupby(['SITE_ID', 'IGBP']).apply(lambda x: sklearn.metrics.r2_score(msc(x.GT), msc(x.Pred)))
        r2_msc.name = 'r2_msc'
        r2_anomalies = y_eval.groupby(['SITE_ID', 'IGBP']).apply(lambda x: sklearn.metrics.r2_score(iav(x.GT, detrend=True), iav(x.Pred, detrend=True)))
        r2_anomalies.name = 'r2_anomalies'
        rep_idx = pd.Series(rep.split('/')[-1], index=r2_anomalies.index, name='exp_id')

        final_df.append(pd.concat([rep_idx, r2_overall, r2_msc, r2_anomalies], axis=1))

    final_df = pd.concat(final_df).reset_index(level='IGBP', drop=False).set_index('exp_id', append=True)
    
    if out_path is not None:
        os.makedirs(os.path.join(out_path, exp_id), exist_ok=True)
        final_df.to_csv(os.path.join(out_path, exp_id, 'metrics_lc.csv'))

    return final_df

def plt_lc_violin(data, out_dir, lc, exp, **kwargs):
    '''Creates violin plot for all models in different LCs

    Args:
        data (pd.DataFrame): Data frame of modeling results
        out_dir (str): Path to output directory
        lc (str): Name of lc column
        exp (str): Name of experiment repetition id
        metric (str): Name of metric column
        **kwargs: Arguments for seaborn
    '''
    fig, ax = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for exp_id in data[exp].unique():
        df = data[data[exp] == exp_id]

        sns.violinplot(data=df, x=lc, y='r2_overall', ax=ax[0], inner='box', **kwargs)
        sns.violinplot(data=df, x=lc, y='r2_msc', ax=ax[1], inner='box', **kwargs)
        sns.violinplot(data=df, x=lc, y='r2_anomalies', ax=ax[2], inner='box', **kwargs)

        # workaround for transparent faces
        for ax_ii in ax:
            for col in ax_ii.collections[::2]:
                col.set_facecolor((0, 0, 0, 0))
                col.set_edgecolor((0, 0, 0, 0.4))
            
            for col in ax_ii.collections[1::2]:
                col.set_alpha(0)
                
            for lines in ax_ii.lines:
                lines.set_alpha(0)

    ax[0].set_ylim(-4, 1.5)
    ax[1].set_ylim(-4, 1.5)
    ax[2].set_ylim(-4, 1.5)

    li = ax[0].axhline(y=0, color='gray', zorder=0, linestyle='--')
    li.set_alpha(1)
    ax[1].axhline(y=0, color='gray', zorder=0, linestyle='--')
    ax[2].axhline(y=0, color='gray', zorder=0, linestyle='--')

    ax[0].set_title('Overall')
    ax[1].set_title('Mean Seasonal Cycle')
    ax[2].set_title('Anomalies')

    ax[0].set_ylabel('$r^2$')
    ax[1].set_ylabel('$r^2$')
    ax[2].set_ylabel('$r^2$')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'benchmark_r2_lc.pdf'))

def plt_lc_meanbox(data, out_dir, lc, exp, **kwargs):
    '''Creates box plot for one model in different LCs

    Args:
        data (pd.DataFrame): Data frame of modeling results
        out_dir (str): Path to output directory
        lc (str): Name of lc column
        exp (str): Name of experiment repetition id
        metric (str): Name of metric column
        **kwargs: Arguments for seaborn
    '''
    fig, ax = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    ax[0].axhline(y=0, color='gray', zorder=0, linestyle='--')
    ax[1].axhline(y=0, color='gray', zorder=0, linestyle='--')
    ax[2].axhline(y=0, color='gray', zorder=0, linestyle='--')

    data = data.drop(exp, axis=1).groupby(['SITE_ID', lc]).mean().reset_index(level=lc)

    sns.boxplot(data=data, x=lc, y='r2_overall', ax=ax[0], color=sns.color_palette()[0], **kwargs)
    sns.boxplot(data=data, x=lc, y='r2_msc', ax=ax[1], color=sns.color_palette()[0], **kwargs)
    sns.boxplot(data=data, x=lc, y='r2_anomalies', ax=ax[2], color=sns.color_palette()[0], **kwargs)

    ax[0].set_ylim(-7.5, 1.5)
    ax[1].set_ylim(-10, 1.5)
    ax[2].set_ylim(-3, 1.5)

    ax[0].set_title('Overall')
    ax[1].set_title('Mean Seasonal Cycle')
    ax[2].set_title('Anomalies')

    ax[0].set_ylabel('$r^2$')
    ax[1].set_ylabel('$r^2$')
    ax[2].set_ylabel('$r^2$')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'benchmark_r2_lc_box.pdf'))

def evaluation_plot(y_eval):
    '''
    Creates evaluation plots for model performance

    Args:
        y_eval (pd.DataFrame): DataFrame with ground truth values in `GT` column and predictions in `Pred` column. Index must consist of SITE_ID and Date (datetimeindex)
    '''
    r2_overall = sklearn.metrics.r2_score(y_eval.GT.values, y_eval.Pred.values)
    r2_trend = sklearn.metrics.r2_score(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values)
    # r2_seasonal = sklearn.metrics.r2_score(seasonal_cycle_mean(y_eval.GT).values, seasonal_cycle_mean(y_eval.Pred).values)
    r2_anomalies = sklearn.metrics.r2_score(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred, detrend=True).values)
    r2_sites = sklearn.metrics.r2_score(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values)
    r2_iav = sklearn.metrics.r2_score(iav(y_eval.GT).values, iav(y_eval.Pred).values)
    r2_msc = sklearn.metrics.r2_score(msc(y_eval.GT).values, msc(y_eval.Pred).values)

    rmse_overall = sklearn.metrics.mean_squared_error(y_eval.GT.values, y_eval.Pred.values, squared=False)
    rmse_trend = sklearn.metrics.mean_squared_error(across_site_trend(y_eval.GT).values, across_site_trend(y_eval.Pred).values, squared=False)
    # rmse_seasonal = sklearn.metrics.mean_squared_error(seasonal_cycle_mean(y_eval.GT).values, seasonal_cycle_mean(y_eval.Pred).values, squared=False)
    rmse_anomalies = sklearn.metrics.mean_squared_error(iav(y_eval.GT, detrend=True).values, iav(y_eval.Pred).values, squared=False)
    rmse_sites = sklearn.metrics.mean_squared_error(across_site_variability(y_eval.GT).values, across_site_variability(y_eval.Pred).values, squared=False)
    rmse_iav = sklearn.metrics.mean_squared_error(iav(y_eval.GT).values, iav(y_eval.Pred).values, squared=False)
    rmse_msc = sklearn.metrics.mean_squared_error(msc(y_eval.GT).values, msc(y_eval.Pred).values, squared=False)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # plot 1
    x = y_eval.Pred
    y = y_eval.GT
    r2_plot(x, y, ax[0, 0], r2=r2_overall, rmse=rmse_overall)
    ax[0, 0].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 0].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 0].set_title('Overall prediction')

    # plot 2
    x = across_site_trend(y_eval.Pred)
    y = across_site_trend(y_eval.GT)
    r2_plot(x, y, ax[0, 1], r2=r2_trend, rmse=rmse_trend)
    ax[0, 1].set_xlabel('Predicted slope GPP [$gC m^{-2} d^{-1} month^{-1}$]')
    ax[0, 1].set_ylabel('FLUXNET slope GPP [$gC m^{-2} d^{-1} month^{-1}$]')
    ax[0, 1].set_title('Trend')

    # plot 3
    # x = seasonal_cycle_mean(y_eval.Pred)
    # y = seasonal_cycle_mean(y_eval.GT)
    # r2_plot(x, y, ax[0, 1], r2=r2_seasonal, rmse=rmse_seasonal)
    # ax[0, 1].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    # ax[0, 1].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    # ax[0, 1].set_title('Seasonal cycle')

    # plot 5
    x = across_site_variability(y_eval.Pred)
    y = across_site_variability(y_eval.GT)
    r2_plot(x, y, ax[0, 2], r2=r2_sites, rmse=rmse_sites)
    ax[0, 2].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 2].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[0, 2].set_title('Across-site variability')

    # plot 7
    x = msc(y_eval.Pred)
    y = msc(y_eval.GT)
    im = r2_plot(x, y, ax[1, 0], r2=r2_msc, rmse=rmse_msc)
    ax[1, 0].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 0].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 0].set_title('Mean seasonal cycle')

    # plot 6
    x = iav(y_eval.Pred)
    y = iav(y_eval.GT)
    r2_plot(x, y, ax[1, 1], r2=r2_iav, rmse=rmse_iav)
    ax[1, 1].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 1].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 1].set_title('Interanual variability')

    # plot 4
    x = iav(y_eval.Pred, detrend=True)
    y = iav(y_eval.GT, detrend=True)
    r2_plot(x, y, ax[1, 2], r2=r2_anomalies, rmse=rmse_anomalies)
    ax[1, 2].set_xlabel('Predicted GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 2].set_ylabel('FLUXNET GPP [$gC m^{-2} d^{-1}$]')
    ax[1, 2].set_title('Interannual variability (detrended)')

    # ax_cbar = fig.add_axes([0.3, 0.1, 0.4, 0.03])
    # plt.colorbar(im, cax=ax_cbar, orientation='horizontal')

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    return fig

def annual_mean(ts, transform=False):
    '''Calculates annual mean values'''
    grp = ts.groupby(['SITE_ID', ts.index.get_level_values('Date').year])
    if transform:
        return grp.transform('mean')
    return grp.mean()

def across_site_variability(ts):
    return ts.groupby('SITE_ID').mean()

def iav(ts, detrend=False):
    ts = ts.sub(msc(ts, transform=True))
    if detrend == True:
        ts = ts.sub(trend(ts))

    return ts

def msc(ts, transform=False, no_mean=False):
    if no_mean:
        ts = ts.sub(ts.groupby(['SITE_ID']).transform('mean'))
    grp = ts.groupby(['SITE_ID', ts.index.get_level_values('Date').month])
    if transform:
        return grp.transform('mean')
    return grp.mean()

def lr_model(series_inp, return_coef=False):
    series = series_inp.droplevel(0)
    x = ((series.index - pd.to_datetime(date(1970, 1, 31))) / np.timedelta64(1, 'M')).values.round().reshape(-1, 1)
    y = series.values
    lr = LinearRegression()
    lr.fit(x, y)
    if return_coef == True:
            return lr.coef_
    return pd.Series(lr.predict(x), index=series_inp.index)

def trend(ts):
    '''Calculates trend and intercept
    
    Groups by sites in performs a linear regression for each site

    Args:
        ts (DataFrame): time series with sites and dates as index

    Returns:
        Slope in pd.Series
    '''     
    grp = ts.groupby('SITE_ID', group_keys=False).apply(lr_model)
    return grp

def across_site_trend(ts):
    '''
    Calculates trend on site level from linear regression

    Args:
        ts (DataFrame): time series with sites and dates as index

    Returns:
        pd.Series with trend-only values
    '''
    grp = ts.groupby('SITE_ID').apply(lr_model, return_coef=True)
    return pd.Series(np.concatenate(grp.values).ravel(), index=grp.index)