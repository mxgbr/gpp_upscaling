from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import itertools

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
sns.set_palette(sns.color_palette(list(lc_colors.values()))

# default chart layout
plt.rcParams['figure.figsize'] = [9, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

def eval_metrics(exp_id, exp_dir='experiments/', out_path=None):
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
    for rep in repetitions
        _, y, _, _, _, test_idx, y_pred = utils.Experiment.load(os.path.join(exp_dir, rep))
        y.name = 'GT'
        y_pred.name = 'Pred'

        test_idx = list(itertools.chain(*test_idx))
        y_pred = pd.concat(y_pred)
        y_eval = pd.concat([y.iloc[test_idx], y_pred])

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

        final_df.append([rep, r2_overall, r2_trend, r2_anomalies, r2_sites, r2_msc, rmse_overall, rmse_trend, rmse_anomalies, rmse_sites, rmse_msc])

    final_df = pd.DataFrame(final_df, columns=['exp_id', 'r2_overall', 'r2_trend', 'r2_anomalies', 'r2_sites', 'r2_msc', 'rmse_overall', 'rmse_trend', 'rmse_anomalies', 'rmse_sites', 'rmse_msc']).set_index('exp_id')
    
    if out_path is not None:
        os.makedirs(os.path.join(out_path, exp_id), exist_ok=True)
        final_df.to_csv(os.path.join(out_path, exp_id, 'metrics.csv'))

    return final_df

def plt_model_comparison():
    '''Creates violin plot for models and variables

    Args:
        data (pd.DataFrame): Data frame of modeling results
        var_set (str): Name of var set column
        model (str): Name of model column
        metric (str): Name of metric column
    '''
    pass

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
