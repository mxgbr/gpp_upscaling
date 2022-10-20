from matplotlib import cm
import matplotlib.pyplot as plt

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
