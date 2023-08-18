# Creates maps from XR datasets
#
# Args:
#    1. Path, can use {month}, {year}, {pred_id}
#    2. Prediction ID (<pred_id>_<training_exp_id>)
#    3. Start date MM-YYYY
#    4. End date MM-YYYY
#    5. Map type

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import scipy

import modules.analysis as analysis

from dateutil import rrule
from datetime import datetime, timedelta
import re
import warnings

import sys
import os
import pickle

def read_dask(path, date_range, pred_ids, variables=['GPP'], dims=('lat', 'lon', 'time'), freq='1M'):
    '''Reads zarr or netcdf files

    Args:
        path (str): Path to file
        date_range (tuple): (start, end)
        pred_ids (list): List of predictor IDs
        variables (list): Variable names in list of strings
        dims (tuple): Dimension names in tuple
        freq (str): Frequency string

    Returns:
        xarray dataset with dimensions (lat, lon, time, rep)
    '''

    # determine file type
    if os.path.splitext(path)[1] in ['.nc4', '.nc']:
        engine = 'netcdf4'
    else:
        engine = 'zarr'

    date_start = date_range[0]
    date_end = date_range[1]

    # determine frequency
    if bool(re.search(r'{month.*}', path)) & bool(re.search(r'{year.*}', path)):
        rule = rrule.MONTHLY
        date_kwargs = lambda x: {'year': x.year, 'month': x.month}
    elif bool(re.search(r'{year.*}', path)):
        rule = rrule.YEARLY
        date_kwargs = lambda x: {'year': x.year}
    else:
        # no time dirs, creates one dummy month to read file
        rule = rrule.MONTHLY
        date_end = date_start + timedelta(days=28)
        date_kwargs = lambda x: {}

    # determine if multiple models
    if '{pred_id}' in path:
        model_kwargs = lambda y: {'pred_id': y}

    else:
        model_kwargs = lambda y: {}
        pred_ids = range(0, 1)

    # load files
    sets_model = []
    for model_id in pred_ids:
        sets = []
        for date in rrule.rrule(rule, dtstart=date_start, until=date_end + timedelta(days=-1)):
            try:
                ds = xr.open_dataset(path.format(**date_kwargs(date), **model_kwargs(model_id)), engine=engine, chunks='auto')
            except:
                warnings.warn('Could not read ' + str(path.format(**date_kwargs(date), **model_kwargs(model_id))))
            else:
                sets.append(ds)

        sets_model.append(xr.concat(sets, dim=dims[2])[variables])

    with dask.config.set({"array.slicing.split_large_chunks": True}):
        ds = xr.concat(sets_model, dim='rep')

    # rename dimensions
    ds = ds.rename({'y': 'lat', 'x': 'lon'})

    print(ds)
    print('Range:')
    print(date_range)

    # sel time range
    ds = ds.sel(time=slice(*date_range))

    # resample
    ds = ds.resample(time=freq).mean(dim='time')

    return ds

def create_dummy_data():
    '''Creates dummy data to display on the map
    '''
    lats = np.sort(np.random.rand(100) * 90*2 - 90).astype(int)
    lons = np.sort(np.random.rand(100) * 180*2 - 180).astype(int)
    reps = [1, 2]

    vals = np.random.rand(100, 100, 1, 2) * 12

    dummy_data = xr.Dataset(
        data_vars=dict(
            GPP=(['lon', 'lat', 'time', 'rep'], vals)
        ),
        coords=dict(
            lon=(['lon'], lons),
            lat=(['lat'], lats),
            time=np.atleast_1d(pd.to_datetime('31/05/2005', dayfirst=True)),
            rep=reps
        )
    )

    print(dummy_data)

    return dummy_data

def mask(ds, lsmask=True, vegmask=False, sea_val=0, noveg_val=0, set_invalid_0=False, custom=None, custom_val=np.nan):
    '''Masks dataset with land-sea mask and vegetation mask

    Replaces sea and non-vegetated areas with replacement values

    Args:
        ds (xarray.Dataset): Dataset with spatial coordinates x, y
        lsmask (bool): Apply lsm
        vegmask (bool): Apply veg
        sea_val (float): Value for masked regions
        nonveg_val (float): Value for masked regions
        set_invalid_0 (bool): Use if dataset already contains masked regions
        custom (xr.Dataset): Mask dataset for custom masking
        custom_val (float): Value for masked regions

    Returns:
        xarray dataset

    '''
    lon = ds['lon']
    lat = ds['lat']

    if vegmask == True:
        # get vegetation mask
        veg_mask_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/mcd12c1_veg_mask'
        veg_mask = xr.open_zarr(veg_mask_path,consolidated=False)
        veg_mask = veg_mask.rename({'veg_mask': 'GPP', 'x': 'lon', 'y': 'lat'})
        veg_mask = veg_mask.interp(coords={'lon': lon, 'lat': lat}, method='nearest')

        # replace masked locations with 0
        ds = ds.where(veg_mask > 0, noveg_val)

    if lsmask == True:
        # get land-sea mask
        lsm_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc'
        lsm = xr.open_dataset(lsm_path,engine='netcdf4')
        lsm = lsm.rename({'longitude':'lon','latitude':'lat'})
        lsm = lsm.rename({'lsm': 'GPP'})
        lsm = lsm.squeeze('time').drop_vars(['time'])
        # adjust x axis from 0-360 to -180-180
        lsm.coords['lon'] = (lsm.coords['lon'] + 180) % 360 - 180
        lsm = lsm.sortby(lsm.lon)
        lsm = lsm.interp(coords={'lon': lon, 'lat': lat}, method='nearest')

        # replace masked locations with 0
        ds = ds.where(lsm > 0, sea_val)

    if custom is not None:
        ds = ds.where(custom > 0, custom_val)

    if set_invalid_0 == True:
        # replaces masked out values with 0, usefull if ds is masked already with nan or neg. values
        ds = ds.where(ds > 0, 0)

    return ds

def create_map(xds, out_path, cmap='Greens', label='', vmin=None, vmax=None, extend='neither', pickle_fig=True, dataset=True, title=''):
    '''Creates and saves a map

    Args:
        xds (xr.DataSet): Dataset with dimensions time, x, y
        out_path (str): Path incl file name for saving the file
        cmap: Colormap
        label (str): Label
        vmin (float): Min value
        vmax (float): Max value
        extend (str): Extension arrow for colorbar
        pickle_fig (bool): Indicator if fig should be pickled additionally
        dataset (bool): Set true if xds is a dataset
        title (str): Figure title
    '''
    if dataset:
        xds = xds.to_array().squeeze()#.transpose()

    fig = plt.figure(figsize=(9, 7))
    crs = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, zorder=0)
    ax.coastlines(zorder=1)

    plt_im = xds.plot.pcolormesh(ax=ax, cmap=cmap, transform=data_crs, add_colorbar=False, vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(plt_im, location='bottom',fraction=0.04, pad=0.045, extend=extend)
    cbar.set_label(label)

    plt.title(title)
    plt.tight_layout()
    if pickle_fig == True:
        pickle.dump(fig, open(out_path + '.pkl', 'wb'))
        pickle.dump(xds, open(out_path + '_data.pkl', 'wb'))

    plt.savefig(out_path)

def map_mean(xds):
    '''Averages map over time

    Args:
        xds (xr.DataSet): Dataset with time dimension

    Returns:
        mean
    '''
    return xds.mean(dim=['rep', 'time'])

def map_msc(xds):
    '''Calculates MSC amplitude

    Args:
        xds (xr.DataSet): Dataset with time dimension

    Returns:
        msc amplitude
    '''
    # calc mean ts
    xds = xds.mean(dim='rep')

    # calc amplitude
    msc = xds.groupby('time.month').mean()
    msc_amp = (msc.max(dim='month') - msc.min(dim='month')) / 2
   
    return msc_amp

def map_trend(xds):
    '''Calculates trend

    Resamples data to annual time steps for computational reasons, maps significant p values

    Args:
        xds (xr.Dataset): Dataset with time dimension

    Returns:
        slopes
    '''
    # calc mean ts
    xds = xds.mean(dim='rep')

    ds_trend = xds.resample(time='1Y').mean(dim='time') * xds.time.dt.days_in_month * 12
    ds_trend['time'] = ds_trend['time'].dt.year

    def lr(x, y):
        '''Linear regression, includes p values
        '''
        slope, _, _, p, _ = scipy.stats.linregress(x, y)
        return np.array([slope, p])

    ds_trend = ds_trend.squeeze().chunk(dict(time=-1))

    coefs = xr.apply_ufunc(lr, 
                ds_trend['time'],
                ds_trend, 
                input_core_dims=[["time"], ['time']], 
                output_core_dims=[["stat"]],
                output_sizes=dict(stat= 2),
                vectorize=True,
                output_dtypes=['float64'],
                dask='parallelized'
                )

    coefs = mask(coefs.sel(stat=0), lsmask=False, custom=(coefs.sel(stat=1) < 0.05))
    print(coefs)

    return coefs

def map_anomalies(xds):
    '''Calculates anomalies

    Args:
        xds (xr.DataSet): Dataset with time dimension

    Returns:
        std of anomalies
    '''
    # calc mean ts
    iav = xds.mean(dim='rep')

    # subtract from month-mean, then calc std over entire ts
    iav = iav.groupby('time.month') - iav.groupby('time.month').mean(dim='time')
    iav_std = iav.std(dim='time')
    
    return iav_std

def map_err(xds, abs=True):
    '''Calculates standard error

    Args:
        xds (xr.DataSet): Dataset with time dimension

    Returns:
        std error of ensemble
    '''
    # Standard Error sem = std / sqrt(N) 
    std_err = xds.std(dim='rep') / (xds.count(dim='rep')**0.5)
    std_err = std_err.mean(dim='time')

    if abs == False:
        std_err = std_err * 100 / np.fabs(xds.mean(dim=['rep', 'time']))

    return std_err

if __name__ == '__main__':
    path = sys.argv[1]
    pred_id = sys.argv[2]
    # start date
    date_start = datetime.strptime(sys.argv[3], '%m-%Y')
    # end date (open interval)
    date_end = datetime.strptime(sys.argv[4], '%m-%Y')
    map_type = sys.argv[5]

    # 30 repetitions
    repetitions = range(0, 30)
    pred_ids = [pred_id + '_' + str(ii) for ii in repetitions]

    # read data
    data = read_dask(path, (date_start, date_end), pred_ids, dims=('y', 'x', 'time'))
    #data = create_dummy_data()

    # create saving path
    out_path = os.path.join('analysis/maps/', pred_id, date_start.strftime('%m%Y') + '-' + date_end.strftime('%m%Y'))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    def dask_mean(data):
        ## default: lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan
        data = mask(data, lsmask=False, vegmask=False)
        xds_mean = map_mean(data)
        create_map(xds_mean, os.path.join(out_path, 'mean_nomask.pdf'), cmap=analysis.cmap_gpp_1, label='GPP [$gC m^{-2} d^{-1}$]', vmin=0, vmax=10, extend='max', title='Mean')

    def dask_trend(data):
        data = mask(data, lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan)
        xds_trend = map_trend(data)
        create_map(xds_trend, os.path.join(out_path, 'trend.pdf'), label='GPP [$gC m^{-2} y^{-2}$]', vmin=-20, vmax=20, extend='both', cmap=analysis.cmap_gpp_2, pickle_fig=False, title='Trend')

    def dask_msc(data):
        data = mask(data, lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan)
        msc_amp = map_msc(data)
        create_map(msc_amp, os.path.join(out_path, 'msc.pdf'), label='GPP [$gC m^{-2} d^{-1}$]', vmin=0, vmax=6, extend='max', cmap=analysis.cmap_gpp_1, title='Seasonality')

    def dask_anomalies(data):
        data = mask(data, lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan)
        anomalies = map_anomalies(data)
        create_map(anomalies, os.path.join(out_path, 'anomalies.pdf'), label='GPP [$gC m^{-2} d^{-1}$]', vmin=0, vmax=1.5, extend='max', cmap=analysis.cmap_gpp_1, title='Anomalies')

    def dask_err_abs(data):
        data = mask(data, lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan)
        std_err = map_err(data, abs=True)
        create_map(std_err, os.path.join(out_path, 'std_err_abs.pdf'), label='Standard Error [$gC m^{-2} d^{-1}$]', vmin=0, vmax=0.1, extend='max', cmap=analysis.cmap_gpp_3, title='Standard Error')

    def dask_err(data):
        data = mask(data, lsmask=True, vegmask=True, sea_val=np.nan, noveg_val=np.nan)
        std_err = map_err(data, abs=False)
        create_map(std_err, os.path.join(out_path, 'std_err.pdf'), label='Standard Error [%]', vmin=0, vmax=15, extend='max', cmap=analysis.cmap_gpp_3, title='Relative Standard Error')


    # run computation
    #delayed_objs = [
        #dask.delayed(dask_mean)(data),
        #dask.delayed(dask_msc)(data),
        #dask.delayed(dask_anomalies)(data),
        #dask.delayed(dask_trend)(data),
        #dask.delayed(dask_err)(data)
    #]
    #[x.compute() for x in delayed_objs]

    dask_err(data)
    dask_err_abs(data)
    dask_mean(data)
    dask_msc(data)
        
    print('PYTHON DONE')