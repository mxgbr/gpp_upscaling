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
import numpy as np
import pandas as pd

import modules.analysis as analysis

from dateutil import rrule
from datetime import datetime, timedelta
import re
import warnings

import sys
import os

def read_dask(path, date_range, pred_ids, variables=['GPP'], dims=('lat', 'lon', 'time'), freq='1M'):
    '''Reads zarr or netcdf files

    Args:
        path (str): Path to file
        date_range (tuple): (start, end)
        variables (list): Variable names in list of strings
        dims (tuple): Dimension names in tuple
        freq (str): Frequency string
        pred_ids (list): List of predictor IDs

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
                sets.append(xr.open_dataset(path.format(**date_kwargs(date), **model_kwargs(model_id)), engine=engine, chunks='auto'))
            except:
                warnings.warn('Could not read ' + str(path.format(**date_kwargs(date), **model_kwargs(model_id))))

        sets_model.append(xr.concat(sets, dim=dims[2])[variables])

    with dask.config.set({"array.slicing.split_large_chunks": True}):
        ds = xr.concat(sets_model, dim='rep')

    # rename dimensions
    ds = ds.rename({dims[0]: 'lat', dims[1]: 'lon', dims[2]: 'time'})

    print(ds)
    print('Range:')
    print(date_range)

    # sel time range
    # TODO disabled, does not work well withc chunking
    ds = ds.sel(time=slice(*date_range))

    print(ds)

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

def create_map(xds, out_path, cmap='Greens', label='', vmin=None, vmax=None, extend='neither'):
    '''Creates and saves a map

    Args:
        xds (xr.DataSet): Dataset with dimensions time, x, y
        cmap: Colormap
        label (str): Label
        vmin (float): Min value
        vmax (float): Max value
        extend (str): Extension arrow for colorbar
        out_path (str): Path incl file name for saving the file

    Returns:

    '''
    print(xds)
    xds = xds.to_array().squeeze().transpose()
    print(xds)

    fig = plt.figure(figsize=(9, 7))
    crs = ccrs.PlateCarree()
    data_crs = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_global()
    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, zorder=0)
    ax.coastlines(zorder=1)

    plt_im = xds.plot.pcolormesh(ax=ax, cmap=cmap, transform=data_crs, add_colorbar=False, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(plt_im, location='bottom',fraction=0.04, pad=0.045, extend=extend)
    cbar.set_label(label)

    plt.title('')
    plt.tight_layout()
    plt.savefig(out_path)

def map_mean(xds):
    '''Averages map over time

    Args:
        xds (xr.DataSet): Dataset with time dimension

    Returns:
        mean
    '''
    return xds.mean(dim='rep').mean(dim='time')

def map_msc():
    '''Calculates MSC amplitude
    '''
    pass

def map_trend():
    '''Calculates trend
    '''
    pass

def map_annomalies():
    '''Calculates annomalies
    '''
    pass

def map_err():
    '''Calculates standard error
    '''
    pass

if __name__ == '__main__':
    path = sys.argv[1]
    pred_id = sys.argv[2]
    # start date
    date_start = datetime.strptime(sys.argv[3], '%m-%Y')
    # end date (open interval)
    date_end = datetime.strptime(sys.argv[4], '%m-%Y')
    map_type = sys.argv[5]

    # 30 repetitions
    repetitions = range(1, 3)
    pred_ids = [pred_id + '_' + str(ii) for ii in repetitions]

    # read data
    data = read_dask(path, (date_start, date_end), pred_ids, dims=('y', 'x', 'time'))
    #data = create_dummy_data()

    # mask data
    ## TODO

    # create saving path
    out_path = os.path.join('analysis/maps/', pred_id, date_start.strftime('%m%Y') + '-' + date_end.strftime('%m%Y'))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # run computation
    if map_type == 'mean':
        xds_mean = map_mean(data)
        create_map(xds_mean, os.path.join(out_path, 'mean.pdf'), cmap='plasma', label='GPP [$gC m^{-2} d^{-1}$]', vmax=12, extend='max')

    print('PYTHON DONE')