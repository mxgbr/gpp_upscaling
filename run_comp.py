# This file contains code to compare our predictions to fluxcom and fluxsat

import xarray as xr
from datetime import datetime
import pandas as pd
import numpy as np
import dask.dataframe as dd

if __name__ == '__main__':
    # paths
    path_as = 'predictions/{year}/14246832_2022110313584614_{rep}'
    reps = range(0, 30)

    path_fluxcom = 'data/fluxcom_v006/rs/monthly/GPP.RS_V006.FP-ALL.MLM-ALL.METEO-NONE.4320_2160.monthly.{year}.nc'
    path_fluxsat = 'data/fluxsat/FluxSat/GPP_FluxSat_daily_v2_{year}{month}.nc4'
    path_trendy = ''

    start_date = datetime.strptime('01-2001', '%m-%Y')
    end_date = datetime.strptime('01-2020', '%m-%Y')

    # load as
    ds_as = []
    for date in pd.date_range(start_date, end_date, freq='Y'):
        ds_as_y = []
        for rep in reps:
            ds_as_y.append(xr.open_dataset(path_as.format(year=date.year, rep=rep), chunks='auto', engine='zarr'))
            
        ds_as.append(xr.concat(ds_as_y, dim='rep'))
    ds_as = xr.concat(ds_as, dim='time')
    ds_as = ds_as.mean(dim='rep')
    ds_as = ds_as.rename({'y': 'lat', 'x': 'lon'})

    # load fluxsat
    ds_fs = []
    for date in pd.date_range(start_date, end_date, freq='M'):
        try:
            ds_fs.append(xr.open_dataset(path_fluxsat.format(year=date.year, month="%02d" % (date.month,)), chunks='auto'))
        except:
            print(str(date) + ' not loaded')
        
    ds_fs = xr.concat(ds_fs, dim='time')[['GPP']].resample(time='M').mean()

    # load fluxcom
    ds_fc = []
    for date in pd.date_range(start_date, end_date, freq='Y'):
        ds_fc.append(xr.open_dataset(path_fluxcom.format(year=date.year), chunks='auto'))
        
    ds_fc = xr.concat(ds_fc, dim='time')[['GPP']].resample(time='M').mean()

    # load land-sea mask and vegetation mask
    veg_mask_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/mcd12c1_veg_mask'
    veg_mask = xr.open_zarr(veg_mask_path,consolidated=False)
    veg_mask = veg_mask.rename({'veg_mask': 'GPP', 'x': 'lon', 'y': 'lat'})
    veg_mask = veg_mask.interp(coords={'lon': ds_as['lon'], 'lat': ds_as['lat']}, method='nearest')
    veg_mask = veg_mask > 0

    lsm_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc'
    lsm = xr.open_dataset(lsm_path,engine='netcdf4')
    lsm = lsm.rename({'longitude':'lon','latitude':'lat'})
    lsm = lsm.rename({'lsm': 'GPP'})
    lsm = lsm.squeeze('time').drop_vars(['time'])
    lsm = lsm > 0

    # adjust x axis from 0-360 to -180-180
    lsm.coords['lon'] = (lsm.coords['lon'] + 180) % 360 - 180
    lsm = lsm.sortby(lsm.lon)
    lsm = lsm.interp(coords={'lon': ds_as['lon'], 'lat': ds_as['lat']}, method='nearest')

    comb = lsm * veg_mask

    # randomly sample coordinates (only non-NaN locations)
    n = 10000

    # this samples locations where at least one GPP values is not NaN
    ds_nonnan = comb.stack(loc=['lat', 'lon']).to_dask_dataframe()
    ds_nonnan = ds_nonnan[ds_nonnan.GPP]

    # weighted by latitude to represent the areas
    weights = np.cos(np.deg2rad(ds_nonnan.lat))
    loc_s = ds_nonnan.loc[np.random.choice(ds_nonnan.index, n, p=weights/np.sum(weights)), ['lat', 'lon']]
    lat_s = xr.DataArray(loc_s['lat'], dims='loc')
    lon_s = xr.DataArray(loc_s['lon'], dims='loc')

    # sub-sample datasets
    ds_as_s = ds_as.sel(lat=lat_s, lon=lon_s, method='nearest')
    ds_fc_s = ds_fc.sel(lat=lat_s, lon=lon_s, method='nearest')
    ds_fs_s = ds_fs.sel(lat=lat_s, lon=lon_s, method='nearest')

    # change coordinates to sample coordinates
    ds_as_s['lat'] = lat_s
    ds_as_s['lon'] = lon_s
    ds_as_s = ds_as_s.rename({'GPP': 'GPP_as'})
    ds_fc_s['lat'] = lat_s
    ds_fc_s['lon'] = lon_s
    ds_fc_s = ds_fc_s.rename({'GPP': 'GPP_fc'})
    ds_fs_s['lat'] = lat_s
    ds_fs_s['lon'] = lon_s
    ds_fs_s = ds_fs_s.rename({'GPP': 'GPP_fs'})

    ds = xr.merge([ds_as_s, ds_fc_s, ds_fs_s]).unify_chunks()
    df = ds.to_dask_dataframe()

    # calc evaluation metrices (RMSE)
    def rmse(x, gt_col, pred_col):
        return ((x[pred_col] - x[gt_col])**2).mean() ** 0.5

    def r2(x, gt_col, pred_col):
        return 1 - ((x[gt_col] - x[pred_col])**2).sum() / ((x[gt_col] - x[gt_col].mean())**2).sum()

    df_r2_fc = df.groupby('loc').apply(r2, 'GPP_fc', 'GPP_as', meta=pd.Series(dtype=float))
    df_r2_fs = df.groupby('loc').apply(r2, 'GPP_fs', 'GPP_as', meta=pd.Series(dtype=float))

    dd.to_csv(df_r2_fc, 'analysis/ds_comparison/14246832_2022110313584614/r2_fc.csv', single_file=True)
    dd.to_csv(df_r2_fs, 'analysis/ds_comparison/14246832_2022110313584614/r2_fs.csv', single_file=True)