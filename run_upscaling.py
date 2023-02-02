# Predicts geospatial GPP
# Args:
#    1) Year
#    2) Slurm ID
#    3) Exp name
#    4) Repetition

## TODO: This needs a possibility to read the var set name and select the relevant columns, also in the right order!!

import os
import xarray as xr
import pandas as pd
import numpy as np
#import h2o
from pandas.tseries.offsets import MonthEnd
import sys
import modules.utils as utils
import sklearn
import glob
#from dask_ml.preprocessing import DummyEncoder

debug = False

pft_replacements = dict(zip(
    np.arange(1, 18),
    ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'SH', 'SH', 'SAV', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BAR', 'WAT']   
))

def load_ds(year, month, dataset_dict):
    '''Loads the explanatory variables and merges them into one xarray

    Yanghui's code

    Args:
        year (int): Year
        month (int): Month
        dataset_dict (dict): Dataset name as keys, variables as values (list)

    Returns:
        XR dataset
    '''

    zarr_path = '/global/scratch/users/yanghuikang/upscale/data/processed/monthly/'
    ds = []

    for dataset_name in dataset_dict.keys():
        var_list = dataset_dict[dataset_name]

        for var_name in var_list:

            zarr_file = zarr_path+dataset_name+'/zarr_yearly/'+var_name+'/'+str(year)

            with xr.open_zarr(zarr_file,consolidated=False) as ds_single:
                
                # select the month
                if dataset_name != 'MCD12Q1':
                    ds_single = ds_single.isel(time=(ds_single.time.dt.month==month))
                
                # round up the coordiantes to align them
                ds_single['x'] = [k.round(3) for k in ds_single.x.values]
                ds_single['y'] = [k.round(3) for k in ds_single.y.values]
                
                # fix the time dimension of MODIS Land Cover data
                if dataset_name == 'MCD12Q1':
                    ds_single['time']=pd.to_datetime([str(year)+str(month).zfill(2)], format="%Y%m") + MonthEnd(1)
                
                # rename ERA5 bands
                if dataset_name == 'ERA5':
                    era_dict = {'temperature_2m':'Tmean','potential_evaporation':'PET','skin_temperature':'Ts',
                        'evaporation_from_vegetation_transpiration':'transpiration',
                        'dewpoint_temperature_2m':'dewpoint','total_precipitation':'prcp'}
                    col_dict = {key:value for (key,value) in era_dict.items() if key in ds_single.keys()}
                    ds_single = ds_single.rename(col_dict)
                
                # remove QA bands
                if 'QA' in list(ds_single.keys()):
                    # ds_single = ds_single.rename({'QA':var_name+'_QA'})
                    ds_single = ds_single.drop_vars(['QA'])
                
                # drop spatial_ref
                if 'spatial_ref' in ds_single.dims:
                    ds_single = ds_single.drop_dims(['spatial_ref'])
                
                #ds = xr.merge([ds,ds_single])
                ds.append(ds_single)

    return xr.merge(ds) #ds

def preproc(ds, exclude_lc=[]):
    '''Pre-process data

    Removes unnecessary dimensions and variables, one-hot-encodes LC

    Args:
        ds (xr.Dataset): Data
        exclude_lc (list): Land cover types to be excluded (e.g., WAT)
    
    Returns:
        processed dataset

    TODO:
        what if MODIS_LC not available?
    '''
    # rm unnecessary dimensions and variables
    ds = ds.squeeze(dim='time',drop=True)
    
    if 'IGBP' in list(ds.keys()):
        ds = ds.rename({'IGBP': 'MODIS_LC'})
    
    # convert to dask df
    ds = ds.to_dataframe()

    if 'MODIS_LC' in ds.columns: 
        ds['MODIS_LC'] = ds['MODIS_LC'].map(pft_replacements)
        ds = ds[~ds.MODIS_LC.isin(exclude_lc + [np.nan])]
        ds = pd.get_dummies(ds, columns=['MODIS_LC'])

    # remove inf values
    ds = ds.replace([np.inf, -np.inf], np.nan)
    return ds

def load_ds_test():
    '''Dummy data'''
    data = pd.read_csv('/global/scratch/users/yanghuikang/upscale/data/site/global_sample/global_sample_10000_input_st.csv', parse_dates=True)

    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index(['x', 'y', 'time'])
    data = data.loc[(slice(None), slice(None), '2005-05-31'),:]

    data = data.drop([col for col in data if col.startswith('MODIS_PFT_')], axis=1)

    return data.to_xarray()

def predict(df, model):
    '''
    TODO: remove, deprecated
    '''
    # get pandas dtypes and create mapping dict
    mask = [(np.issubdtype(x, np.floating), np.issubdtype(x, np.integer))for x in df.dtypes]
    types = np.select(list(zip(*mask)), ['real', 'int'], default=np.nan)
    mapping = dict(zip(df.columns[types != 'nan'], types[types != 'nan']))

    #hf = h2o.H2OFrame(df, column_types=mapping)
    y_pred = model.predict(hf)
    #h2o.remove(hf)
    #y_pred = y_pred.as_data_frame()
    return y_pred


def main():   
    '''Includes executable code
    ''' 
    # list of datasets and variables
    dataset_dict = {
        'ALEXI':['ET'],
        'BESS_Rad':['BESS_PAR','BESS_PARdiff','BESS_RSDN'],
        'CSIF':['CSIF-SIFdaily','CSIF-SIFinst'],
        #'ERA5':['p1','p2'],
        'ESA_CCI':['ESACCI-sm'],
        'MCD12Q1':['IGBP'],
        'MCD43C4v006':['b1','b2','b3','b4','b5','b6','b7'], #,'EVI','GCI','NDVI','NDWI','NIRv','kNDVI'
        'MODIS_LAI':['Fpar','Lai'],
        'MODIS_LST':['LST_Day','LST_Night']
        }

    year = int(sys.argv[1])
    slurm_id = str(sys.argv[2])
    exp_id = str(sys.argv[3])
    rep = int(sys.argv[4])
    print(year)

    exp = utils.Experiment(exp_id=slurm_id, suffix=exp_id + '_' + str(rep), output_dir='predictions/' + str(year), logging=False, prepend_date=False)

    # load parameters
    training_exp_path = os.path.join('experiments', exp.suffix)
    with open(os.path.join(training_exp_path, 'parameters.txt')) as file:
        params = eval(file.read())

    params['training_model_id'] = exp.suffix 

    # load model (only for fold 0)
    if params['model'] == 'random_forest':
        from models.rf import RandomForestCV as ModelWrapper

    elif params['model'] == 'h2o':
        from models.h2o import H2o as ModelWrapper

    elif params['model'] == 'autosklearn':
        from models.autosklearn import AutoSklearn as ModelWrapper

    else:
        raise AttributeError('Invalid model choice') 

    model = ModelWrapper.load(os.path.join(training_exp_path, 'fold_0'))

    for month in range(1, 13):
        print('Month', month)

        if debug == False:     
            ds = load_ds(year, month, dataset_dict)
        else:
            ds = load_ds_test()

        date = pd.to_datetime(ds['time'][0].to_numpy())

        ds = preproc(ds, exclude_lc=['WAT'])

        # predict
        idx = ds.index
        # deletes ds
        #y_pred = predict(ds, model)
        print(list(ds.columns))
        y_pred = model.predict(ds)
        y_pred.index = idx
        y_pred['time'] = date
        y_pred = y_pred.set_index('time', append=True)

        # create xarray
        y_pred = y_pred.to_xarray()
        y_pred = y_pred.rename({'predict': 'GPP'})
    
        # save
        out_path = '{path}/{month}'.format(path=exp.path, month=date.month)
        print(out_path)
        y_pred.to_zarr(out_path, mode='w', consolidated=True)

    exp.save(params=params)

    print('PYTHON DONE')

if __name__== '__main__':
    main()
