# Predicts geospatial GPP
# Args:
#    1) Year
#    2) Slurm ID
#    3) Exp name
#    4) Repetition

import os
import xarray as xr
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import sys
import modules.utils as utils
import sklearn
import glob
import dask.dataframe
#from dask_ml.preprocessing import DummyEncoder

debug = False

pft_replacements = dict(zip(
    [-1] + np.arange(1, 18),
    ['NAN', 'ENF', 'EBF', 'DNF', 'DBF', 'MF', 'SH', 'SH', 'SAV', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'SNO', 'BSV', 'WAT']   
))

#categorical_dtype = pd.CategoricalDtype(categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1],
#                                        ordered=False)

# without SNO and NAN, since they're excluded from the prediction
categorical_dtype = pd.CategoricalDtype(categories=['ENF', 'EBF', 'DBF', 'DNF', 'MF', 'SH', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CVM', 'BSV', 'WAT'],
                                        ordered=False)

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
        'MCD43C4v006':['b1','b2','b3','b4','b5','b6','b7', 'EVI','GCI','NDVI','NDWI','NIRv','kNDVI'],
        'MODIS_LAI':['Fpar','Lai'],
        'MODIS_LST':['LST_Day','LST_Night']
        }

    categoricals = ['MODIS_LC']

    year = int(sys.argv[1])
    slurm_id = str(sys.argv[2])
    exp_id = str(sys.argv[3])
    rep = int(sys.argv[4])
    print(year)

    # starts prediction experiment
    exp = utils.Experiment(exp_id=slurm_id, suffix=exp_id + '_' + str(rep), output_dir='predictions/' + str(year), logging=False, prepend_date=False)

    # load parameters
    training_exp_path = os.path.join('experiments', exp.suffix)
    with open(os.path.join(training_exp_path, 'parameters.txt')) as file:
        params = eval(file.read())

    params['training_model_id'] = exp.suffix 

    # load model (only for fold 0), since this is the bootstrapped model
    if params['model'] == 'random_forest':
        from models.rf import RandomForestCV as ModelWrapper

    elif params['model'] == 'h2o':
        from models.h2o import H2o as ModelWrapper

    elif params['model'] == 'autosklearn':
        from models.autosklearn import AutoSklearn as ModelWrapper

    else:
        raise AttributeError('Invalid model choice') 

    model = ModelWrapper.load(os.path.join(training_exp_path, 'fold_0'))

    # select necessary variables from dataset dict
    if len(getattr(model, 'vars', [])) == 0:
        # this applies to old models where the var list is not saved
        # get var list
        model_vars = utils.var_sets[params['variable_set']]

        # manuall add modis vars
        model_vars_ohc = model_vars + ['MODIS_LC_BSV', 'MODIS_LC_CRO', 'MODIS_LC_CVM', 'MODIS_LC_DBF', 'MODIS_LC_EBF', 'MODIS_LC_ENF', 'MODIS_LC_GRA', 'MODIS_LC_MF', 'MODIS_LC_SAV', 'MODIS_LC_SH', 'MODIS_LC_URB', 'MODIS_LC_WAT', 'MODIS_LC_WET']
        model_vars_ohc.remove('MODIS_LC')

    else:
        # this applies to models where var list is saved
        model_vars_ohc = model.vars  

    # create list with var names following the data files
    data_vars = []

    # delete OHC MODIS vars and add IGBP instead
    data_vars_modis = False
    for ii in model_vars_ohc:
        if ii.startswith('MODIS_LC'):
            if data_vars_modis == False:
                data_vars.append('IGBP')
                data_vars_modis = True
            else:
                continue

        elif ii == 'BESS-PAR':
            data_vars.append('BESS_PAR')

        elif ii == 'BESS-PARdiff':
            data_vars.append('BESS_PARdiff')

        elif ii == 'BESS-RSDN':
            data_vars.append('BESS_RSDN')

        else:
            data_vars.append(ii)

    # adjust dataset dict to variables needed (to avoid loading unnecessary data)
    dataset_dict_select = {}
    for key, values in dataset_dict.items():
        dataset_dict_select[key] = [value for value in values if value in data_vars]     

    def process_chunk(ds):
        '''applies the prediction for each chunk

        Args:
            ds(dask.DataFrame): Chunked data

        Returns:
            chunked prediction
        ''' 
        print('----')
        idx = ds[['time', 'x', 'y']]
        ds = ds[model_vars_ohc]

        # predict (no compute() needed here,)
        y_pred = model.predict(ds)

        y_pred = pd.DataFrame(y_pred, columns=['GPP'], index=ds.index)
        y_pred['x'] = idx['x']
        y_pred['y'] = idx['y']
        y_pred['time'] = idx['time']

        return y_pred

    def process_chunk_test(ds):
        '''Function for testing without expensive prediction
        '''
        idx = ds[['time', 'x', 'y']]
        ds = ds[model_vars_ohc]

        y_pred = pd.DataFrame([1.] * len(ds), columns=['GPP'], index=ds.index)
        y_pred['x'] = idx['x']
        y_pred['y'] = idx['y']
        y_pred['time'] = idx['time']

        return y_pred

    # load data into xarray
    xr_months_lst = []
    for month in range(1, 13):
        print('Month', month)    
        xr_months_lst.append(load_ds(year, month, dataset_dict_select))

    data_xarr = xr.concat(xr_months_lst, dim='time')

    print(data_xarr)
        
    # re-translate data vars to model vars
    if 'IGBP' in list(data_xarr.keys()):
        data_xarr = data_xarr.rename({'IGBP': 'MODIS_LC'})

    # convert to dask df
    ds = data_xarr.unify_chunks().to_dask_dataframe()
    ds = ds.repartition(npartitions=12*80)

    # remove regions with LCs out of interest and make MODIS_LC one-hot encoded
    if 'MODIS_LC' in ds.columns:
        ds['MODIS_LC'] = ds['MODIS_LC'].fillna(-1).astype('int32')
        ds['MODIS_LC'] = ds['MODIS_LC'].map(pft_replacements)

        # delete rows for LCs where we don't want a prediction
        ds = ds[~ds.MODIS_LC.isin(['SNO', 'BSV', 'WAT', 'NAN'])]

        ds['MODIS_LC'] = ds['MODIS_LC'].astype(categorical_dtype)

        ds = ds.categorize(columns=['MODIS_LC'])
        ds = dask.dataframe.reshape.get_dummies(ds, columns=['MODIS_LC'])

        # remove inf values
        ds = ds.replace([np.inf, -np.inf], np.nan)

    # iterate over chunks and predict
    y_pred = ds.map_partitions(process_chunk)

    # seems like there's no function yet to convert a dask array to an xarray dataset. Use pandas instead
    y_pred_df = y_pred.compute()
    y_pred_df = y_pred_df.set_index(['x', 'y', 'time'], drop=True)
    print(y_pred_df)
    y_pred_df = y_pred_df.to_xarray()

    # save
    out_path = exp.path
    print(out_path)
    print(y_pred_df)
    y_pred_df.to_zarr(out_path, mode='w', consolidated=True)

    exp.save(params=params)

    print('PYTHON DONE')

if __name__== '__main__':
    main()
