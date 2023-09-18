import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

def main():
    
    #-------------------------
    # initialize ee
    #-------------------------
    ee.Initialize()
    
    #-------------------------
    # data
    #-------------------------
    data_name = 'pheno'
    train_path = '../'+data_name+'-train.csv'
    test_path = '../'+data_name+'-test.csv'
    train_filtered = pd.read_csv(train_path)
    test_filtered = pd.read_csv(test_path)
    days = 24
    scale = 4000
    train_data = []
    test_data = []
    variables = ['total_precipitation_sum', 
                  'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
                  'soil_temperature_level_1', 'soil_temperature_level_1_min', 'soil_temperature_level_1_max',
                  'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_1_min', 'volumetric_soil_water_layer_1_max',
                  'surface_solar_radiation_downwards_sum', 'surface_solar_radiation_downwards_min', 'surface_solar_radiation_downwards_max',
                  'surface_pressure', 'surface_pressure_min', 'surface_pressure_max'
                 ]
    
    #-------------------------
    # train data iteration
    #-------------------------    
    for i in tqdm(range(len(train_filtered))):
        
        ## load precipitation data 
        prec = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        prec = prec.select(variables)
        
        ## end date
        if data_name == 'onetime':
            end_date = train_filtered.Observation_Date.values[i].split()[0]
            end_month, end_day, end_year = end_date.split('/')
            end_month = int(end_month)
            end_day = int(end_day)
            end_year = int(end_year)
            end = datetime(end_year, end_month, end_day)
            
            ## start date
            start_date = (datetime.strptime(end_date, "%m/%d/%Y") - timedelta(days=days)).strftime("%m/%d/%Y")
            start_month, start_day, start_year = start_date.split('/')
            start_month = int(start_month)
            start_day = int(start_day)
            start_year = int(start_year)
            start = datetime(start_year, start_month, start_day)
        else:
            end_date = train_filtered.Observation_Date.values[i]
            end_year, end_month, end_day = end_date.split('-')
            end_month = int(end_month)
            end_day = int(end_day)
            end_year = int(end_year)
            end = datetime(end_year, end_month, end_day)
            
            ## start date
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
            start_year, start_month, start_day = start_date.split('-')
            start_month = int(start_month)
            start_day = int(start_day)
            start_year = int(start_year)
            start = datetime(start_year, start_month, start_day) 
        
        prec = prec.filterDate(start, end)
        
        ## point location
        lat = train_filtered.Latitude.values[i]
        long = train_filtered.Longitude.values[i]
        point = ee.Geometry.Point([long, lat])
        
        ## get precipitation time series data
        ts_prec = prec.getRegion(point, scale).getInfo()
        ts_prec = pd.DataFrame(ts_prec)
        ts_prec.columns = ts_prec.iloc[0]
        ts_prec = ts_prec.iloc[1:,4:]
        ts_prec[ts_prec.isna()] = 0
        ts_prec[ts_prec.isnull()] = 0
        train_data.append(ts_prec.values)

    #-------------------------
    # test data iteration
    #-------------------------            
    for i in tqdm(range(len(test_filtered))):
        
        ## load precipitation data 
        prec = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        prec = prec.select(variables)
        
        ## end date
        if data_name == 'onetime':
            end_date = test_filtered.Observation_Date.values[i].split()[0]
            end_month, end_day, end_year = end_date.split('/')
            end_month = int(end_month)
            end_day = int(end_day)
            end_year = int(end_year)
            end = datetime(end_year, end_month, end_day)
            
            ## start date
            start_date = (datetime.strptime(end_date, "%m/%d/%Y") - timedelta(days=days)).strftime("%m/%d/%Y")
            start_month, start_day, start_year = start_date.split('/')
            start_month = int(start_month)
            start_day = int(start_day)
            start_year = int(start_year)
            start = datetime(start_year, start_month, start_day)
        else:
            end_date = test_filtered.Observation_Date.values[i]
            end_year, end_month, end_day = end_date.split('-')
            end_month = int(end_month)
            end_day = int(end_day)
            end_year = int(end_year)
            end = datetime(end_year, end_month, end_day)
            
            ## start date
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
            start_year, start_month, start_day = start_date.split('-')
            start_month = int(start_month)
            start_day = int(start_day)
            start_year = int(start_year)
            start = datetime(start_year, start_month, start_day) 
        
        prec = prec.filterDate(start, end)
        
        ## point location
        lat = test_filtered.Latitude.values[i]
        long = test_filtered.Longitude.values[i]
        point = ee.Geometry.Point([long, lat])
        
        ## get precipitation time series data
        ts_prec = prec.getRegion(point, scale).getInfo()
        ts_prec = pd.DataFrame(ts_prec)
        ts_prec.columns = ts_prec.iloc[0]
        ts_prec = ts_prec.iloc[1:,4:]
        ts_prec[ts_prec.isna()] = 0
        ts_prec[ts_prec.isnull()] = 0
        test_data.append(ts_prec.values)
    
    train_data = np.stack(train_data)
    test_data = np.stack(test_data)
    np.save('../'+data_name+'-train-features.npy', train_data)
    np.save('../'+data_name+'-test-features.npy', test_data)
    np.save('../variables.npy', variables)

if __name__ == "__main__":
    main()