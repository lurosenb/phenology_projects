import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def main():

    
    # ----------------
    # iterate each split
    # ----------------
    
    print('-- Load Data...')
    ee.Initialize()
    data_path = '../buffelgrass-onetime-test.csv'
    df_filtered = pd.read_csv(data_path)
    label = df_filtered.Abundance_Binary.values ## labels
    days = 24
    pred = []
        
    for i in tqdm(range(len(df_filtered))):
            
        ## load precipitation data 
        prec = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        prec = prec.select('total_precipitation_sum')
    
        ## end date
        end_date = df_filtered.Observation_Date.values[i].split()[0]
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
        
        ## filter
        prec = prec.filterDate(start, end)
        
        ## point location
        lat = df_filtered.Latitude.values[i]
        long = df_filtered.Longitude.values[i]
        point = ee.Geometry.Point([long, lat])
        
        
        ## get precipitation time series data
        try:
            ts_prec = data['prcp'].values*0.0393701 ## 1mm = 0.0393701 inches        
            pred.append(1*(sum(ts_prec)>1.7))
        except:
            pred.append(0)
    
    tn, fp, fn, tp = confusion_matrix(pred, label).ravel()            
    acc = 100*(tn+tp)/len(label)
    fp_rate = 100*(fp)/len(label)
    fn_rate = 100*(fn)/len(label)
    
    print(f'Accumulative Days {days}.')
    print(f'Accuracy: {acc}%')
    print(f'FP: {fp_rate}%')
    print(f'FN: {fn_rate}%')  
    
if __name__ == "__main__":
    main()