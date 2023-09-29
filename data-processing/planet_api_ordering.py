from planet import Auth
from planet import Session, data_filter
import planet

from datetime import datetime, timedelta
import json
import math

import os
import traceback

import pandas as pd
import numpy as np

import rasterio
import matplotlib.pyplot as plt

from tqdm import tqdm

import asyncio
import traceback
from IPython.display import clear_output 

auth = Auth.from_key('KEY')
auth.store()

onetime_or_pheno = 'onetime'

def convert_to_datetime_tuple(s):
    date_str, _ = s.split("_", 1)
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return (s, date_obj)

def process_string_list(str_list):
    tuple_list = [convert_to_datetime_tuple(s) for s in str_list]
    
    unique_dates = set([t[1].date() for t in tuple_list])
    
    num_unique_days = len(unique_dates)
    
    return num_unique_days, tuple_list

def get_first_entry_per_day(tuple_list):
    first_entry_per_day = {}
    for original_str, date_obj in tuple_list:
        date_only = date_obj.date()
        if date_only not in first_entry_per_day:
            first_entry_per_day[date_only] = original_str

    return list(first_entry_per_day.values())

def get_surrounding_datetimes(date_str):
    # string to a datetime object
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # datetimes for one day before and one day after the input date
    day_before = input_date - timedelta(days=25)
    day_after = input_date
    
    # format for use in the filter
    gt_datetime = datetime(day_before.year, day_before.month, day_before.day, 0)
    lt_datetime = datetime(day_after.year, day_after.month, day_after.day, 0)
    
    return gt_datetime, lt_datetime

def create_date_range_filter(date_str):
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # the dates for one day before and after the input date
    day_before = input_date - timedelta(days=1)
    day_after = input_date + timedelta(days=1)
    gte_str = day_before.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    lte_str = day_after.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    # different format for date range filter object
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": gte_str,
            "lte": lte_str
        }
    }
    
    return date_range_filter

def get_middle_item(lst):
    if not lst:
        return None

    if len(lst) == 1:
        return lst[0]

    middle_idx = len(lst) // 2
    
    return lst[middle_idx]

def create_geojson_bbox(latitude, longitude):
    # length in meters for each side of the bounding box
    meters_per_degree = 111000 # approx
    delta_lat = 90 / meters_per_degree
    delta_lon = 90 / (meters_per_degree * math.cos(math.radians(latitude)))

    # four corners of the bounding box
    south_west = [longitude - delta_lon / 2, latitude - delta_lat / 2]
    south_east = [longitude + delta_lon / 2, latitude - delta_lat / 2]
    north_east = [longitude + delta_lon / 2, latitude + delta_lat / 2]
    north_west = [longitude - delta_lon / 2, latitude + delta_lat / 2]

    geojson_geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                south_west,
                south_east,
                north_east,
                north_west,
                south_west  # last thing closes  loop
            ]
        ]
    }
    
    return geojson_geometry

async def search_items(date_str, latitude, longitude):
    gt_datetime, lt_datetime = get_surrounding_datetimes(date_str)
    geo_bbox = create_geojson_bbox(latitude, longitude)

    print(gt_datetime, lt_datetime)

    sfilter = data_filter.and_filter([
        data_filter.permission_filter(),
        data_filter.date_range_filter('acquired', gt=gt_datetime, lt=lt_datetime),
        data_filter.geometry_filter(geo_bbox)
    ])

    async with Session() as sess:
        cl = sess.client('data')
        items = [i async for i in cl.search(['PSScene'], sfilter)]
        return [i['id'] for i in items]
    
def create_request(item_ids, geo_json_aoi, pheno_id=0):
    oregon_order = planet.order_request.build_request(
       name='buffelgrass_order'+str(pheno_id),
       products=[
           planet.order_request.product(item_ids=item_ids,
                                        product_bundle='analytic_sr_udm2',
                                        item_type='PSScene')
       ],
       tools=[planet.order_request.clip_tool(aoi=geo_json_aoi)],
       order_type='partial')

    return oregon_order

async def create_and_download(client, order_detail, directory):
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(order_detail)
        reporter.update(state='created', order_id=order['id'])
        await client.wait(order['id'], callback=reporter.update_state)

    await client.download_order(order['id'], directory, progress_bar=False)

def raster_image(image_file, obs_id, date, colors_to_include=['blue','green','red'], output_dir='rastered_all'):
    my_raster_image = rasterio.open(image_file)

    def scale(band): 
        return band / 5000.0

    blue = np.ones_like(my_raster_image.read(1))
    green = np.ones_like(my_raster_image.read(2))
    red = np.ones_like(my_raster_image.read(3))
    ir = np.ones_like(my_raster_image.read(4))

    if 'blue' in colors_to_include:
        blue = scale(my_raster_image.read(1))
    if 'green' in colors_to_include:
        green = scale(my_raster_image.read(2))
    if 'red' in colors_to_include:
        red = scale(my_raster_image.read(3))
    if 'ir' in colors_to_include:
        ir = scale(my_raster_image.read(4))

    my_image = np.dstack((red, green, blue))

    if np.any(np.isnan(my_image)) or np.any(np.isinf(my_image)):
        print(f"Warning: Image contains invalid values. Skipping {image_file}.")
        return

    if my_image.shape[-1] != 3:
        print(f"Warning: Image shape is not valid for display. Skipping {image_file}.")
        return

    plt.imshow(my_image)
    plt.axis('off') 

    plt.savefig(f'rastered_all/{obs_id}_{date}.png', bbox_inches='tight', pad_inches=0)

    plt.close()

def process_file(full_path, obs_id, date):
    output_dir='rastered_all'
    if not os.path.exists(output_dir+f'/{obs_id}_{date}.png'):
        raster_image(full_path, obs_id, date, output_dir=output_dir)

pheno_df = pd.read_csv('data/status_intensity_observation_data.csv')
status_leaves = pheno_df[(pheno_df['Phenophase_Description'] == 'Leaves (grasses)') & (pheno_df['Intensity_Value'] != '-9999')]
pheno_df = status_leaves[['Observation_ID', 'Observation_Date', 'Latitude', 'Longitude', 'Intensity_Value']]
pheno_df['Abundance_Binary'] = pheno_df['Intensity_Value'].apply(lambda x: 1 if x == '75-94%' or x == '50-74%' or x == '95% or more' else 0)
pheno_df = pheno_df.sort_values('Observation_Date', ascending=[False])

one_time_df = pd.read_csv('data/buffelgrass_one_time.csv')
one_time_df = one_time_df[['Observation_ID', 'Observation_Date', 'Latitude', 'Longitude', 'Abundance_Name']]
one_time_df['Observation_Date'] = one_time_df['Observation_Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M").strftime("%Y-%m-%d"))
one_time_df['Abundance_Binary'] = one_time_df['Abundance_Name'].apply(lambda x: 1 if x == '75-94%' or x == '50-74%' or x == '95% or more' else 0)
one_time_df = one_time_df.sort_values('Observation_Date', ascending=[False])

df_filtered = pd.concat([pheno_df,one_time_df])

band_type = 'normal'
str_dates, lats, longs, ids = [], [], [], []
for index, row in tqdm(df_filtered.iterrows()):
    row_id = row['Observation_ID']
    lat = row['Latitude']
    lon = row['Longitude']
    date = row['Observation_Date']
    
    if band_type == 'ir':
        filename = f'pheno/{row_id}_ir.png'
    else:
        filename = f'pheno/{row_id}.png'

    str_dates.append(date)
    lats.append(lat)
    longs.append(lon)
    ids.append(row_id)

start_dir = "planet_downloads"

# exponential backoff parameters
rate_limit = 10  
time_interval = 1 / rate_limit

async def main(str_dates, lats, longs, ids):
    tasks = []
    
    for idx, (str_date, lat, long, id) in enumerate(zip(str_dates, lats, longs, ids)):
        tasks.append(gen_request(str_date, lat, long, id, idx))
    
    await asyncio.gather(*tasks)

requests_list = []

async def gen_request(str_date, lat, long, id, idx):
    max_retries = 5  
    backoff_factor = 1 
    retry = 0
    
    while retry <= max_retries:
        try:
            await asyncio.sleep(time_interval * idx)
            
            async with planet.Session() as sess:
                cl = sess.client('orders')
                valid_ids = await search_items(str_date, lat, long)
                
                if valid_ids:
                    geo_json_aoi = create_geojson_bbox(lat, long)
                    num_unique_days, tuple_list = process_string_list(valid_ids)
                    first_entry_list = get_first_entry_per_day(tuple_list)
                    request = create_request(first_entry_list, geo_json_aoi)
                    requests_list.append((f"{id}", request))
                    # clear output when request is successful
                    clear_output(wait=True)  
                    print(f"Request for {id} successful.")
                    break 
                else:
                    print(f'Failed for: {str_date}, {lat}, {long}')
                    retry += 1 
                    if retry <= max_retries:
                        delay = backoff_factor * (2 ** (retry - 1))
                        print(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay) 
                    else:
                        clear_output(wait=True)  
                        print(f"Max retries reached for {id}. Giving up.")
        except Exception as e:
            print(f"An exception occurred: {e}")
            traceback.print_exc()
            retry += 1 
            if retry <= max_retries:
                delay = backoff_factor * (2 ** (retry - 1))
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay) 
            else:
                clear_output(wait=True)
                print(f"Max retries reached for {id}. Giving up.")

await main(str_dates, lats, longs, ids)

rate_limit = 2
time_interval = 1 / rate_limit

batch_size = 50
batch_interval = 60

async def main(requests):
    batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
    
    for idx, batch in enumerate(batches):
        tasks = []
        for index, (id, request) in enumerate(batch):
            dir_path = os.path.join(start_dir, f"{id}")
            if os.path.exists(dir_path) and os.listdir(dir_path):
                print(f"Directory for id {id} already exists and is not empty. Skipping...")
            else:
                tasks.append(single_run(id, request, index))
        
        await asyncio.gather(*tasks)

        if idx < len(batches) - 1:
            print(f"Waiting for {batch_interval/60} minutes before sending the next batch...")
            await asyncio.sleep(batch_interval)
            clear_output(wait=True)


async def single_run(id, request, idx):
    max_retries = 2
    backoff_factor = 2
    retry = 0
    
    while retry <= max_retries:
        try:
            await asyncio.sleep(time_interval * idx)
            
            download_dir = os.path.join("pheno", f"{id}")
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            
            async with planet.Session() as sess:
                cl = sess.client('orders')
                await create_and_download(cl, request, download_dir)
                clear_output(wait=True)
                print(f"Request for {id} successful.")
                break
        except Exception as e:
            print(f"An exception occurred: {e}")
            traceback.print_exc()
            retry += 1
            if retry <= max_retries:
                delay = backoff_factor * (2 ** (retry - 1))
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                clear_output(wait=True)
                print(f"Max retries reached for {id}. Giving up.")

await main(requests_list)