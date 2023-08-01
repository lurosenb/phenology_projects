# NOTE: Follow the authentication workflow in "satellite_imagery_exploration.ipynb" in order to initialize ee

import ee
from datetime import datetime, timedelta
import requests
import shutil
import pandas as pd

ee.Initialize()

def maskS2clouds(image):
    qa = image.select('QA60')

    # bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)

def get_satellite_image(lat, lon, observation_date, buffer_meters=200, date_range_days=7, band_variety=['B4', 'B3', 'B2']):
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # define the region
    point = ee.Geometry.Point([lon, lat])

    # buffer the point and create a square region around it
    region = point.buffer(buffer_meters).bounds()

    # define the date range based on the observation date
    start_date = (datetime.strptime(observation_date, "%Y-%m-%d") - timedelta(days=date_range_days)).strftime("%Y-%m-%d")
    end_date = (datetime.strptime(observation_date, "%Y-%m-%d") + timedelta(days=date_range_days)).strftime("%Y-%m-%d")

    # filter the collection for a specific date
    filtered = collection.filterDate(start_date, end_date)

    # filter the collection by region
    filtered = filtered.filterBounds(region)

    # check if the filtered collection is empty
    if filtered.size().getInfo() == 0:
        raise ValueError("No images found in the specified date range.")

    # sort the collection by metadata
    sorted = filtered.sort('system:time_start', False)

    # get the first image in the sorted collection
    image = ee.Image(sorted.first())

    # check if the image is None
    if image is None:
        raise ValueError("No images found after sorting.")

    # define the visualization parameters
    visParams = {
        'bands': band_variety,
        'min': 0.0,
        'max': 3000,
    }

    url = image.getThumbUrl({
        'region': region.getInfo(),
        'scale': 10,  # adjust scale as needed - Sentinel-2 images have 10m resolution
        **visParams
    })

    return url

# test or run the function
quick_test = False
if __name__ == '__main__':
    if quick_test:
        url = get_satellite_image(32.145890, -110.958221, '2019-07-09')
        print(url)
    else:
        band_type = 'ir' # 'color'
        df = pd.read_csv(f'data/buffelgrass_one_time.csv')
        df_filtered = df[['Observation_ID', 'Observation_Date', 'Create_Date','Latitude', 'Longitude', 'Abundance_Name']]

        for index, row in df_filtered.iterrows():
            row_id = row['Observation_ID']
            lat = row['Latitude']
            lon = row['Longitude']
            date = row['Observation_Date'].strftime("%Y-%m-%d") # assuming the date is in datetime format

            # get the image URL
            try:
                if 'ir':
                    url = get_satellite_image(lat, lon, date, band_variety=['B8', 'B11', 'B12'])
                else:
                    url = get_satellite_image(lat, lon, date)
            except ValueError as e:
                print(f"Skipping ({row_id}, {lat}, {lon}, {date}) due to error: {str(e)}")
                continue

            # make the request and open the image
            response = requests.get(url, stream=True)

            # check if the image was retrieved successfully
            if response.status_code == 200:
                # set decode_content value to True, 
                # otherwise the downloaded image file's size will be zero.
                response.raw.decode_content = True

                if 'ir':
                    filename = f'images/{row_id}_ir.png'
                else:
                    filename = f'images/{row_id}.png'
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)

                # print(f"Image successfully downloaded: {filename}")
            else:
                print(f"Image couldn't be retrieved for ({lat}, {lon}, {date}).")