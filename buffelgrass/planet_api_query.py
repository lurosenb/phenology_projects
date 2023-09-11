from datetime import datetime, timedelta
from planet import Session, data_filter
import json
import math

import getpass
from planet import Auth
import planet

def get_surrounding_datetimes(date_str):
    # string to a datetime object
    input_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # datetimes for one day before and one day after the input date
    day_before = input_date - timedelta(days=1)
    day_after = input_date + timedelta(days=1)
    
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
    delta_lat = 30 / meters_per_degree
    delta_lon = 30 / (meters_per_degree * math.cos(math.radians(latitude)))

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
    
    sfilter = data_filter.and_filter([
        data_filter.permission_filter(),
        data_filter.date_range_filter('acquired', gt=gt_datetime, lt=lt_datetime),
        data_filter.geometry_filter(geo_bbox)
    ])

    async with Session() as sess:
        cl = sess.client('data')
        items = [i async for i in cl.search(['PSScene'], sfilter)]
        return [i['id'] for i in items]
    
def create_request(item_id, geo_json_aoi):
    oregon_order = planet.order_request.build_request(
       name='oregon_order',
       products=[
           planet.order_request.product(item_ids=[item_id],
                                        product_bundle='analytic_udm2',
                                        item_type='PSScene')
       ],
       tools=[planet.order_request.clip_tool(aoi=geo_json_aoi)])

    return oregon_order

async def create_and_download(client, order_detail, directory):
    with planet.reporting.StateBar(state='creating') as reporter:
        order = await client.create_order(order_detail)
        reporter.update(state='created', order_id=order['id'])
        await client.wait(order['id'], callback=reporter.update_state)

    await client.download_order(order['id'], directory, progress_bar=True)

async def main(str_date, lat, long):
    async with planet.Session() as sess:
        cl = sess.client('orders')
        valid_ids = await search_items('2020-09-06', 32.354759, -110.939194)
        geo_json_aoi = create_geojson_bbox(lat, long)

        if valid_ids is not None:
            request = create_request(item_id, geo_json_aoi)
            # create and download the order
            await create_and_download(cl, request, './downloads')
        else:
            print(f'Failed for: {str_date}, {lat}, {long}')

user = input("Username: ")
pw = getpass.getpass()
auth = Auth.from_login(user,pw)
auth.store()

await main('2020-09-06', 32.354759, -110.939194)
