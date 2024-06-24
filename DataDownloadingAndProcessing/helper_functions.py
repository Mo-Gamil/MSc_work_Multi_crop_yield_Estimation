import pandas as pd
import csv
import geopandas as gpd
from shapely.geometry import Polygon


import numpy as np
from tqdm.notebook import tqdm  # a progress bar
import ee
# import geetools
from datetime import datetime
import rasterio as rio
import pyproj
from shapely.geometry import Point, box
from functools import partial
from pyproj import Proj, transform
from pyproj import Transformer, CRS
from dotenv import load_dotenv
import geopandas as gpd
import ee
from datetime import datetime, timedelta
from tqdm.auto import tqdm  # Importing the auto version which is notebook-friendly
import ee
import time
import shutil
import os
import time
import glob
from pathlib import Path
import re
from datetime import datetime
from sklearn.model_selection import train_test_split


from shapely.geometry import mapping

import geemap

import fiona
import csv
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.windows import Window


from osgeo import gdal
from rasterio.enums import Resampling

import os
from osgeo import gdal
import numpy as np


def authenticate_gee():
    ee.Authenticate()
    ee.Initialize()




def read_random_points (center_points_shapefile_path):
    points = gpd.read_file(center_points_shapefile_path)
    # points = 'hi'
    return points

def get_bounding_box(point, buffer_dist): # 3360 for 224 # 3840 is used to get 256*256 pixel images with 30 meters pixel resolution
    # Define the WGS84 and UTM projection, UTM will be determined dynamically for accuracy
    wgs84 = pyproj.CRS("EPSG:4326")

    # Determine the UTM zone dynamically for the given point
    utm_zone = int((point.long + 180) / 6) + 1
    utm_crs = CRS(f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs")
    # print(utm_zone)
    # print(utm_crs)
    # Initialize transformers
    transformer_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Transform point to UTM
    point_utm_x, point_utm_y = transformer_to_utm.transform(point.long, point.lat)

    # Create bounding box in UTM
    buffer_distance = buffer_dist #3840  # Half of 7680 meters to create the box around the point
    bbox_utm = box(
        point_utm_x - buffer_distance,
        point_utm_y - buffer_distance,
        point_utm_x + buffer_distance,
        point_utm_y + buffer_distance,
    )

    # Get corners of the bounding box in UTM, then transform back to WGS84
    bottom_left_x, bottom_left_y = transformer_to_wgs84.transform(
        bbox_utm.bounds[0], bbox_utm.bounds[1]
    )
    top_right_x, top_right_y = transformer_to_wgs84.transform(
        bbox_utm.bounds[2], bbox_utm.bounds[3]
    )

    # return (bottom_left_y, bottom_left_x, top_right_y, top_right_x)
    return (bottom_left_x,bottom_left_y,top_right_x, top_right_y)

def download_sent2_image_patches_from_GEE_to_local_desk(points_shp__gdf,start_date,end_date,cloud_cover_max,buffer_dist,output_sen2_path,bands,crs):
    time_range = (ee.Date(start_date), ee.Date(end_date))
    bands = bands
    for index, row in points_shp__gdf.iterrows():
        # Get the feature's geometry (point)
        # Extract the point geometry from the current row
        (bottom_left_x,bottom_left_y,top_right_x, top_right_y) = get_bounding_box(row,buffer_dist)

        bounding_box = ee.Geometry.BBox(bottom_left_x,bottom_left_y,top_right_x, top_right_y)


        # Construct the image collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(bounding_box)
                      .filterDate(time_range[0], time_range[1])
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max)))

        # Calculate mean reflectance
        mean_image = collection.mean()

        # Select specific bands
        selected_image = mean_image.select(bands)
        # Download the image to local drive


        # output_path = os.path.join("sent2_illinois_2021",f"{row.unique_ID}.tif")
        output_path = os.path.join(output_sen2_path,f"{row.unique_id}.tif")


        geemap.ee_export_image(selected_image, filename=output_path, scale=10, region=bounding_box, crs=crs)

        print(f"Downloaded {row.unique_id} image patch for Illinois 2021.")
    print("Download completed.")



def download_CDL_patches_from_GEE_to_local_desk(points_shp__gdf, start_date, end_date, buffer_dist, output_sen2_path, crs):
    time_range = (ee.Date(start_date), ee.Date(end_date))
    
    for index, row in points_shp__gdf.iterrows():
        # Get the feature's geometry (point)
        (bottom_left_x, bottom_left_y, top_right_x, top_right_y) = get_bounding_box(row, buffer_dist)
        bounding_box = ee.Geometry.BBox(bottom_left_x, bottom_left_y, top_right_x, top_right_y)

        # Construct the image collection for CDL
        collection = (ee.ImageCollection('USDA/NASS/CDL')
                      .filterBounds(bounding_box)
                      .filterDate(time_range[0], time_range[1])
                      .select('cropland')
                      )

        # Check if the collection is empty
        if collection.size().getInfo() == 0:
            print(f"No images found for {row.unique_id} within the specified date range.")
            continue

        # Apply mask to filter only corn (class 1) and soybeans (class 5) and set background to 0
        def mask_corn_soy(image):
            corn_mask = image.eq(1).multiply(1)  # Corn as 1
            soy_mask = image.eq(5).multiply(2)  # Soybeans as 2
            combined_mask = corn_mask.add(soy_mask).selfMask()  # Combine and apply mask
            return combined_mask.rename('cropland')

        masked_collection = collection.map(mask_corn_soy)

        # Mosaic the collection to create a single image
        mosaic_image = masked_collection.mosaic()

        # Set background (NoData) to 0
        mosaic_image = mosaic_image.unmask(0)

        # Check if the mosaic has any valid data
        if mosaic_image.reduceRegion(reducer=ee.Reducer.sum(), geometry=bounding_box, scale=10).getInfo().get('cropland', 0) == 0:
            print(f"No valid corn or soybean data found for {row.unique_id}.")
            continue

        output_path = os.path.join(output_sen2_path, f"{row.unique_id}.tif")

        # Export the image to a local file
        geemap.ee_export_image(mosaic_image, filename=output_path, scale=10, region=bounding_box, crs=crs, file_per_band=False)

        print(f"Downloaded {row.unique_id} image patch for corn and soybeans.")
    print("Download completed.")


def clip_an_image(image_path,image_name, output_path,output_image_size):
    # Open the TIFF image
    with rasterio.open(image_path) as src:
        # Get the dimensions of the image
        width = src.width
        height = src.height

        # Check if the number of columns is not equal to the number of rows
        if width != height or width == height:
            # Calculate the size of the window for clipping
            window_size = min(width, height)
            
            # Calculate the dimensions for the new clipped image
            new_width = new_height = 224
            
            # Calculate the window coordinates for clipping
            col_off = (width - window_size) // 2
            row_off = (height - window_size) // 2
            
            # Read the data within the window
            window = Window(col_off, row_off, window_size, window_size)
            clipped_data = src.read(window=window)

            # Update the metadata for the new clipped image
            transform = src.window_transform(window)
            new_meta = src.meta.copy()
            new_meta.update({
                'width': new_width,
                'height': new_height,
                'transform': transform
            })

            # Write the clipped image to a new file
            with rasterio.open(output_path+'/'+f'{image_name}', 'w', **new_meta) as dst:
                dst.write(clipped_data)

            print(f"Image clipped and saved to {output_path}")
        else:
            print("Image already has equal number of columns and rows.")



# iterate over each image and perform the function clip_sen2_image
def change_patches_size_in_folder(folder_path,output_path,output_image_size):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                file_path = os.path.join(root, file)
                print(file_path)
                print(file)
                clip_an_image(file_path,file, output_path,output_image_size)





def normalize_sent2_pixel_values(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        if file.endswith('.tif'):
            input_filepath = os.path.join(input_folder, file)
            output_filepath = os.path.join(output_folder, file)
            
            # Open the input raster file
            dataset = gdal.Open(input_filepath, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"Failed to open {input_filepath}")
                continue
            
            # Get the number of bands in the raster dataset
            num_bands = dataset.RasterCount

            # Create output raster
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.CreateCopy(output_filepath, dataset, strict=0)

            # Loop through each band
            for i in range(1, num_bands + 1):
                band = output_dataset.GetRasterBand(i)
                # Read band data as an array
                band_data = band.ReadAsArray().astype(float)
                # Divide pixel values by 10000
                band_data /= 10000.0
                # Write the modified band data back to the output raster
                band.WriteArray(band_data)

            # Close the datasets
            dataset = None
            output_dataset = None

            print(f"Processed {file}")

import os
from osgeo import gdal

def normalize_sent2_pixel_values_less_size(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        if file.endswith('.tif'):
            input_filepath = os.path.join(input_folder, file)
            output_filepath = os.path.join(output_folder, file)
            
            # Open the input raster file
            dataset = gdal.Open(input_filepath, gdal.GA_ReadOnly)
            if dataset is None:
                print(f"Failed to open {input_filepath}")
                continue
            
            # Get the number of bands in the raster dataset
            num_bands = dataset.RasterCount

            # Create output raster with compression options
            driver = gdal.GetDriverByName('GTiff')
            creation_options = ['COMPRESS=LZW', 'TILED=YES']
            output_dataset = driver.Create(output_filepath, 
                                           dataset.RasterXSize, 
                                           dataset.RasterYSize, 
                                           num_bands, 
                                           gdal.GDT_Float32, 
                                           options=creation_options)

            # Set the geotransform and projection on the output dataset
            output_dataset.SetGeoTransform(dataset.GetGeoTransform())
            output_dataset.SetProjection(dataset.GetProjection())

            # Loop through each band
            for i in range(1, num_bands + 1):
                band = dataset.GetRasterBand(i)
                output_band = output_dataset.GetRasterBand(i)
                
                # Read band data as an array
                band_data = band.ReadAsArray().astype(float)
                
                # Divide pixel values by 10000
                band_data /= 10000.0
                
                # Write the modified band data back to the output raster
                output_band.WriteArray(band_data)

                # Set the no data value
                if band.GetNoDataValue() is not None:
                    output_band.SetNoDataValue(band.GetNoDataValue())

            # Close the datasets
            dataset = None
            output_dataset = None

            print(f"Processed {file}")



def setBackgroundToZero_of_CDL(input_dir, output_dir):
    """
    Sets the background (no data) values to zero for all TIFF images in the input directory
    and saves the processed images to the output directory.

    Parameters:
    input_dir (str): Path to the directory containing input TIFF files.
    output_dir (str): Path to the directory where processed TIFF files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all TIFF files in the input directory
    tiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    # Process each TIFF file
    for tiff_file in tiff_files:
        input_tiff = os.path.join(input_dir, tiff_file)
        output_tiff = os.path.join(output_dir, tiff_file)
        
        # Open the input TIFF file
        with rasterio.open(input_tiff) as src:
            # Read the image data
            image_data = src.read(1)  # Read the first band
            # image_data = np.where(image_data == 1, 1, image_data)
            # image_data = np.where(image_data == 2, 2, image_data)
            # image_data = np.where(image_data == , 0, image_data)
            # Get the no data value
            no_data_value = src.nodata

            # Replace the no data values with 0
            if no_data_value is not None:
                image_data = np.where(image_data == no_data_value, 3, image_data)
            else:
                raise ValueError(f"No data value is not set in the input TIFF file: {input_tiff}")


            # Define the metadata for the output TIFF file
            profile = src.profile
            profile.update(dtype=rasterio.float64)  # Update the data type if necessary

        # Write the modified data to the output TIFF file
        with rasterio.open(output_tiff, 'w', **profile) as dst:
            dst.write(image_data, 1)  # Write to the first band

def remap_image_pixel_values(input_path, output_path):
    # Open the input TIFF file
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    if not dataset:
        print(f"Failed to open file {input_path}")
        return

    # Read the raster band (assuming single band raster)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # Change all pixels with value 3 to 0
    data[data == 3] = 0

    # Create a new TIFF file to write the modified data
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, band.XSize, band.YSize, 1, band.DataType)

    # Set the geo-transform and projection
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    # Write the modified data to the new TIFF file
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data)

    # Close the datasets
    dataset = None
    out_dataset = None

def map_cdl_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            remap_image_pixel_values(input_path, output_path)



def count_pixel_values(image_path):
    with rasterio.open(image_path) as src:
        data = src.read(1)  # Assuming single-band image
        counts = {
            2: 0,  # soy
            1: 0,  # corn
            0: 0   # background
        }
        for value in [2, 1, 0]:
            counts[value] = (data == value).sum()
            # print(counts[value])
    return counts

def calculate_crop_yield(folder_path,points_shapefile_path):
    points_shp = gpd.read_file(points_shapefile_path)
    # points_shp=points_shp[0:10]
    crop_yield = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            counts = count_pixel_values(image_path)
            print(filename)
            print(f"Counts for {filename}: {counts}")
            filename = filename[:-4]
            point = points_shp[points_shp['unique_id'] == filename]
            Id = point['unique_id'].values[0]
            soy_y_bu_per_m2 = point['soy_bu_m2'].values[0]
            corn_y_bu_per_m2 = point['corn_bu_m2'].values[0]
            
            soy_y_bu_per_patch =counts[2] *10*10 * soy_y_bu_per_m2
            corn_y_bu_per_patch =counts[1]* 10*10 * corn_y_bu_per_m2
            
            crop_yield.append((Id,filename,soy_y_bu_per_m2, corn_y_bu_per_m2, soy_y_bu_per_patch, corn_y_bu_per_patch ))
    return crop_yield       
