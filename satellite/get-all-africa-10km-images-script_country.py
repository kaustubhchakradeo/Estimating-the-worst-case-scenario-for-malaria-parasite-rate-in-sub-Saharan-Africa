from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import os
import ee
print('imports done')

service_account = 'id-230224@ee-chakradeokaustubh.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'credentials.json')
ee.Initialize(credentials)
print('init done')

gauth = GoogleAuth()
scopes = ['https://www.googleapis.com/auth/drive']
gauth.credentials = ServiceAccountCredentials(service_account, 'credentials.json')

drive = GoogleDrive(gauth)


import time
#import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Here we have different csv files containing the lat lons for all countries. 
africa_all_db = pd.read_csv('/home/download_ee/country_Guinea.csv')

def downloadMapDataToDrive(bandSelection = "RGB", db = africa_all_db,
                           batchSize = 5000, batchStart=0, resolution = 0.04166665 ,
                           dims = '224x224'):
    
    print('starting downloadMapDataToDrive at :', batchStart, flush=True)
    #These bands are reasonable, but also explore others.
    if bandSelection == "RGB":
        bands = ['B4','B3','B2']
        #bands = ['SR_B4','SR_B3','SR_B2']
    elif bandSelection == "IR":
        bands = ['B7','B6','B5']
    else:
        print("pick custom bands")

    # Landsat has since updated their collection
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
    raster = ee.Algorithms.Landsat.simpleComposite(collection, 50, 10).select(bands)

    for i in range(batchStart, batchSize):
        lonMin =  db['lon'].iloc[i] - resolution
        latMin =  db['lat'].iloc[i] - resolution
        lonMax =  db['lon'].iloc[i] + resolution
        latMax =  db['lat'].iloc[i] + resolution
        geometry = ee.Geometry.Rectangle([lonMin, latMin, lonMax, latMax])

        geometry = geometry['coordinates'][0]
        #Google doesn't like dots in the name, so replacing with dd
        #pw=str(db['pixel_width'].iloc[i]).replace('.', 'dd')
        #ph=str(db['pixel_height'].iloc[i]).replace('.', 'dd')
        lon=str(db['lon'].iloc[i]).replace('.', 'dd')
        lat=str(db['lat'].iloc[i]).replace('.', 'dd')
        #cell_name=str(db['cellname'].iloc[i]).replace('.','dd')
        country=str(db['country_code'].iloc[i]).replace('.','dd')
        fileName = str(i)+"-IMAGE_"+bandSelection+"_"+str(resolution)[2:5]+"_"+dims+"_"+str(lon)+"_"+str(lat)+"_"+str(country)

        print(fileName, flush=True)
        # Images can only be saved to drive
        task = ee.batch.Export.image.toDrive(image=raster,
                                             description="imageToDrive",
                                             folder="guinea",
                                             fileNamePrefix=fileName,
                                             dimensions=dims,
                                             region=geometry)
        #drive_folder = 'ee'
        #target_path = os.path.join(drive_folder, fileName)
        #if os.path.exists(target_path):
        #    print("Image already exists in the directory. Skipping export.")
        #else:    
        task.start()
        time.sleep(10)


        if (i+1) % 100 == 0:
            print('sleeping to write', flush=True)
            time.sleep(10*60)

downloadMapDataToDrive(bandSelection = "RGB",
                       db = africa_all_db,
                       batchSize = len(africa_all_db),
                       batchStart=1,
                       resolution = 0.04166665,
                       dims = '224x224')