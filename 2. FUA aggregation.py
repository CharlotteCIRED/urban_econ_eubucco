# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:17:04 2022

@author: charl
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import json
from pyproj import Proj, transform
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas.tools import sjoin
import copy
from shapely import wkt
from shapely.geometry import Polygon
from sklearn import linear_model
import math
from esda.moran import Moran
import scipy
from os import listdir
import fiona

path_folder = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/"

FUA = gpd.read_file("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FUA = FUA.to_crs('epsg:3035')

for name_file in listdir(path_folder + "Data/EUBUCCO_COUNTRY/"):
    
    print(name_file)
    name_file = str(name_file)
    data = pd.read_csv(path_folder + "Data/EUBUCCO_COUNTRY/" + name_file)
    data['geometry'] = data['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data, crs='epsg:3035')

    inter_FUA = sjoin(FUA, gdf, how='inner')
    name_inter_FUA = list(np.unique(inter_FUA.eFUA_name))
     
    print(name_inter_FUA)
    
    for name_FUA in name_inter_FUA:
          
        FUA_here = FUA.loc[FUA.eFUA_name == name_FUA,:]
        gdf_FUA = sjoin(gdf, FUA_here, how='left')
        gdf_FUA = gdf_FUA.loc[gdf_FUA.eFUA_name == name_FUA,:]
        
        if name_FUA.find('/')!=-1:
            name_FUA=name_FUA.replace("/", "-")
            
        gdf_FUA.to_csv(path_folder + "Data/FUA/"+name_FUA+"_" + name_file+".csv")
        
#Correction for FUAs over different countries

folder_FUA = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/FUA/"

#Bilbao
city1 = pd.read_csv(folder_FUA + "Bilbao_v0_1-ESP_3.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Bilbao_v0_1-ESP_16.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Bilbao.csv")

#Brussels
city1 = pd.read_csv(folder_FUA + "Brussels_v0_1-BEL_1.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Brussels_v0_1-BEL_2.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city3 = pd.read_csv(folder_FUA + "Brussels_v0_1-BEL_3.csv.csv")
city3['geometry'] = city3['geometry'].apply(wkt.loads)
city3 = gpd.GeoDataFrame(city3, crs='epsg:3035')

city = pd.concat([city1, city2, city3], ignore_index=True)
city.to_csv(folder_FUA + "Brussels.csv")

#Donostia
city1 = pd.read_csv(folder_FUA + "Donostia - San Sebastián_v0_1-ESP_9.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Donostia - San Sebastián_v0_1-ESP_16.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Donostia_San_Sebastian.csv")

#Ghent
city1 = pd.read_csv(folder_FUA + "Ghent_v0_1-BEL_2.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Ghent_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Ghent.csv")

#Kortrijk
city1 = pd.read_csv(folder_FUA + "Kortrijk_v0_1-BEL_2.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Kortrijk_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Kortrijk.csv")

#Leuven
city1 = pd.read_csv(folder_FUA + "Leuven_v0_1-BEL_2.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Leuven_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Leuven.csv")

#Liège
city1 = pd.read_csv(folder_FUA + "Liège_v0_1-BEL_2.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Liège_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Liège.csv")

#Lille
city1 = pd.read_csv(folder_FUA + "Lille_v0_1-BEL_2.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Lille_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Lille.csv")

#Logroño
city1 = pd.read_csv(folder_FUA + "Logroño_v0_1-ESP_9.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Logroño_v0_1-ESP_15.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city3 = pd.read_csv(folder_FUA + "Logroño_v0_1-ESP_16.csv.csv")
city3['geometry'] = city3['geometry'].apply(wkt.loads)
city3 = gpd.GeoDataFrame(city3, crs='epsg:3035')

city = pd.concat([city1, city2, city3], ignore_index=True)
city.to_csv(folder_FUA + "Logroño.csv")

#Luxembourg
city1 = pd.read_csv(folder_FUA + "Luxembourg_v0_1-LUX.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Luxembourg_v0_1-BEL_3.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Luxembourg.csv")

#Madrid
city1 = pd.read_csv(folder_FUA + "Madrid_v0_1-ESP_4.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Madrid_v0_1-ESP_8.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Madrid.csv")

#Murcia
city1 = pd.read_csv(folder_FUA + "Murcia_v0_1-ESP_10.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Murcia_v0_1-ESP_18.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Murcia.csv")

#Torrevieja
city1 = pd.read_csv(folder_FUA + "Torrevieja_v0_1-ESP_10.csv.csv")
city1['geometry'] = city1['geometry'].apply(wkt.loads)
city1 = gpd.GeoDataFrame(city1, crs='epsg:3035')

city2 = pd.read_csv(folder_FUA + "Torrevieja_v0_1-ESP_18.csv.csv")
city2['geometry'] = city2['geometry'].apply(wkt.loads)
city2 = gpd.GeoDataFrame(city2, crs='epsg:3035')

city = pd.concat([city1, city2], ignore_index=True)
city.to_csv(folder_FUA + "Torrevieja.csv")