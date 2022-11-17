# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:30:29 2022

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
import csv

path_folder = "C:/Users/charl/OneDrive/Bureau/"
country = "BEL"

if country == "LUX":

    EUBUCCO = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-LUX.csv")
    EUBUCCO['geometry'] = EUBUCCO['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(EUBUCCO, crs='epsg:3035')
    
elif country == "BEL":
    
    EUBUCCO1 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-BEL_1.csv")
    EUBUCCO1['geometry'] = EUBUCCO1['geometry'].apply(wkt.loads)
    gdf1 = gpd.GeoDataFrame(EUBUCCO1, crs='epsg:3035')
    
    EUBUCCO2 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-BEL_2.csv")
    EUBUCCO2['geometry'] = EUBUCCO2['geometry'].apply(wkt.loads)
    gdf2 = gpd.GeoDataFrame(EUBUCCO2, crs='epsg:3035')
    
    EUBUCCO3 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-BEL_3.csv")
    EUBUCCO3['geometry'] = EUBUCCO3['geometry'].apply(wkt.loads)
    gdf3 = gpd.GeoDataFrame(EUBUCCO3, crs='epsg:3035')
    
    gdf = pd.concat([gdf1, gdf2], ignore_index=True)
    gdf = pd.concat([gdf, gdf3], ignore_index=True)
    
elif country == "ESP":
    
    EUBUCCO1 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_1.csv")
    EUBUCCO1['geometry'] = EUBUCCO1['geometry'].apply(wkt.loads)
    gdf1 = gpd.GeoDataFrame(EUBUCCO1, crs='epsg:3035')
    
    EUBUCCO2 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_2.csv")
    EUBUCCO2['geometry'] = EUBUCCO2['geometry'].apply(wkt.loads)
    gdf2 = gpd.GeoDataFrame(EUBUCCO2, crs='epsg:3035')
    
    EUBUCCO3 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_3.csv")
    EUBUCCO3['geometry'] = EUBUCCO3['geometry'].apply(wkt.loads)
    gdf3 = gpd.GeoDataFrame(EUBUCCO3, crs='epsg:3035')
    
    EUBUCCO4 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_4.csv")
    EUBUCCO4['geometry'] = EUBUCCO4['geometry'].apply(wkt.loads)
    gdf4 = gpd.GeoDataFrame(EUBUCCO4, crs='epsg:3035')
    
    EUBUCCO5 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_5.csv")
    EUBUCCO5['geometry'] = EUBUCCO5['geometry'].apply(wkt.loads)
    gdf5 = gpd.GeoDataFrame(EUBUCCO5, crs='epsg:3035')
    
    EUBUCCO6 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_6.csv")
    EUBUCCO6['geometry'] = EUBUCCO6['geometry'].apply(wkt.loads)
    gdf6 = gpd.GeoDataFrame(EUBUCCO6, crs='epsg:3035')
    
    EUBUCCO7 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_7.csv")
    EUBUCCO7['geometry'] = EUBUCCO7['geometry'].apply(wkt.loads)
    gdf7 = gpd.GeoDataFrame(EUBUCCO7, crs='epsg:3035')
    
    EUBUCCO8 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_8.csv")
    EUBUCCO8['geometry'] = EUBUCCO8['geometry'].apply(wkt.loads)
    gdf8 = gpd.GeoDataFrame(EUBUCCO8, crs='epsg:3035')
    
    EUBUCCO9 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_9.csv")
    EUBUCCO9['geometry'] = EUBUCCO9['geometry'].apply(wkt.loads)
    gdf9 = gpd.GeoDataFrame(EUBUCCO9, crs='epsg:3035')
    
    EUBUCCO10 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_10.csv")
    EUBUCCO10['geometry'] = EUBUCCO10['geometry'].apply(wkt.loads)
    gdf10 = gpd.GeoDataFrame(EUBUCCO10, crs='epsg:3035')
    
    EUBUCCO11 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_11.csv")
    EUBUCCO11['geometry'] = EUBUCCO11['geometry'].apply(wkt.loads)
    gdf11 = gpd.GeoDataFrame(EUBUCCO11, crs='epsg:3035')
    
    EUBUCCO12 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_12.csv")
    EUBUCCO12['geometry'] = EUBUCCO12['geometry'].apply(wkt.loads)
    gdf12 = gpd.GeoDataFrame(EUBUCCO12, crs='epsg:3035')
    
    EUBUCCO13= pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_13.csv")
    EUBUCCO13['geometry'] = EUBUCCO13['geometry'].apply(wkt.loads)
    gdf13 = gpd.GeoDataFrame(EUBUCCO13, crs='epsg:3035')
    
    EUBUCCO14 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_14.csv")
    EUBUCCO14['geometry'] = EUBUCCO14['geometry'].apply(wkt.loads)
    gdf14 = gpd.GeoDataFrame(EUBUCCO14, crs='epsg:3035')
    
    EUBUCCO15 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_15.csv")
    EUBUCCO15['geometry'] = EUBUCCO15['geometry'].apply(wkt.loads)
    gdf15 = gpd.GeoDataFrame(EUBUCCO15, crs='epsg:3035')
    
    EUBUCCO16 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_16.csv")
    EUBUCCO16['geometry'] = EUBUCCO16['geometry'].apply(wkt.loads)
    gdf16 = gpd.GeoDataFrame(EUBUCCO16, crs='epsg:3035')
    
    EUBUCCO17 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_17.csv")
    EUBUCCO17['geometry'] = EUBUCCO17['geometry'].apply(wkt.loads)
    gdf17 = gpd.GeoDataFrame(EUBUCCO17, crs='epsg:3035')
    
    EUBUCCO18 = pd.read_csv(path_folder + "EUBUCCO/Data/EUBUCCO_COUNTRY/v0_1-ESP_18.csv")
    EUBUCCO18['geometry'] = EUBUCCO18['geometry'].apply(wkt.loads)
    gdf18 = gpd.GeoDataFrame(EUBUCCO18, crs='epsg:3035')
    
    gdf = pd.concat([gdf1, gdf2, gdf3, gdf4, gdf5, gdf6, gdf7, gdf8, gdf9, gdf10, gdf11, gdf12, gdf13, gdf14, gdf15, gdf16, gdf17, gdf18], ignore_index=True)


### COUNTRY LEVEL

#1b. Geometry quality.
#Are small buidings included or not?
#Do buidings tend to be merged or not?
#Are geometries precise or not?


gdf["buildings_area"] = gdf.area
gdf["volume"] = gdf["buildings_area"] * gdf["height"]

gdf["buildings_area"].describe()
gdf["height"].describe()
gdf["volume"].describe()

plt.hist(gdf["buildings_area"][gdf["buildings_area"]<1000], bins = 50)
plt.hist(gdf["height"][gdf["height"] < 25], bins = 50)
plt.hist(gdf["volume"][gdf["volume"]<20000], bins = 50)
plt.close()

gdf["buildings_area"][gdf["buildings_area"]<1000].plot(kind='density')
gdf["height"][gdf["height"] < 25].plot(kind='density')

gdf.loc[[4500],'geometry'].plot()
plt.savefig(path_folder + "EUBUCCO/Sorties/" + country + "_example.png")
plt.close()

#1c. Attribute quality
unique_height = len(np.unique(gdf.height))
unique_age = len(np.unique(gdf.age))
unique_type = len(np.unique(gdf["type"].astype(str))) #fit_transform(gdf["type"].astype(str))

#2a. Outlier assessment
 
over_320_height = sum(gdf["height"] > 320) / len(gdf)
over_3000_area = sum(gdf["buildings_area"] > 3000) / len(gdf)
over_100000_area = sum(gdf["buildings_area"] > 100000) / len(gdf)
under_2_height = sum(gdf["height"] < 2) / len(gdf)
under_4_area = sum(gdf["buildings_area"] < 4) / len(gdf)

#gdf.iloc[80882]['geometry']
#Differdange = gdf.loc[gdf.id_source.str.startswith("Differdange"),:]
#fig, ax = plt.subplots(figsize=(15, 15))
#Differdange.plot(ax=ax, alpha=0.7, color="grey") #ARCELOR-MITTAL

#gdf.iloc[88159]['geometry']
#Esch = gdf.loc[gdf.id_source.str.startswith("Esch-sur-Alzette"),:]
#fig, ax = plt.subplots(figsize=(15, 15))
#Esch.plot(ax=ax, alpha=0.7, color="grey") #ARCELOR-MITTAL

#2b. Attribute distribution assessment

def outlier_aware_hist(data, param_bins, lower=None, upper=None, filename = None):
    if not lower:# or lower < data.min():
        lower = data.min()
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = plt.hist(data, range=(lower, upper), bins=param_bins)

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('c')
        patches[0].set_label('Lower outliers: ({:.2f}, {:.2f})'.format(data.min(), lower))

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('m')
        patches[-1].set_label('Upper outliers: ({:.2f}, {:.2f})'.format(upper, data.max()))

    if lower_outliers or upper_outliers:
        plt.legend()
        
    plt.savefig(filename)
    plt.close()
        
outlier_aware_hist(gdf["height"], 50, lower=2, upper=25, filename = path_folder + "EUBUCCO/Sorties/" + country + "_height.png")
outlier_aware_hist(gdf["buildings_area"], 50, lower=4, upper=1000, filename = path_folder + "EUBUCCO/Sorties/" + country + "_area.png")
outlier_aware_hist(gdf["volume"], 50, lower=8, upper=20000, filename = path_folder + "EUBUCCO/Sorties/" + country + "_volume.png")
outlier_aware_hist(gdf["height"], 50, lower=2, upper=25, filename = path_folder + "EUBUCCO/Sorties/" + country + "_height.png")

thisdict = {
  "unique_height": unique_height,
  "unique_age": unique_age,
  "unique_type": unique_type,
  "over_320_height": over_320_height,
  "over_3000_area": over_3000_area,
  "over_100000_area": over_100000_area,
  "under_2_height": under_2_height,
  "under_4_area": under_4_area
  
}

w = csv.writer(open(path_folder + "EUBUCCO/Sorties/" + country + ".csv", "w"))
    
with open(path_folder + "EUBUCCO/Sorties/" + country + ".csv", 'w') as f:
    w = csv.DictWriter(f, thisdict.keys())
    w.writeheader()
    w.writerow(thisdict)


