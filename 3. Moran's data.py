# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:09:19 2022

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

### Spatial weight?
### Duplicate FUA?

path_folder = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/"

FUA = gpd.read_file("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FUA = FUA.to_crs('epsg:3035')

moran = pd.read_excel(path_folder + "Data/Moran/allcountries_onlycities_summary.xlsx")
moran2 = pd.read_excel(path_folder + "Data/Moran/allcountries_summary.xlsx")

list_countries = pd.read_excel(path_folder + "Data/Moran/moran_admin_level.xlsx")

for country in list_countries.moran_name:
    print(country)
    country = str(country)
    admin_level = str(list_countries.admin_level.loc[list_countries.moran_name == country].squeeze())
    moran3= gpd.read_file(path_folder + "Data/Moran/allcountries.geojson/data/"+country+"/" + admin_level + ".geojson")
    moran3_with_data=moran3.merge(moran.loc[(moran.Country == country) &( moran.admin_level ==int(admin_level)),:], left_on = "rname", right_on = "Region Name")
    moran3_with_data = moran3_with_data.drop_duplicates(subset = "Region Name")
    moran3_with_data = moran3_with_data.to_crs('epsg:3035')

    moran_with_FUA = sjoin(FUA, moran3_with_data, how="right")
    moran_with_FUA = moran_with_FUA.loc[~np.isnan(moran_with_FUA.eFUA_ID), :]
    emissions_per_FUA = moran_with_FUA.loc[:,['Est. Population', 'Total (t CO2)', 'buildings', 'fuelstations', 'eFUA_ID']].groupby('eFUA_ID').agg('sum')
    emissions_per_FUA = emissions_per_FUA.merge(FUA.loc[:,['FUA_p_2015', 'eFUA_ID','eFUA_name', 'Cntry_name']], on = "eFUA_ID")
    emissions_per_FUA.to_excel(path_folder+'Data/moran_aggregated/'+country+'.xlsx')
    
### FUA over multiple countries

df = pd.read_excel(path_folder+'Data/moran_aggregated/'+ 'austria'+'.xlsx', index_col = 0)
for country in list_countries.moran_name[1:]:
    country= str(country)
    df2 = pd.read_excel(path_folder+'Data/moran_aggregated/'+ country+'.xlsx', index_col = 0)
    df = df.append(df2)

duplicate_FUA = set([x for x in df.eFUA_ID if list(df.eFUA_ID).count(x) > 1])
for id_dup in duplicate_FUA:
    print(id_dup)
    subset_dup = df.loc[df.eFUA_ID == id_dup,:]
    df = df.loc[df.eFUA_ID != id_dup,:]
    new_row = {'eFUA_ID':subset_dup.eFUA_ID.iloc[0], 'Est. Population':np.nansum(subset_dup["Est. Population"]), 'Total (t CO2)':np.nansum(subset_dup["Total (t CO2)"]), 'buildings':np.nansum(subset_dup["buildings"]),'fuelstations':np.nansum(subset_dup["fuelstations"]), 'FUA_p_2015':subset_dup.FUA_p_2015.iloc[0], 'eFUA_name':subset_dup.eFUA_name.iloc[0], 'Cntry_name':subset_dup.Cntry_name.iloc[0]}
    df = df.append(new_row, ignore_index=True)
  
      
df.to_excel(path_folder+'Data/moran_aggregated/'+'all_countries'+'.xlsx')


### FUA and Moran's population don't match?

plt.scatter(df["Est. Population"], df["FUA_p_2015"])
plt.xlim(0,16000000)
plt.ylim(0,16000000)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["FUA_p_2015"], df["Est. Population"].astype(float))


plt.scatter(df["Est. Population"],df["Total (t CO2)"])
plt.scatter(df["Est. Population"],df["buildings"])
plt.scatter(df["Est. Population"],df["fuelstations"])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["Est. Population"], df["Total (t CO2)"].astype(float))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["Est. Population"], df["buildings"].astype(float))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["Est. Population"], df["fuelstations"].astype(float))


plt.scatter(df["Total (t CO2)"], df["buildings"])
plt.scatter(df["Total (t CO2)"], df["fuelstations"])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["Total (t CO2)"].astype(float), df["buildings"].astype(float))
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df["Total (t CO2)"].astype(float), df["fuelstations"].astype(float))
