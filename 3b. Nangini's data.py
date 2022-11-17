# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:45:32 2022

@author: charl
"""

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


path_folder = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/"

FUA = gpd.read_file("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FUA = FUA.to_crs('epsg:3035')

nangini = pd.read_excel(path_folder + 'Data/Nangini/Scripts_and_datafiles/SCRIPTS/DATA/' + "D_FINAL.xlsx")
nangini_gdf = gpd.GeoDataFrame(
    nangini, geometry=gpd.points_from_xy(nangini["Longitude (others) [degrees]"], nangini["Latitude (others) [degrees]"]))

nangini_gdf = nangini_gdf.set_crs('epsg:4326')
nangini_gdf = nangini_gdf.to_crs('epsg:3035')
nangini_gdf = nangini_gdf.loc[nangini_gdf.Region == "Europe",:]

nangini_with_FUA = sjoin(nangini_gdf, FUA, how="right")
nangini_with_FUA = nangini_with_FUA.loc[~np.isnan(nangini_with_FUA.index_left), :]

nangini_with_FUA = nangini_with_FUA.loc[:,['City name', 'Scope-1 GHG emissions [tCO2 or tCO2-eq]', 'Scope-2 (CDP) [tCO2-eq]', 'Total emissions (CDP) [tCO2-eq]',  'Population (others)',  'Population (CDP)', 'eFUA_ID', 'eFUA_name', 'FUA_p_2015']]
duplicate_FUA = set([x for x in nangini_with_FUA.eFUA_ID if list(nangini_with_FUA.eFUA_ID).count(x) > 1])

for id_dup in duplicate_FUA:
    print(id_dup)
    subset_dup = nangini_with_FUA.loc[nangini_with_FUA.eFUA_ID == id_dup,:]
    nangini_with_FUA = nangini_with_FUA.loc[nangini_with_FUA.eFUA_ID != id_dup,:]
    new_row = {'City name':subset_dup['City name'].iloc[0],
               'Scope-1 GHG emissions [tCO2 or tCO2-eq]':sum(subset_dup['Scope-1 GHG emissions [tCO2 or tCO2-eq]']),
               'Scope-2 (CDP) [tCO2-eq]':sum(subset_dup['Scope-2 (CDP) [tCO2-eq]']),
               'Total emissions (CDP) [tCO2-eq]':sum(subset_dup['Total emissions (CDP) [tCO2-eq]']),
               'Population (others)':sum(subset_dup['Population (others)']),
               'Population (CDP)':sum(subset_dup['Population (CDP)']),
               'eFUA_ID':subset_dup['eFUA_ID'].iloc[0],
               'eFUA_name':subset_dup['eFUA_name'].iloc[0],
               'FUA_p_2015':subset_dup['FUA_p_2015'].iloc[0]}
    nangini_with_FUA = nangini_with_FUA.append(new_row, ignore_index=True)
  
    
nangini_with_FUA.to_excel(path_folder+'Data/nangini_FUA.xlsx')


#### COMPARISON WITH MORAN

moran = pd.read_excel(path_folder+'Data/moran_aggregated/'+'all_countries'+'.xlsx')

moran_and_nangini = nangini_with_FUA.merge(moran, on = "eFUA_ID")
moran_and_nangini = moran_and_nangini.loc[(moran_and_nangini["City name"] != "Barreiro") & (moran_and_nangini["City name"] != "Heidelberg"),:]

x = moran_and_nangini["Population (CDP)"]
y = moran_and_nangini["Est. Population"]
subset = ~np.isnan(x)
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
 
x = moran_and_nangini["Total emissions (CDP) [tCO2-eq]"]
y = moran_and_nangini["Total (t CO2)"]
subset = ~np.isnan(x)
fig, ax = plt.subplots()
ax.scatter(x[subset], y[subset])
ax.set_yscale('log')
ax.set_xscale('log')
for i, txt in enumerate(moran_and_nangini["City name"][subset]):
    print(i)
    print(txt)
    ax.annotate(txt, (x[subset].iloc[i], y[subset].iloc[i]))
    
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
       
x = moran_and_nangini["Scope-1 GHG emissions [tCO2 or tCO2-eq]"] / moran_and_nangini["Population (CDP)"]
y = moran_and_nangini["Total (t CO2)"] / moran_and_nangini["Est. Population"]
subset = ((~np.isnan(x)) & (~np.isnan(y)))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
 
x = moran_and_nangini["Total emissions (CDP) [tCO2-eq]"] / moran_and_nangini["Population (CDP)"]
y = moran_and_nangini["Total (t CO2)"] / moran_and_nangini["Est. Population"]
subset = ((~np.isnan(x)) & (~np.isnan(y)))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
 


