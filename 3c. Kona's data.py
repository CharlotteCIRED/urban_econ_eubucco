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

kona = pd.read_excel(path_folder + "Data/GlobalCoM.xlsx", sheet_name = "Table 2")
kona_ancillary = pd.read_excel(path_folder + "Data/GlobalCoM.xlsx", sheet_name = "Table 3")
kona = kona.merge(kona_ancillary, on = "GCoM_ID")

kona_gdf = gpd.GeoDataFrame(
    kona, geometry=gpd.points_from_xy(kona["longitude"], kona["latitude"]))

kona_gdf = kona_gdf.set_crs('epsg:4326')
kona_gdf = kona_gdf.to_crs('epsg:3035')

kona_with_FUA = sjoin(kona_gdf, FUA, how="right")
kona_with_FUA = kona_with_FUA.loc[~np.isnan(kona_with_FUA.index_left), :]

kona_with_FUA = kona_with_FUA.loc[:,['GCoM_ID', 'emission_inventory_id', 'emission_inventory_sector', 'type_of_emissions', 'inventory_year', 'population_in_the_inventory_year', 'emissions', 'eFUA_ID', 'eFUA_name', 'FUA_p_2015', 'geometry']]
kona_with_FUA = kona_with_FUA.pivot(index=['GCoM_ID', 'inventory_year'], columns=['emission_inventory_sector', 'type_of_emissions'], values='emissions').reset_index()

a = kona_with_FUA.columns
ind = pd.Index([e[0] + e[1] for e in a.tolist()])
kona_with_FUA.columns = ind
kona_with_FUA = kona_with_FUA.reset_index()

duplicate_kona = set([x for x in kona_with_FUA.GCoM_ID if list(kona_with_FUA.GCoM_ID).count(x) > 1])

for id_dup in duplicate_kona:
    print(id_dup)
    subset_dup = kona_with_FUA.loc[kona_with_FUA.GCoM_ID == id_dup,:]
    max_year = max(subset_dup.inventory_year)
    kona_with_FUA = kona_with_FUA.loc[(kona_with_FUA.GCoM_ID != id_dup) | ((kona_with_FUA.GCoM_ID == id_dup) & (kona_with_FUA.inventory_year == max_year)),:]
    
kona_with_FUA = kona_with_FUA.merge(kona_gdf.loc[:,['GCoM_ID', 'geometry', 'population_in_the_inventory_year','inventory_year']].drop_duplicates(), on = ["GCoM_ID",'inventory_year'] , how = 'left')
kona_with_FUA = gpd.GeoDataFrame(
    kona_with_FUA, geometry=kona_with_FUA['geometry'])

kona_with_FUA = sjoin(kona_with_FUA, FUA, how="right")
kona_with_FUA = kona_with_FUA.loc[~np.isnan(kona_with_FUA.index_left), :]
kona_with_FUA = kona_with_FUA.merge(kona_ancillary.loc[:,['GCoM_ID', 'signatory name']], on = "GCoM_ID")

kona_with_FUA = kona_with_FUA.loc[:,['GCoM_ID', 'inventory_year',
       'Municipal buildings and facilitiesindirect_emissions',
       'Institutional/tertiary buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesindirect_emissions',
       'Transportationdirect_emissions',
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Transportationindirect_emissions', 'Waste/wastewaterdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions',
       'Manufacturing and construction industriesdirect_emissions',
       'Manufacturing and construction industriesindirect_emissions',
       'population_in_the_inventory_year', 'eFUA_ID',
       'eFUA_name', 'FUA_p_2015', 'geometry',
       'signatory name']]

kona_with_FUA = kona_with_FUA.groupby('eFUA_ID').agg({'Municipal buildings and facilitiesindirect_emissions':'sum',
                                      'Institutional/tertiary buildings and facilitiesindirect_emissions':'sum', 
                                      'Residential buildings and facilitiesindirect_emissions':'sum', 
                                      'Transportationdirect_emissions':'sum', 
                                      'Residential buildings and facilitiesdirect_emissions':'sum', 
                                      'Municipal buildings and facilitiesdirect_emissions':'sum', 
                                      'Transportationindirect_emissions':'sum', 
                                      'Waste/wastewaterdirect_emissions':'sum', 
                                      'Institutional/tertiary buildings and facilitiesdirect_emissions':'sum', 
                                      'Manufacturing and construction industriesdirect_emissions':'sum', 
                                      'Manufacturing and construction industriesindirect_emissions':'sum',
                                      'population_in_the_inventory_year':'sum',
                                      'eFUA_name':'first', 
                                      'FUA_p_2015':'first', 'geometry':'first'})


kona_with_FUA.to_excel(path_folder+'Data/kona_FUA.xlsx')

####

kona_with_FUA["emissions"] = np.nansum(kona_with_FUA.loc[:,[
       'Municipal buildings and facilitiesindirect_emissions',
       'Institutional/tertiary buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesindirect_emissions',
       'Transportationdirect_emissions',
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Transportationindirect_emissions', 'Waste/wastewaterdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions',
       'Manufacturing and construction industriesdirect_emissions',
       'Manufacturing and construction industriesindirect_emissions']], 1)

kona_with_FUA["buildings_kona"] = np.nansum(kona_with_FUA.loc[:,[
       'Municipal buildings and facilitiesindirect_emissions',
       'Institutional/tertiary buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions']], 1)

kona_with_FUA["transport_kona"] = np.nansum(kona_with_FUA.loc[:,[
       'Transportationdirect_emissions',
       'Transportationindirect_emissions']], 1)


kona_with_FUA["emissions_per_c"] = kona_with_FUA["emissions"] / kona_with_FUA["population_in_the_inventory_year"]

x = kona_with_FUA["population_in_the_inventory_year"]
y = kona_with_FUA["FUA_p_2015"]
subset = ~np.isnan(x)
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
 
x = kona_with_FUA["population_in_the_inventory_year"]
y = kona_with_FUA["emissions"]
subset = ~np.isnan(x)
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
 
####☺

moran = pd.read_excel(path_folder+'Data/moran_aggregated/'+'all_countries'+'.xlsx')
kona_moran = kona_with_FUA.merge(moran, right_on = "eFUA_ID", left_index = True)

x = kona_moran["emissions"]
y = kona_moran["Total (t CO2)"]
subset = ~np.isnan(x)
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_moran["emissions_per_c"]
y = kona_moran["Total (t CO2)"] / kona_moran['Est. Population']
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_moran["buildings_kona"]
y = kona_moran["buildings"]
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_moran["transport_kona"]
y = kona_moran["fuelstations"]
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_moran["buildings_kona"]/ kona_moran["population_in_the_inventory_year"]
y = kona_moran["buildings"] / kona_moran['Est. Population']
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_moran["transport_kona"]/ kona_moran["population_in_the_inventory_year"]
y = kona_moran["fuelstations"] / kona_moran['Est. Population']
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

###

nangini = pd.read_excel(path_folder+'Data/nangini_FUA.xlsx')
kona_nangini = kona_with_FUA.merge(nangini, right_on = "eFUA_ID", left_index = True)

x = kona_nangini["emissions"]
y = kona_nangini["Scope-1 GHG emissions [tCO2 or tCO2-eq]"]
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])

x = kona_nangini["emissions"]
y = kona_nangini["Total emissions (CDP) [tCO2-eq]"]
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])


x = kona_nangini["emissions_per_c"]
y = kona_nangini["Total emissions (CDP) [tCO2-eq]"] / kona_nangini['Population (CDP)']
subset = (~np.isnan(x)) & (~np.isnan(y))
plt.scatter(x[subset], y[subset])
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[subset], y[subset])
