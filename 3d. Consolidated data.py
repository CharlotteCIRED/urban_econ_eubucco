# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:19:50 2022

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

nangini = pd.read_excel(path_folder+'Data/nangini_FUA.xlsx')
moran = pd.read_excel(path_folder+'Data/moran_aggregated/'+'all_countries'+'.xlsx')
kona = pd.read_excel(path_folder+'Data/kona_FUA.xlsx')

nangini = nangini.loc[:,['City name', 'Scope-1 GHG emissions [tCO2 or tCO2-eq]',
       'Total emissions (CDP) [tCO2-eq]',
       'Population (others)', 'Population (CDP)', 'eFUA_ID', 'eFUA_name',
       'FUA_p_2015']]


nangini.columns = ['city_name_nangini', 'scope1_nangini', 'scope2_nangini', 'population2_nangini', 'population1_nangini', 'eFUA_ID', 'eFUA_name', 'FUA_p_2015']

moran = moran.loc[:,['eFUA_ID', 'Est. Population', 'Total (t CO2)',
       'buildings', 'fuelstations']]

moran.columns = ['eFUA_ID', 'population_moran', 'scope1_moran',
       'buildings_moran', 'transport_moran']

#Kona: direct and indirect?

kona["emissions_kona"] = np.nansum(kona.loc[:,[
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

kona["buildings_kona"] = np.nansum(kona.loc[:,[
       'Municipal buildings and facilitiesindirect_emissions',
       'Institutional/tertiary buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesindirect_emissions',
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions']], 1)

kona["transport_kona"] = np.nansum(kona.loc[:,[
       'Transportationdirect_emissions',
       'Transportationindirect_emissions']], 1)

kona["emissions_scope1_kona"] = np.nansum(kona.loc[:,[
       'Transportationdirect_emissions',
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Waste/wastewaterdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions',
       'Manufacturing and construction industriesdirect_emissions']], 1)

kona["buildings_scope1_kona"] = np.nansum(kona.loc[:,[
       'Residential buildings and facilitiesdirect_emissions',
       'Municipal buildings and facilitiesdirect_emissions',
       'Institutional/tertiary buildings and facilitiesdirect_emissions']], 1)

kona["transport_scope1_kona"] = np.nansum(kona.loc[:,[
       'Transportationdirect_emissions']], 1)

kona = kona.loc[:,['eFUA_ID','population_in_the_inventory_year','emissions_kona', 'buildings_kona', 'transport_kona', 'emissions_scope1_kona', 'buildings_scope1_kona', 'transport_scope1_kona']]
kona.columns = ['eFUA_ID','population_kona','emissions_kona', 'buildings_kona', 'transport_kona', 'emissions_scope1_kona', 'buildings_scope1_kona', 'transport_scope1_kona']

emissions_data = kona.merge(moran, on = 'eFUA_ID', how = "outer")
emissions_data = emissions_data.merge(nangini, on = 'eFUA_ID', how = "outer")

emissions_data.to_excel(path_folder+'Data/emissions_data.xlsx')

####
sum(~np.isnan(emissions_data.emissions_kona))
sum(~np.isnan(emissions_data.scope3_moran))
sum(~np.isnan(emissions_data.scope3_nangini))


correlation_population = emissions_data.loc[:,['population_kona','population_moran','population2_nangini', 'population1_nangini',
'FUA_p_2015']].corr()

correlation_emissions = emissions_data.loc[:,['emissions_kona', 'buildings_kona',
       'transport_kona', 'emissions_scope1_kona', 'buildings_scope1_kona',
       'transport_scope1_kona', 'scope1_moran',
       'buildings_moran', 'transport_moran',
       'scope1_nangini', 'scope2_nangini']].corr()

emissions_data["emissions_pc_kona"] = emissions_data["emissions_kona"] / emissions_data["population_kona"]
emissions_data["emissions_pc_moran"] = emissions_data["scope1_moran"] / emissions_data["population_moran"]
emissions_data["emissions_pc_nangini"] = emissions_data["scope2_nangini"] / emissions_data["population1_nangini"]
emissions_data["emissions_pc_kona_scope1"] = emissions_data["emissions_scope1_kona"] / emissions_data["population_kona"]
emissions_data["emissions_pc_nangini_scope1"] = emissions_data["scope1_nangini"] / emissions_data["population1_nangini"]
emissions_data["emissions_pc_moran_transport"] = emissions_data["transport_moran"] / emissions_data["population_moran"]
emissions_data["emissions_pc_moran_buildings"] = emissions_data["buildings_moran"] / emissions_data["population_moran"]
emissions_data["emissions_pc_kona_transport"] = emissions_data["transport_kona"] / emissions_data["population_kona"]
emissions_data["emissions_pc_kona_buildings"] = emissions_data["buildings_kona"] / emissions_data["population_kona"]


emissions_data["emissions_pc_kona"] = emissions_data["emissions_kona"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_moran"] = emissions_data["scope1_moran"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_nangini"] = emissions_data["scope2_nangini"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_kona_scope1"] = emissions_data["emissions_scope1_kona"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_nangini_scope1"] = emissions_data["scope1_nangini"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_moran_transport"] = emissions_data["transport_moran"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_moran_buildings"] = emissions_data["buildings_moran"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_kona_transport"] = emissions_data["transport_kona"] / emissions_data["FUA_p_2015"]
emissions_data["emissions_pc_kona_buildings"] = emissions_data["buildings_kona"] / emissions_data["FUA_p_2015"]



correlation_emissions_pc = emissions_data.loc[:,['emissions_pc_kona', 'emissions_pc_moran',
       'emissions_pc_nangini','emissions_pc_kona_scope1', 'emissions_pc_nangini_scope1',
       'emissions_pc_moran_transport', 'emissions_pc_moran_buildings',
       'emissions_pc_kona_transport', 'emissions_pc_kona_buildings']].corr()
