# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:20:33 2022

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
from os import listdir
import time

folder_FUA = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/FUA/"

FUA_data = gpd.read_file("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg")
FUA_data = FUA_data.to_crs('epsg:3035')

results = pd.DataFrame(index = listdir(folder_FUA), columns = ['n_buildings', 'n_pop', 'UA', 'BA', 'BV', 'density_b', 'avg_height', 'vol_per_cap', 'centrality1', 'centrality2', 'volume_profile', 'footprint_profile', 'agglo_spag', 'vol_around', 'area_around'])

for FUA in listdir(folder_FUA)[84:]:
    
    print(FUA)
    import time
    start = time.time()
    
    
    gdf = pd.read_csv(folder_FUA + FUA)
    gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf, crs='epsg:3035')
    
    ### 1. DESCRIPTIVE STATISTICS OVER THE DATABASE

    gdf["buildings_area"] = gdf.area

    nb_tot_buildings = len(gdf)


    ### 2. PHYSICAL CHARACTERISTICS OF THE BUILT ENVIRONMENT

    BA = np.nansum(gdf["buildings_area"])
    gdf["volume"] = gdf["buildings_area"] * gdf["height"]
    BV = np.nansum(gdf["volume"])

    N = gdf["FUA_p_2015"].iloc[0]
    UA = gdf["FUA_area"].iloc[0] * 1000000
    density_buildings = BA/UA
    avg_height = BV/BA
    volume_per_capita = BV/N
    
    ### SAVE
    
    results.loc[results.index == FUA, 'n_buildings'] = nb_tot_buildings
    results.loc[results.index == FUA, 'n_pop'] = N
    results.loc[results.index == FUA, 'UA'] = UA
    results.loc[results.index == FUA, 'BA'] = BA
    results.loc[results.index == FUA, 'BV'] = BV
    results.loc[results.index == FUA, 'density_b'] = density_buildings
    results.loc[results.index == FUA, 'avg_height'] = avg_height
    results.loc[results.index == FUA, 'vol_per_cap'] = volume_per_capita

    step2 = time.time()
    
    ### 3. CENTRALITY/MONOCENTRICITY
    
    FUA_name = gdf.eFUA_name.iloc[0]
    GHS_cent = gpd.read_file("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Data/GHS_STAT_UCDB2015MT_GLOBE_R2019A/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg")
    GHS_cent = GHS_cent.loc[GHS_cent.UC_NM_MN == FUA_name,:]
    
    GHS_cent = GHS_cent.to_crs('epsg:3035')

    gdf_center = sjoin(gdf.drop(columns = "index_right"), GHS_cent, how='left')
    gdf_center = gdf_center.loc[~np.isnan(gdf_center.index_right),:]
    
    gdf_centroid = copy.deepcopy(gdf_center)
    gdf_centroid["geometry"] = gdf_centroid.centroid
    
    cx = np.average(gdf_centroid["geometry"].x[~np.isnan(gdf_centroid["volume"])], weights = gdf_centroid["volume"][~np.isnan(gdf_centroid["volume"])])
    cy = np.average(gdf_centroid["geometry"].y[~np.isnan(gdf_centroid["volume"])], weights = gdf_centroid["volume"][~np.isnan(gdf_centroid["volume"])])
    center_gdf2 = gpd.GeoSeries((Point(cx, cy)))
    
    #centrality_index
    gdf["distance_center"] = gdf.distance(center_gdf2.geometry[0])
    avg_dist = np.average(a = gdf["distance_center"][~np.isnan(gdf["volume"])], weights = gdf["volume"][~np.isnan(gdf["volume"])])
    R1 = np.sqrt(np.nansum(gdf["buildings_area"])/np.pi)
    R2 = np.sqrt(UA/np.pi)

    centrality_index1 = avg_dist/R1
    centrality_index2 = avg_dist/R2
    
    ### SAVE
    
    results.loc[results.index == FUA, 'centrality1'] = centrality_index1
    results.loc[results.index == FUA, 'centrality2'] = centrality_index2

    step3 = time.time()
    
    #fig, ax = plt.subplots(figsize=(15, 15))
    #GHS_cent.plot(ax=ax, alpha=0.7, color="pink")
    #gdf.plot(ax=ax)
    #center_gdf2.plot(ax = ax, color = "red")

    #buffer_compactness_index
    #max_dist = math.ceil(np.nanmax(gdf["distance_center"])/1000)
    #buffer_df = pd.DataFrame(columns = ["dist_center", "share_volume"], index = np.arange(1, max_dist+1))
    #buffer_df.dist_center = buffer_df.index
    #for i in buffer_df.index:
    #    print(i)
    #    buffer = center_gdf2.buffer(i*1000)
    #    inter_buffer = sjoin(gdf.drop(columns = "index_right"), gpd.GeoDataFrame(geometry = buffer), how='left')
    #    inter_buffer = inter_buffer.loc[~np.isnan(inter_buffer.index_right),:]
    #    buffer_df.share_volume.loc[buffer_df.dist_center == i] = np.nansum(inter_buffer.volume)
    
    #model = linear_model.LinearRegression()
    #results = model.fit(np.array(buffer_df.dist_center).reshape(-1, 1), buffer_df.share_volume)
    #bci = results.coef_

    #plt.plot(buffer_df.dist_center, buffer_df.share_volume)
    #plt.plot(buffer_df.dist_center, results.intercept_ + (results.coef_ * buffer_df.dist_center))

    #density profile

    max_dist = math.ceil(np.nanmax(gdf["distance_center"])/1000)
    buffer_df = pd.DataFrame(columns = ["dist_center", "volume", "footprint", "area_buffer"], index = np.arange(1, max_dist+1))
    buffer_df.dist_center = buffer_df.index
    for i in buffer_df.index:
        print(i)
        buffer = center_gdf2.buffer(i*1000)
        buffer_df.area_buffer.loc[buffer_df.dist_center == i] = buffer[0].intersection(FUA_data.loc[FUA_data.eFUA_name == FUA_name,:].geometry.iloc[0]).area
        if (FUA == 'Cartagena_v0_1-ESP_18.csv.csv') | (FUA == 'Ceuta_v0_1-ESP_7.csv.csv')| (FUA == 'Cáceres_v0_1-ESP_11.csv.csv')| (FUA == 'Córdoba_v0_1-ESP_1.csv.csv')| (FUA == 'Granada_v0_1-ESP_1.csv.csv')|(FUA == 'Guadalajara_v0_1-ESP_4.csv.csv')|(FUA == 'Jaén_v0_1-ESP_1.csv.csv')|(FUA == 'León_v0_1-ESP_5.csv.csv')|(FUA == 'Pamplona_v0_1-ESP_9.csv.csv')|(FUA == 'Salamanca_v0_1-ESP_5.csv.csv')|(FUA == 'Valencia_v0_1-ESP_10.csv.csv')|(FUA == 'Valladolid_v0_1-ESP_5.csv.csv'):
            buffer_df.area_buffer.loc[buffer_df.dist_center == i] = buffer[0].intersection(FUA_data.loc[(FUA_data.eFUA_name == FUA_name) & (FUA_data.Cntry_name == "Spain"),:].geometry.iloc[0]).area
            
        
        inter_buffer = sjoin(gdf.drop(columns = "index_right"), gpd.GeoDataFrame(geometry = buffer), how='inner')
        buffer_df.volume.loc[buffer_df.dist_center == i] = np.nansum(inter_buffer.volume)
        buffer_df.footprint.loc[buffer_df.dist_center == i] = np.nansum(inter_buffer.buildings_area)
    
    

    for i in (np.arange(max_dist, 1, -1)):
        buffer_df.volume.loc[buffer_df.dist_center == i] = buffer_df.volume.loc[ buffer_df.dist_center == i].squeeze() - buffer_df.volume.loc[ buffer_df.dist_center == i-1].squeeze()
        buffer_df.area_buffer.loc[buffer_df.dist_center == i] = buffer_df.area_buffer.loc[ buffer_df.dist_center == i].squeeze() - buffer_df.area_buffer.loc[ buffer_df.dist_center == i-1].squeeze()
        buffer_df.footprint.loc[buffer_df.dist_center == i] = buffer_df.footprint.loc[ buffer_df.dist_center == i].squeeze() - buffer_df.footprint.loc[ buffer_df.dist_center == i-1].squeeze()
    
    buffer_df["gradient_volume"] = buffer_df.volume / buffer_df.area_buffer
    buffer_df["gradient_footprint"] = buffer_df.footprint / buffer_df.area_buffer

    #plt.plot(buffer_df["gradient_volume"])
    #plt.plot(np.log(buffer_df["gradient_volume"].astype(float)))
    
    model = linear_model.LinearRegression()
    results_model = model.fit(np.array(buffer_df.dist_center).reshape(-1, 1)[buffer_df["gradient_volume"].astype(float) > 0], np.log(buffer_df["gradient_volume"].astype(float))[buffer_df["gradient_volume"].astype(float) > 0], buffer_df["gradient_volume"].astype(float)[buffer_df["gradient_volume"].astype(float) > 0])
    volume_profile = results_model.coef_

    #plt.plot(np.log(buffer_df["gradient_volume"].astype(float)))
    #plt.plot(buffer_df.dist_center, np.exp(results.intercept_ + (results.coef_ * buffer_df.dist_center)))


    #plt.plot(buffer_df["gradient_footprint"])
    #plt.plot(np.log(buffer_df["gradient_footprint"].astype(float)))

    model = linear_model.LinearRegression()
    results_model = model.fit(np.array(buffer_df.dist_center).reshape(-1, 1)[buffer_df["gradient_footprint"].astype(float) > 0], np.log(buffer_df["gradient_footprint"].astype(float))[buffer_df["gradient_footprint"].astype(float) > 0], (buffer_df["gradient_footprint"].astype(float))[buffer_df["gradient_footprint"].astype(float) > 0])
    footprint_profile = results_model.coef_

    #plt.plot(buffer_df.dist_center, (buffer_df["gradient_footprint"].astype(float)))
    #plt.plot(buffer_df.dist_center, np.exp(results.intercept_ + (results.coef_ * buffer_df.dist_center)))

    ### SAVE
    
    results.loc[results.index == FUA, 'volume_profile'] = volume_profile
    results.loc[results.index == FUA, 'footprint_profile'] = footprint_profile
    
    step4 = time.time()
    
    ### AGGLOMERATION

    gdf_centroid = copy.deepcopy(gdf)
    gdf_centroid["geometry"] = gdf_centroid.centroid
    #gdf_centroid.plot(markersize = 0.1)

    coef_radius = (UA.squeeze() / np.nansum(gdf_centroid["volume"]))

    gdf_centroid["areas_for_spag"] = gdf_centroid["volume"] * coef_radius
    gdf_centroid["radius_spag"] = np.sqrt(gdf_centroid["areas_for_spag"] / np.pi)
    
    gdf_centroid['geometry'] = gdf_centroid.buffer(gdf_centroid['radius_spag'], resolution=16)
    #gdf_centroid.plot()

    spag = gdf_centroid.dissolve()
    agglo_spag = spag.area.squeeze() / UA.squeeze()

    ### SAVE
    
    results.loc[results.index == FUA, 'agglo_spag'] = agglo_spag
    
    step5 = time.time()
    
    print("3D metrics", step2 - start)
    print("Centrality", step3 - step2)
    print("Density profiles", step4 - step3)
    print("SPAG", step5 - step4)
    
results.to_excel("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Sorties/urban_form_metrics.xlsx")
results = pd.read_excel("C:/Users/charl/OneDrive/Bureau/EUBUCCO/Sorties/urban_form_metrics.xlsx", index_col = 0)
    
    ### BUFFER
    gdf["volume_around"] = np.nan
    gdf["area_around"] = np.nan

    buffer_500 = copy.deepcopy(gdf)
    buffer_500["geometry"] = gdf.buffer(500, resolution=16)
    inters = sjoin(buffer_500.drop(columns = "index_right"), gdf.drop(columns = "index_right"))
    result = inters.loc[:, ["id_left", "volume_right", "buildings_area_right"]].groupby("id_left").sum()

    buffer_500=buffer_500.merge(result, left_on = "id", right_index= True)
    buffer_500["volume_around"] = buffer_500["volume_right"] - buffer_500["volume"] 
    buffer_500["area_around"] = buffer_500["buildings_area_right"] - buffer_500["buildings_area"] 

    vol_around = np.average(buffer_500["volume_around"], weights = buffer_500["volume"]) / BV
    area_around = np.average(buffer_500["area_around"], weights = buffer_500["volume"]) / BA
    
    ### SAVE
    
    results.loc[results.index == FUA, 'vol_around'] = vol_around
    results.loc[results.index == FUA, 'area_around'] = area_around
    
    step6 = time.time()
    
    print("3D metrics", step2 - start)
    print("Centrality", step3 - step2)
    print("Density profiles", step4 - step3)
    print("SPAG", step5 - step4)
    print("buffer", step6 - step5)
    
    
results = results.loc[~np.isnan(results.agglo_spag), ['n_buildings', 'n_pop', 'UA', 'BA', 'BV', 'density_b', 'avg_height',
       'vol_per_cap', 'centrality1', 'centrality2', 'volume_profile',
       'footprint_profile', 'agglo_spag', 'eFUA_ID']]

sum_results = results.describe()

results = results.loc[(results.n_buildings > 500) & (results.density_b > 0.01) & (results.avg_height>2) & (results.vol_per_cap > 30) & (results.centrality1 > 1), :  ]
   
sum_results = results.describe()

correlations = pd.DataFrame(np.corrcoef(np.transpose(results)), columns = results.columns, index = results.columns)

results_for_reg = results.loc[:,['n_pop', 'density_b', 'avg_height',
       'vol_per_cap', 'centrality1', 'centrality2', 'volume_profile',
       'footprint_profile', 'agglo_spag', 'eFUA_ID']]

path_folder = "C:/Users/charl/OneDrive/Bureau/EUBUCCO/"

emissions_data = pd.read_excel(path_folder+'Data/emissions_data.xlsx')

results_for_reg['City'] = results_for_reg.index
results_for_reg = results_for_reg.merge(emissions_data, on = 'eFUA_ID', how = 'inner')

import statsmodels.formula.api as smf

plt.scatter(results_for_reg.n_pop, results_for_reg.scope1_moran)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_moran)
plt.scatter(results_for_reg.n_pop, results_for_reg.buildings_moran)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_moran_buildings)
plt.scatter(results_for_reg.n_pop, results_for_reg.log_moran_transport)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_kona)
plt.scatter(results_for_reg.n_pop, results_for_reg.emissions_kona)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_kona)
plt.scatter(results_for_reg.n_pop, results_for_reg.buildings_kona)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_kona_buildings)
plt.scatter(results_for_reg.n_pop, results_for_reg.transport_kona)
plt.scatter(results_for_reg.log_n_pop, results_for_reg.log_kona_transport)

from sklearn import preprocessing

results_for_reg["log_moran"] = np.log(results_for_reg["scope1_moran"])
results_for_reg["log_moran_buildings"] = np.log(results_for_reg["buildings_moran"])
results_for_reg["log_moran_transport"] = np.log(results_for_reg["transport_moran"])
results_for_reg["log_kona"] = np.log(results_for_reg["emissions_kona"])
results_for_reg["log_n_pop"] = np.log(results_for_reg["n_pop"])
results_for_reg["log_kona_buildings"] = np.log(results_for_reg["buildings_kona"])
results_for_reg["log_kona_transport"] = np.log(results_for_reg["transport_kona"])

results_for_reg["log_density_b"] = np.log(results_for_reg["density_b"])
results_for_reg["log_avg_height"] = np.log(results_for_reg["avg_height"])
results_for_reg["log_vol_per_cap"] = np.log(results_for_reg["vol_per_cap"])
results_for_reg["log_centrality1"] = np.log(results_for_reg["centrality1"])
results_for_reg["log_agglo_spag"] = np.log(results_for_reg["agglo_spag"])
results_for_reg["log_volume_profile"] = np.log(np.abs(results_for_reg["volume_profile"]))

st_results = pd.DataFrame(preprocessing.scale(results_for_reg.loc[results_for_reg.transport_kona > 0,['log_moran', 'log_n_pop', 'log_kona', 'log_moran_buildings',
 'log_moran_transport', 'log_kona_buildings', 'log_kona_transport',
 'log_density_b', 'log_avg_height', 'log_vol_per_cap', 'log_centrality1',
 'log_agglo_spag', 'log_volume_profile']]), columns = ['log_moran', 'log_n_pop', 'log_kona', 'log_moran_buildings',
  'log_moran_transport', 'log_kona_buildings', 'log_kona_transport',
  'log_density_b', 'log_avg_height', 'log_vol_per_cap', 'log_centrality1',
  'log_agglo_spag', 'log_volume_profile'])

mod = smf.ols('log_moran ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('log_moran_transport ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('log_moran_buildings ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('log_kona ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('log_kona_buildings ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('log_kona_transport ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('scope1_nangini ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

mod = smf.ols('scope2_nangini ~ log_n_pop + log_density_b + log_avg_height + log_vol_per_cap + log_centrality1 + log_volume_profile + log_agglo_spag', data=st_results)
res = mod.fit()
print(res.summary())

results_for_reg["emissions_pc_kona"] = results_for_reg["emissions_kona"] / results_for_reg["population_kona"]
results_for_reg["emissions_pc_moran"] = results_for_reg["scope1_moran"] / results_for_reg["population_moran"]
results_for_reg["emissions_pc_nangini"] = results_for_reg["scope2_nangini"] / results_for_reg["population1_nangini"]
results_for_reg["emissions_pc_kona_scope1"] = results_for_reg["emissions_scope1_kona"] / results_for_reg["population_kona"]
results_for_reg["emissions_pc_nangini_scope1"] = results_for_reg["scope1_nangini"] / results_for_reg["population1_nangini"]
results_for_reg["emissions_pc_moran_transport"] = results_for_reg["transport_moran"] / results_for_reg["population_moran"]
results_for_reg["emissions_pc_moran_buildings"] = results_for_reg["buildings_moran"] / results_for_reg["population_moran"]
results_for_reg["emissions_pc_kona_transport"] = results_for_reg["transport_kona"] / results_for_reg["population_kona"]
results_for_reg["emissions_pc_kona_buildings"] = results_for_reg["buildings_kona"] / results_for_reg["population_kona"]

results_for_reg["emissions_pc_kona"] = results_for_reg["emissions_kona"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_moran"] = results_for_reg["scope1_moran"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_nangini"] = results_for_reg["scope2_nangini"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_kona_scope1"] = results_for_reg["emissions_scope1_kona"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_nangini_scope1"] = results_for_reg["scope1_nangini"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_moran_transport"] = results_for_reg["transport_moran"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_moran_buildings"] = results_for_reg["buildings_moran"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_kona_transport"] = results_for_reg["transport_kona"] / results_for_reg["n_pop"]
results_for_reg["emissions_pc_kona_buildings"] = results_for_reg["buildings_kona"] / results_for_reg["n_pop"]

var_emissions = "scope1_moran"

plt.scatter((results_for_reg.density_b), (results_for_reg[var_emissions]))
plt.scatter(np.log(results_for_reg.density_b), np.log(results_for_reg[var_emissions]))

plt.scatter((results_for_reg.avg_height), (results_for_reg[var_emissions]))
plt.scatter(np.log(results_for_reg.avg_height), np.log(results_for_reg[var_emissions]))

plt.scatter((results_for_reg.vol_per_cap), (results_for_reg[var_emissions]))
plt.scatter(np.log(results_for_reg.vol_per_cap), np.log(results_for_reg[var_emissions]))

plt.scatter((results_for_reg.centrality1), (results_for_reg[var_emissions]))
plt.scatter(np.log(results_for_reg.centrality1), np.log(results_for_reg[var_emissions]))

plt.scatter(np.abs(results_for_reg.volume_profile), (results_for_reg[var_emissions]))
plt.scatter(np.log(np.abs(results_for_reg.volume_profile)), np.log(results_for_reg[var_emissions]))

plt.scatter((results_for_reg.agglo_spag), (results_for_reg[var_emissions]))
plt.scatter(np.log(results_for_reg.agglo_spag), np.log(results_for_reg[var_emissions]))



mod = smf.ols('emissions_pc_moran ~ n_pop + density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_moran_buildings ~ n_pop + density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_moran_transport ~  density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_kona ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_kona_scope1 ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_kona_buildings ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_kona_transport ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + footprint_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_nangini_scope1 ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + footprint_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())

mod = smf.ols('emissions_pc_nangini ~ density_b + avg_height + vol_per_cap + centrality1 + volume_profile + footprint_profile + agglo_spag', data=results_for_reg)
res = mod.fit()
print(res.summary())