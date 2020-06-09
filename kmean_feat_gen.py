import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn

df = pd.read_csv('fma_all_features_genres.csv')

file_name = df[['file_name']]
file_name_list = file_name.values.tolist()

name_array = []

for i in range(0, len(file_name)):
	name = str(file_name_list[i])
	name = name.split("/")[-1]
	name = name.split(".")[0]
	name_array.append(name)	

spectral_flux_arr = 1000*sklearn.preprocessing.minmax_scale(df[['spectral_flux']], axis=0)
zero_cross_rate_arr = 1000*sklearn.preprocessing.minmax_scale(df[['zcr_mean']], axis=0)
spectral_rolloff_arr = 1000*sklearn.preprocessing.minmax_scale(df[['spectral_rolloff_mean']], axis=0)
sp_fx_df = pd.DataFrame(spectral_rolloff_arr, columns=['spectral_flux'])
zcr_df = pd.DataFrame(zero_cross_rate_arr, columns=['zcr_mean'])
sr_df = pd.DataFrame(spectral_rolloff_arr, columns=['spectral_rolloff_mean'])

# for i in range(0, len(df)):
# 	file_name = df[[zcr_mean[i]]]

kmean_df = pd.concat([pd.DataFrame(name_array,columns=['Song_Name']), zcr_df, sp_fx_df, sr_df, df[['Label']]], axis = 1)
kmean_df = kmean_df.sort_values(by=['zcr_mean'])
kmean_df.to_csv('k_mean_feat.csv', index=False)