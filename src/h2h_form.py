# H2H + Form Features
import pandas as pd
from scripts.load_clean_historical_data import load_clean_historical_data

modelling_df = load_clean_historical_data()
df_train = modelling_df[
    (modelling_df['FTHG'].notna()) & 
    (modelling_df['FTAG'].notna()) & 
    (modelling_df['B365H'].notna())
].copy()