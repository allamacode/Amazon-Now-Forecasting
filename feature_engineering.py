import pandas as pd
import numpy as np
import os
import requests

print("Starting Feature Engineering...")

godowns = pd.read_csv('data/godowns.csv')
products = pd.read_csv('data/products.csv')
orders = pd.read_csv('data/orders.csv')
macro_log = pd.read_csv('data/macro_log.csv')
macro_log['date'] = pd.to_datetime(macro_log['date']).dt.date

# Convert timestamp
orders['order_timestamp'] = pd.to_datetime(orders['order_timestamp'])
orders['date'] = orders['order_timestamp'].dt.date
orders['hour'] = orders['order_timestamp'].dt.hour
orders['day_of_week'] = orders['order_timestamp'].dt.dayofweek
orders['is_weekend'] = orders['day_of_week'] >= 5

# Aggregate daily demand per Godown per Product
daily_demand = orders.groupby(['godown_id', 'product_id', 'date']).agg({
    'quantity': 'sum',
    'hour': 'mean', # Average time of orders
    'is_weekend': 'max'
}).reset_index()

daily_demand.rename(columns={'quantity': 'daily_demand'}, inplace=True)
daily_demand['date'] = pd.to_datetime(daily_demand['date'])

# Generate full permutations to fill zero-demand days
print("Generating continuous time-series...")
all_dates = pd.date_range(start=daily_demand['date'].min(), end=daily_demand['date'].max())
all_godowns = godowns['godown_id'].unique()
all_products = products['product_id'].unique()

idx = pd.MultiIndex.from_product([all_dates, all_godowns, all_products], names=['date', 'godown_id', 'product_id'])
full_df = pd.DataFrame(index=idx).reset_index()

# Merge
df = pd.merge(full_df, daily_demand, on=['date', 'godown_id', 'product_id'], how='left')
df['daily_demand'].fillna(0, inplace=True)
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['day_of_week'] = df['date'].dt.dayofweek

# Rolling features (Crucial for learning Personas)
print("Computing rolling window features...")
df.sort_values(by=['godown_id', 'product_id', 'date'], inplace=True)

# 7-day and 14-day rolling averages
df['rolling_7d_demand'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
df['rolling_14d_demand'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())

# Advanced Lags (1-day, 2-day, 7-day)
df['lag_1d'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].shift(1)
df['lag_2d'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].shift(2)
df['lag_7d'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].shift(7)

# Exponential Moving Average (EMA) - more sensitive to recent changes
df['EMA_7d'] = df.groupby(['godown_id', 'product_id'])['daily_demand'].transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean())

# Merge Godown and Product Meta
df = pd.merge(df, godowns[['godown_id', 'locality_type', 'capacity_units', 'latitude', 'longitude']], on='godown_id', how='left')
df = pd.merge(df, products[['product_id', 'category', 'is_perishable', 'base_price', 'launch_day']], on='product_id', how='left')

# Exogenous Features
df['date_only'] = df['date'].dt.date
df = pd.merge(df, macro_log, left_on='date_only', right_on='date', how='left').drop(columns=['date_only', 'date_y']).rename(columns={'date_x': 'date'})

# One-hot encode weather
df['Weather_Rain'] = (df['weather'] == 'Rain').astype(int)
df['Weather_Heat'] = (df['weather'] == 'Extreme Heat').astype(int)
df.drop(columns=['weather'], inplace=True)

# Cold Start Features
start_date = df['date'].min()
df['current_day_index'] = (df['date'] - start_date).dt.days
df['days_since_launch'] = df['current_day_index'] - df['launch_day']
df['days_since_launch'] = df['days_since_launch'].clip(lower=0)

# Build a Lookalike Feature: Category Average 7d Demand
# First, find the average rolling 7d demand per category
df['category_average_7d_demand'] = df.groupby(['category', 'date'])['rolling_7d_demand'].transform('mean')

# Impute rolling features for Cold Start items
cold_start_mask = df['days_since_launch'] <= 14
df.loc[cold_start_mask, 'rolling_7d_demand'] = df.loc[cold_start_mask, 'category_average_7d_demand']
df.loc[cold_start_mask, 'rolling_14d_demand'] = df.loc[cold_start_mask, 'category_average_7d_demand']
df.loc[cold_start_mask, 'EMA_7d'] = df.loc[cold_start_mask, 'category_average_7d_demand']

# Categorical Encoding
df['locality_type'] = df['locality_type'].astype('category').cat.codes
df['category'] = df['category'].astype('category').cat.codes

# Fill NaNs
df.fillna(0, inplace=True)

df.to_csv('data/engineered_features.csv', index=False)
print("Feature Engineering Complete. Shape:", df.shape)
