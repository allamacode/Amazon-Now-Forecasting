import pandas as pd
import numpy as np
import random
import datetime
import os
import requests
import time

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Godowns (Keeping the previously generated ones if they exist to save 60s of geocoding)
if os.path.exists('data/godowns.csv'):
    print("Loading existing Bengaluru godowns...")
    godowns_list = pd.read_csv('data/godowns.csv').to_dict('records')
else:
    print("Godowns file not found. Please run the previous generator once.")
    exit(1)

import requests
import json
import time

def fetch_amazon_skus(api_key, search_terms):
    print("Fetching real SKUs from Amazon.in via Rainforest API...")
    products = []
    pid = 1
    
    for term in search_terms:
        params = {
            'api_key': api_key,
            'type': 'search',
            'amazon_domain': 'amazon.in',
            'search_term': term
        }
        try:
            res = requests.get('https://api.rainforestapi.com/request', params=params).json()
            if 'request_info' in res and not res['request_info'].get('success', True):
                print(f"Rainforest API Error: {res['request_info'].get('message')}")
                continue
                
            for item in res.get('search_results', []):
                # Ensure we have a valid title and price
                if 'title' in item and 'price' in item and 'value' in item['price']:
                    products.append({
                        'product_id': f"SKU_{pid:03d}",
                        'name': item['title'][:60] + "..." if len(item['title']) > 60 else item['title'],
                        'category': term.capitalize(),
                        'base_price': item['price']['value'],
                        'is_perishable': term in ['groceries', 'fresh produce', 'dairy'],
                        'shelf_life_days': random.randint(2, 7) if term in ['groceries', 'fresh produce', 'dairy'] else random.randint(30, 365)
                    })
                    pid += 1
        except Exception as e:
            print(f"Failed to fetch term '{term}': {e}")
            
    return products

products_list = fetch_amazon_skus(os.getenv('RAINFOREST_API_KEY', 'YOUR_API_KEY_HERE'), ['groceries', 'skincare', 'electronics', 'kitchen accessories'])

if not products_list:
    products_list = [{'product_id': 'SKU_001', 'name': 'Fallback Apple', 'category': 'Groceries', 'is_perishable': True, 'shelf_life_days': 5, 'base_price': 100.0}]

df_products = pd.DataFrame(products_list)

# Inject Cold Start logic: 5% of products are "New Launches" at Day 88
print("Tagging new launches for Cold Start simulation...")
df_products['launch_day'] = 0
num_new_launches = max(1, int(len(df_products) * 0.05))
new_launch_indices = np.random.choice(df_products.index, num_new_launches, replace=False)
df_products.loc[new_launch_indices, 'launch_day'] = 88

# Refresh the products_list to include launch_day
products_list = df_products.to_dict('records')

df_products.to_csv('data/products.csv', index=False)
print(f"Saved {len(products_list)} real products.")

# 3. Exogenous Variables: Weather and Promotions
print("Generating Weather and Promotion Calendar...")
start_date = datetime.datetime.now() - datetime.timedelta(days=90)
weather_conditions = ['Clear', 'Clear', 'Clear', 'Rain', 'Extreme Heat']
macro_log = []

for d in range(90):
    date_val = (start_date + datetime.timedelta(days=d)).date()
    weather = random.choice(weather_conditions)
    is_promo = 1 if d in [85, 86] else 0 # Prime Day on day 85 and 86
    macro_log.append({
        'date': date_val,
        'weather': weather,
        'is_promotion': is_promo
    })
    
df_macro = pd.DataFrame(macro_log)
df_macro.to_csv('data/macro_log.csv', index=False)

# 4. Orders History (90 days) with Customer Personas & Exogenous Variables
print("Generating Orders with distinct buying patterns and macro events...")
orders_list = []

for _ in range(1500000):  # Massively Expanded dataset
    godown = random.choice(godowns_list)
    product = random.choice(products_list)
    
    # Time logic
    random_days = random.randint(0, 89)
    
    # Cold Start constraint: Cannot have orders before it was launched!
    if random_days < product.get('launch_day', 0):
        continue
        
    random_hour = random.randint(0, 23)
    order_time = start_date + datetime.timedelta(days=random_days, hours=random_hour)
    is_weekend = order_time.weekday() >= 5
    
    demand = random.randint(1, 4)
    
    # Exogenous Modifiers
    weather_today = df_macro.iloc[random_days]['weather']
    is_promo_today = df_macro.iloc[random_days]['is_promotion']
    
    if weather_today == 'Rain':
        demand += random.randint(2, 5) # People order more delivery in rain
    elif weather_today == 'Extreme Heat' and product['category'] in ['Groceries', 'fresh produce']:
        demand += random.randint(3, 6) # Order cold drinks / fresh items
        
    if is_promo_today == 1:
        demand *= 3 # Massive spike on Prime Day
        
    # PERSONA 1: IT Tech Parks -> High frequency snacking at 3 PM - 6 PM
    if godown['locality_type'] == 'IT Tech Park' and product['category'] == 'Groceries':
        if 15 <= random_hour <= 18 and not is_weekend:
            demand += random.randint(5, 12)
            
    # PERSONA 2: Residential -> Weekend Grocery Stockups (Mornings)
    elif godown['locality_type'] == 'Residential' and product['category'] == 'Groceries':
        if is_weekend and (8 <= random_hour <= 12):
            demand += random.randint(8, 15)
            
    # PERSONA 3: Commercial -> Beauty/Kitchen accessories randomly during day
    elif godown['locality_type'] == 'Commercial' and product['category'] != 'Groceries':
        if 10 <= random_hour <= 19:
            demand += random.randint(3, 7)
            
    orders_list.append({
        'order_id': f'ORD_{random.randint(100000, 999999)}',
        'godown_id': godown['godown_id'],
        'product_id': product['product_id'],
        'order_timestamp': order_time.strftime('%Y-%m-%d %H:%M:%S'),
        'quantity': demand
    })

df_orders = pd.DataFrame(orders_list)
df_orders.to_csv('data/orders.csv', index=False)

print("Data generation complete! Personas injected.")
