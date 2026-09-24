import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
import datetime
import xgboost as xgb
import pickle
import os
import hashlib
import time
import math
import json
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")

def load_model():
    try:
        model = xgb.XGBRegressor()
        model.load_model('models/demand_forecast_model.json')
    except:
        model = None
    return model

def load_encoders():
    try:
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return encoders
    except:
        return {}

def load_data():
    try:
        godowns = pd.read_csv('data/godowns.csv')
        products = pd.read_csv('data/products.csv')
        engineered = pd.read_csv('data/engineered_features.csv')
        # Get only the latest date's features for prediction
        engineered['date'] = pd.to_datetime(engineered['date'])
        latest_date = engineered['date'].max()
        latest_features = engineered[engineered['date'] == latest_date]
        return godowns, products, engineered, latest_features
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_live_traffic_congestion(lat, lon):
    try:
        dest_lat = lat + 0.03
        dest_lon = lon + 0.03
        url = f"http://router.project-osrm.org/route/v1/driving/{lon},{lat};{dest_lon},{dest_lat}?overview=false"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                duration_sec = data['routes'][0]['duration']
                distance_m = data['routes'][0]['distance']
                expected_duration_sec = distance_m / 6.94
                ratio = expected_duration_sec / duration_sec if duration_sec > 0 else 1.0
                fleet_availability = min(100.0, max(40.0, ratio * 100))
                return fleet_availability
    except Exception as e:
        pass
    return 100.0


st.title("Amazon Now (Bengaluru) - Enterprise AI Console")

model = load_model()
encoders = load_encoders()
godowns, products, all_historical, latest_features = load_data()

if model is None or godowns.empty:
    st.warning("Model or data not found. Please run the training scripts first.")
else:
    st.sidebar.header("Command Center")
    
    sorted_localities = sorted(godowns['locality'].unique())
    selected_locality = st.sidebar.selectbox("Select Godown Locality", sorted_localities)
    godown_data = godowns[godowns['locality'] == selected_locality].iloc[0]
    
    selected_godown = godown_data['godown_id']
    city = godown_data['city']
    lat = godown_data['latitude']
    lon = godown_data['longitude']
    loc_type = godown_data['locality_type']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Fleet Telemetry")
    
    live_tracking = st.sidebar.toggle("Enable Live Fleet Radar (Auto-Refresh)", value=False)
    
    with st.spinner("Pinging OSRM..."):
        fleet_availability = fetch_live_traffic_congestion(lat, lon)
        
    alert_trigger = st.sidebar.button("Trigger Fleet Emergency Alert")
    if alert_trigger:
        st.sidebar.error("ALERT ACTIVE: Diverting backup fleet.")
        fleet_availability = max(15.0, fleet_availability - 60.0) # Simulate massive drop
        
    st.sidebar.success(f"Fulfillment Cap: {fleet_availability:.1f}%")
    
    is_weekend = st.sidebar.checkbox("Simulate Weekend Scenario")
    
    # Alert banners
    if fleet_availability < 75:
        st.warning(f"ROUTING BOTTLENECK: Live traffic confirms fleet fulfillment capability is capped at {fleet_availability:.1f}%.")
    if alert_trigger:
        st.error("EMERGENCY ALERT TRIGGERED: Massive demand surge detected. Diverting backup fleet from neighboring zones immediately.")

    tab1, tab2, tab4, tab3 = st.tabs(["Live Dashboard", "Overnight Procurement", "Catalog Expansion", "Historical Demand Trends"])
    
    with tab1:
        if live_tracking:
            st_autorefresh(interval=5000, key="data_refresh")
        st.markdown("### Real-Time Geospatial View")
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Godown", f"{selected_locality}", f"{loc_type.title()} Persona")
        col2.metric("Fleet Availability", f"{fleet_availability:.1f}%", "- Bottlenecked" if fleet_availability < 75 else "Optimal", delta_color="inverse" if fleet_availability < 75 else "normal")
        col3.metric("Supported Catalog", f"{len(products)} Live SKUs")

        base_radius = 4000
        actual_radius = base_radius * (fleet_availability / 100.0)
        radius_color = [255, 50, 50, 150] if fleet_availability < 75 else [255, 153, 0, 100]
            
        selected_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{'lat': lat, 'lon': lon}]),
            get_position='[lon, lat]',
            get_radius=actual_radius,
            get_fill_color=radius_color,
            pickable=True
        )
        
        all_godowns_layer = pdk.Layer(
            "ScatterplotLayer",
            data=godowns,
            get_position='[longitude, latitude]',
            get_radius=300,
            get_fill_color=[255, 255, 255, 255],
            get_line_color=[0, 0, 0, 255],
            lineWidthMinPixels=1,
            pickable=True
        )

        @st.cache_data(ttl=600, show_spinner=False)
        def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
            url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=simplified&geometries=geojson"
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    coords = r.json()['routes'][0]['geometry']['coordinates']
                    return [{"path": coords, "color": [0, 150, 255, 150]}]
            except:
                pass
            return [{"path": [[start_lon, start_lat], [end_lon, end_lat]], "color": [255, 0, 0, 150]}]
            
        def interpolate_along_line(line_coords, fraction):
            if not line_coords: return None
            if fraction <= 0: return line_coords[0]
            if fraction >= 1: return line_coords[-1]
            total_len = 0
            segments = []
            for i in range(len(line_coords)-1):
                p1, p2 = line_coords[i], line_coords[i+1]
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                segments.append(dist)
                total_len += dist
            if total_len == 0: return line_coords[0]
            target_dist = total_len * fraction
            curr_dist = 0
            for i in range(len(line_coords)-1):
                if curr_dist + segments[i] >= target_dist:
                    seg_frac = (target_dist - curr_dist) / segments[i] if segments[i] > 0 else 0
                    p1, p2 = line_coords[i], line_coords[i+1]
                    lon = p1[0] + (p2[0] - p1[0]) * seg_frac
                    lat = p1[1] + (p2[1] - p1[1]) * seg_frac
                    return [lon, lat]
                curr_dist += segments[i]
            return line_coords[-1]

        def get_live_riders(godown_id, center_lat, center_lon, num_riders=5):
            current_time = time.time()
            assignment_window = int(current_time / 600) 
            riders = []
            paths = []
            for i in range(num_riders):
                np.random.seed(int(hashlib.md5(f"{godown_id}_r{i}_{assignment_window}".encode()).hexdigest(), 16) % (2**32))
                dest_lat = center_lat + np.random.uniform(-0.02, 0.02)
                dest_lon = center_lon + np.random.uniform(-0.02, 0.02)
                route_obj = get_osrm_route(center_lat, center_lon, dest_lat, dest_lon)
                rider_id = f"RIDER-{godown_id.split('_')[-1]}-10{i}"
                if route_obj:
                    coords = route_obj[0]["path"]
                    route_obj[0]["rider_id"] = rider_id
                    paths.append(route_obj[0])
                    cycle_time = 300
                    time_in_cycle = (current_time + (i * 60)) % cycle_time
                    if time_in_cycle < 120:
                        fraction = time_in_cycle / 120.0
                        status = "En Route to Customer"
                    elif time_in_cycle < 150:
                        fraction = 1.0
                        status = "Delivering"
                    elif time_in_cycle < 270:
                        fraction = 1.0 - ((time_in_cycle - 150) / 120.0)
                        status = "Returning to Godown"
                    else:
                        fraction = 0.0
                        status = "Idle"
                    rider_pos = interpolate_along_line(coords, fraction)
                    if not rider_pos: rider_pos = [center_lon, center_lat]
                    dist_remaining = 0
                    if status == "En Route to Customer": dist_remaining = (1.0 - fraction) * 2.0
                    elif status == "Returning to Godown": dist_remaining = fraction * 2.0
                    riders.append({
                        "Rider ID": rider_id,
                        "lat": rider_pos[1],
                        "lon": rider_pos[0],
                        "Status": status,
                        "Distance (km)": round(dist_remaining, 2)
                    })
            np.random.seed()
            return pd.DataFrame(riders), paths
            
        rider_df, dispatch_paths = get_live_riders(selected_godown, lat, lon)
        
        tracked_rider = st.selectbox("Track Specific Rider (Optional)", ["None"] + list(rider_df['Rider ID']))
        
        if tracked_rider != "None":
            dispatch_paths = [p for p in dispatch_paths if p.get("rider_id") == tracked_rider]
            rider_df['color'] = rider_df['Rider ID'].apply(lambda x: [255, 255, 0, 255] if x == tracked_rider else [0, 255, 128, 150])
            rider_df['radius'] = rider_df['Rider ID'].apply(lambda x: 200 if x == tracked_rider else 100)
        else:
            dispatch_paths = []
            rider_df['color'] = rider_df['Rider ID'].apply(lambda x: [0, 255, 128, 255])
            rider_df['radius'] = 100
        
        path_layer = pdk.Layer(
            "PathLayer",
            data=dispatch_paths,
            get_path="path",
            get_color="color",
            width_scale=20,
            width_min_pixels=3,
            get_width=5
        )
        
        rider_layer = pdk.Layer(
            "ScatterplotLayer",
            data=rider_df,
            get_position='[lon, lat]',
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[0, 0, 0, 255],
            lineWidthMinPixels=2,
            pickable=True
        )

        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=50),
            layers=[selected_layer, all_godowns_layer, path_layer, rider_layer],
            tooltip={"text": "Live Fleet Asset"}
        ))
        
        st.markdown("#### Active Fleet Telemetry")
        st.dataframe(rider_df[['Rider ID', 'Status', 'Distance (km)']], use_container_width=True, height=200)

        # Prediction
        st.markdown("### 🤖 Enterprise AI Forecasting Engine")
        
        # Load and display model metrics
        try:
            with open('models/metrics.json', 'r') as f:
                metrics = json.load(f)
            
            st.info("The metrics below represent the **Global Model Health** from the most recent overnight training run across all 1.2M rows. They remain static during the day to serve as a baseline trust score for the engine.")
            m1, m2 = st.columns(2)
            m1.metric("Global RMSE (Root Mean Squared Error)", f"{metrics['rmse']} units")
            m2.metric("Global MAE (Mean Absolute Error)", f"{metrics['mae']} units")
        except:
            pass
            
        st.markdown("---")
        
        test_df = latest_features[latest_features['godown_id_x' if 'godown_id_x' in latest_features.columns else 'godown_id'] == selected_godown].copy()
        
        # Display Current Macro Environment
        if not test_df.empty:
            st.markdown("#### 🌍 Macro Environment Context")
            col_weather, col_promo, _ = st.columns([1, 1, 2])
            
            is_rain = test_df.iloc[0].get('Weather_Rain', 0) == 1
            is_heat = test_df.iloc[0].get('Weather_Heat', 0) == 1
            is_promo = test_df.iloc[0].get('is_promotion', 0) == 1
            
            if is_rain:
                weather_str = "🌧️ Heavy Rain"
            elif is_heat:
                weather_str = "☀️ Extreme Heat"
            else:
                weather_str = "⛅ Clear"
                
            promo_str = "🏷️ Prime Day Active (3x Volatility Expected)" if is_promo else "No Active Promotions"
            
            col_weather.info(f"**Current Weather:** {weather_str}")
            if is_promo:
                col_promo.error(f"**Promotion:** {promo_str}")
            else:
                col_promo.success(f"**Promotion:** {promo_str}")
                
        st.markdown("---")
        st.markdown("#### Demand Forecast (Next 24 Hrs)")
        
        if test_df.empty:
            st.info("No rolling data available yet. Please wait for overnight batch sync.")
        else:
            # Re-map unencoded features for ML
            test_df['is_weekend'] = int(is_weekend)
            test_df['day_of_week'] = 6 if is_weekend else 2
            
            # Extract features explicitly expected by the model
            ml_features = [
                'day_of_week', 'is_weekend', 'rolling_7d_demand', 'rolling_14d_demand', 
                'lag_1d', 'lag_2d', 'lag_7d', 'EMA_7d', 'base_price', 'is_perishable', 
                'locality_type', 'category', 'Weather_Rain', 'Weather_Heat', 'is_promotion',
                'days_since_launch', 'category_average_7d_demand', 'godown_id', 'product_id'
            ]
            
            for col in ['godown_id', 'product_id']:
                if col in encoders:
                    le = encoders[col]
                    test_df[col] = test_df[col].apply(lambda x: le.transform([x])[0] if str(x) in le.classes_ else 0)
            
            try:
                X = test_df[ml_features]
                preds = model.predict(X)
                test_df['forecasted_demand'] = np.maximum(0, preds).astype(int)
                
                # Confidence Intervals
                rmse_val = metrics.get('rmse', 3.92) if 'metrics' in locals() else 3.92
                test_df['Lower Bound'] = np.maximum(0, test_df['forecasted_demand'] - (1.96 * rmse_val)).astype(int)
                test_df['Upper Bound'] = (test_df['forecasted_demand'] + (1.96 * rmse_val)).astype(int)
                test_df['Confidence Interval'] = test_df.apply(lambda row: f"{row['Lower Bound']} - {row['Upper Bound']}", axis=1)
                
                if fleet_availability < 100:
                    fulfillment_factor = fleet_availability / 100.0
                    test_df['fulfilled_demand'] = (test_df['forecasted_demand'] * fulfillment_factor).astype(int)
                else:
                    test_df['fulfilled_demand'] = test_df['forecasted_demand']
                
                # Merge back product names for display
                # We need to decode product_id
                le_prod = encoders['product_id']
                
                def get_product_name_with_badge(row):
                    x = row['product_id']
                    days_since = row.get('days_since_launch', 99)
                    base_name = products[products['product_id'] == le_prod.inverse_transform([x])[0]]['name'].values[0] if x < len(le_prod.classes_) else 'Unknown'
                    if days_since <= 14:
                        return f"🆕 [NEW] {base_name}"
                    return base_name

                test_df['Product Name'] = test_df.apply(get_product_name_with_badge, axis=1)
                test_df['Price (₹)'] = test_df['product_id'].apply(lambda x: f"₹{products[products['product_id'] == le_prod.inverse_transform([x])[0]]['base_price'].values[0]:.2f}" if x < len(le_prod.classes_) else '₹0.00')
                
                def get_current_stock(godown_id, product_name):
                    h = int(hashlib.md5(f"{godown_id}_{product_name}".encode('utf-8')).hexdigest(), 16)
                    return h % 15 # Simulated deterministic stock between 0 and 14
                    
                test_df['Current Shelf Stock'] = test_df.apply(lambda row: get_current_stock(row['godown_id'], row['Product Name']), axis=1)
                test_df['Transfer Request'] = np.maximum(0, test_df['forecasted_demand'] - test_df['Current Shelf Stock']).astype(int)
                
                # Constrain the width of the dropdown using columns
                col_sort, _ = st.columns([1, 4])
                with col_sort:
                    sort_option = st.selectbox("Sort Inventory By", ["Alphabetical", "Most Ordered", "Least Ordered"])
                
                if sort_option == "Alphabetical":
                    disp_df = test_df[['Product Name', 'Price (₹)', 'is_perishable', 'Confidence Interval', 'forecasted_demand', 'fulfilled_demand']].sort_values(by='Product Name', ascending=True)
                elif sort_option == "Most Ordered":
                    disp_df = test_df[['Product Name', 'Price (₹)', 'is_perishable', 'Confidence Interval', 'forecasted_demand', 'fulfilled_demand']].sort_values(by='forecasted_demand', ascending=False)
                else:
                    disp_df = test_df[['Product Name', 'Price (₹)', 'is_perishable', 'Confidence Interval', 'forecasted_demand', 'fulfilled_demand']].sort_values(by='forecasted_demand', ascending=True)
                
                def highlight_perishables(val):
                    return 'background-color: #FF9900; color: black' if val else ''
                
                st.dataframe(disp_df.style.map(highlight_perishables, subset=['is_perishable']), height=400, use_container_width=True)
                
                # XAI Panel
                st.markdown("#### 🧠 Explainable AI (XAI) Insights")
                with st.expander("View Global Feature Importances", expanded=False):
                    try:
                        with open('models/feature_importance.json', 'r') as f:
                            importances = json.load(f)
                        imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance Score']).sort_values(by='Importance Score', ascending=True)
                        import plotly.express as px
                        
                        # Clean up feature names for display
                        feature_name_mapping = {
                            'locality_type': 'Locality Demographics',
                            'is_perishable': 'Perishability',
                            'is_weekend': 'Weekend Status',
                            'rolling_14d_demand': '14-Day Rolling Demand',
                            'rolling_7d_demand': '7-Day Rolling Demand',
                            'day_of_week': 'Day of the Week',
                            'product_id': 'Specific Product ID',
                            'godown_id': 'Specific Godown ID',
                            'lag_7d': 'Demand 7 Days Ago',
                            'lag_2d': 'Demand 2 Days Ago',
                            'lag_1d': 'Demand 1 Day Ago',
                            'EMA_7d': '7-Day Exponential Moving Avg',
                            'category': 'Product Category',
                            'base_price': 'Base Price',
                            'is_promotion': 'Active Promotions',
                            'Weather_Rain': 'Heavy Rain',
                            'Weather_Heat': 'Extreme Heat',
                            'days_since_launch': 'Days Since Launch (Cold Start)',
                            'category_average_7d_demand': 'Category 7D Avg (Lookalike)'
                        }
                        imp_df['Feature Name'] = imp_df['Feature'].map(lambda x: feature_name_mapping.get(x, x))
                        
                        fig = px.bar(imp_df, 
                                     x='Importance Score', 
                                     y='Feature Name', 
                                     orientation='h', 
                                     title='Mathematical Drivers Behind the AI Forecast',
                                     color_discrete_sequence=['#FF9900']) # Amazon Orange
                                     
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", 
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Mathematical Weight (Gain)"),
                            yaxis=dict(showgrid=False, title=""),
                            font=dict(color="white", size=12),
                            hovermode="y unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.write("Feature importances not available.", e)
                
            except Exception as e:
                st.error(f"Error making predictions: {e}")

    with tab2:
        st.markdown("### Overnight Procurement & Restock")
        st.info("Automated Transfer Requests generated based on AI Forecast vs Current Shelf Stock.")
        
        try:
            if 'Transfer Request' in test_df.columns:
                restock_df = test_df[test_df['Transfer Request'] > 0][['Product Name', 'Price (₹)', 'is_perishable', 'Current Shelf Stock', 'forecasted_demand', 'Transfer Request']]
                restock_df = restock_df.sort_values(by='Transfer Request', ascending=False)
                
                if not restock_df.empty:
                    def highlight_critical(row):
                        if row['Current Shelf Stock'] == 0 and row['forecasted_demand'] >= 5:
                            return ['background-color: #FF4B4B; color: white'] * len(row)
                        return [''] * len(row)

                    st.dataframe(restock_df.style.apply(highlight_critical, axis=1), height=400, use_container_width=True)
                    
                    total_items = restock_df['Transfer Request'].sum()
                    st.metric("Total Units to Transfer", f"{total_items} Units")
                    
                    if st.button("Send Transfer Request to Main Warehouse", type="primary"):
                        truck_html = """
                        <style>
                        @keyframes drive {
                          0% { transform: translateX(-100%); }
                          20% { transform: translateX(10vw); }
                          80% { transform: translateX(10vw); }
                          100% { transform: translateX(120vw); }
                        }
                        @keyframes load {
                          0%, 20% { opacity: 0; transform: translateY(-20px); }
                          30% { opacity: 1; transform: translateY(0); }
                          40% { opacity: 0; transform: translateY(10px); }
                          50% { opacity: 1; transform: translateY(0); }
                          60% { opacity: 0; transform: translateY(10px); }
                          70% { opacity: 1; transform: translateY(0); }
                          80% { opacity: 0; transform: translateY(10px); }
                          100% { opacity: 0; }
                        }
                        @keyframes fadeInOut {
                          0%, 15% { opacity: 0; }
                          20%, 80% { opacity: 1; }
                          85%, 100% { opacity: 0; }
                        }
                        .animation-container {
                          position: relative;
                          height: 80px;
                          overflow: hidden;
                          background: #1E1E1E;
                          border-radius: 8px;
                          margin: 15px 0;
                          width: 100%;
                        }
                        .truck {
                          position: absolute;
                          font-size: 50px;
                          animation: drive 3.5s ease-in-out forwards;
                          top: 10px;
                        }
                        .worker {
                          position: absolute;
                          font-size: 40px;
                          left: 5vw;
                          top: 15px;
                          opacity: 0;
                          animation: fadeInOut 3.5s forwards;
                        }
                        .package {
                          position: absolute;
                          font-size: 30px;
                          left: 8vw;
                          top: 20px;
                          animation: load 3.5s forwards;
                        }
                        </style>
                        <div class="animation-container">
                            <div class="worker">👷‍♂️</div>
                            <div class="package">📦</div>
                            <div class="truck">🚚</div>
                        </div>
                        """
                        animation_placeholder = st.empty()
                        animation_placeholder.markdown(truck_html, unsafe_allow_html=True)
                        time.sleep(3.5) # Wait for animation
                        animation_placeholder.empty() # Remove the animation box
                        
                        st.session_state['manifest_godown'] = selected_godown
                        st.session_state['manifest_time'] = time.time()
                        st.session_state['manifest_df'] = restock_df[['Product Name', 'Transfer Request']].copy()
                        
                    if st.session_state.get('manifest_godown') == selected_godown:
                        st.success("Transfer Request Transmitted Successfully!")
                        
                        with st.expander("View Generated Packing Slip Manifest", expanded=True):
                            st.write(f"**Manifest ID**: TXN-{int(st.session_state['manifest_time'])}")
                            st.write(f"**Destination**: {selected_locality} Godown")
                            
                            manifest_df = st.session_state['manifest_df']
                            manifest_df.columns = ['Item Description', 'Qty to Pick']
                            st.dataframe(manifest_df, hide_index=True, use_container_width=True)
                            
                            csv = manifest_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV Packing Slip",
                                data=csv,
                                file_name=f"packing_slip_{selected_locality}.csv",
                                mime="text/csv",
                                key="download_slip_btn"
                            )
                else:
                    st.success("No restock required! Current shelf stock is sufficient for tomorrow's demand.")
        except Exception as e:
            st.error("Error loading procurement data.")

    with tab4:
        st.markdown("### Lost Intent & Catalog Expansion")
        st.info("Tracks real-time user searches for items that are currently OUT OF CATALOG in this Godown.")
        
        # Set a deterministic seed based on locality so values don't jump on button clicks
        seed_val = int(hashlib.md5(selected_locality.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed_val)
        
        # Hardcoded Missing Items based on Persona
        missing_catalog = []
        if loc_type == 'IT Tech Park':
            missing_catalog = [
                {"Product Name": "Matcha Green Tea Powder 50g", "Price (₹)": 850, "Searches": np.random.randint(400, 900)},
                {"Product Name": "Kombucha Berry Flavor 330ml", "Price (₹)": 150, "Searches": np.random.randint(300, 700)},
                {"Product Name": "Sugar-Free Energy Drink", "Price (₹)": 110, "Searches": np.random.randint(200, 500)},
                {"Product Name": "Mechanical Keyboard Switch Puller", "Price (₹)": 299, "Searches": np.random.randint(50, 150)}
            ]
        elif loc_type == 'Residential':
            missing_catalog = [
                {"Product Name": "Premium Firm Tofu 200g", "Price (₹)": 95, "Searches": np.random.randint(200, 600)},
                {"Product Name": "Organic Wild Honey 500g", "Price (₹)": 450, "Searches": np.random.randint(150, 450)},
                {"Product Name": "Avocado (Imported) 2pcs", "Price (₹)": 350, "Searches": np.random.randint(300, 550)},
                {"Product Name": "Sourdough Bread Loaf", "Price (₹)": 180, "Searches": np.random.randint(100, 300)}
            ]
        else: # Commercial
            missing_catalog = [
                {"Product Name": "A4 Printer Paper 500 Sheets", "Price (₹)": 320, "Searches": np.random.randint(300, 700)},
                {"Product Name": "Ergonomic Mouse Pad", "Price (₹)": 450, "Searches": np.random.randint(100, 300)},
                {"Product Name": "Filter Coffee Decoction 200ml", "Price (₹)": 120, "Searches": np.random.randint(400, 800)},
                {"Product Name": "Whiteboard Markers (Pack of 4)", "Price (₹)": 150, "Searches": np.random.randint(50, 200)}
            ]
            
        np.random.seed() # Reset seed
        
        lost_intent_df = pd.DataFrame(missing_catalog)
        conversion_rate = 0.15 # 15% of searches actually convert to purchases
        lost_intent_df['Estimated Lost Revenue (₹)'] = (lost_intent_df['Price (₹)'] * lost_intent_df['Searches'] * conversion_rate).astype(int)
        lost_intent_df = lost_intent_df.sort_values(by='Estimated Lost Revenue (₹)', ascending=False)
        
        st.caption("ℹ️ *Lost revenue calculation assumes a realistic 15% Search-to-Purchase conversion rate.*")
        
        st.dataframe(lost_intent_df, height=200, use_container_width=True)
        
        total_lost_rev = lost_intent_df['Estimated Lost Revenue (₹)'].sum()
        top_miss = lost_intent_df.iloc[0]['Product Name']
        
        c1, c2 = st.columns(2)
        c1.metric("Total Estimated Lost Revenue (Daily)", f"₹{total_lost_rev:,.2f}")
        
        item_to_request = c2.selectbox("Select Item for Catalog Expansion", lost_intent_df['Product Name'].tolist())
        
        if st.button("Auto-Request Catalog Expansion", type="primary"):
            with st.spinner("Analyzing demand curves and forwarding to Procurement..."):
                time.sleep(1.5)
            st.session_state['catalog_request_msg'] = f"Expansion Request for '{item_to_request}' sent to Procurement Team!"
            st.session_state['catalog_request_time'] = time.time()
            
        if 'catalog_request_msg' in st.session_state:
            if time.time() - st.session_state.get('catalog_request_time', 0) < 10:
                st.success(st.session_state['catalog_request_msg'])
            else:
                del st.session_state['catalog_request_msg']

    with tab3:
        st.markdown("### 📈 Historical Demand Trends")
        st.info("This tab visualizes the actual past order volumes. Use it to understand historical buying patterns for the entire Godown or drill down into specific products.")
        
        if not all_historical.empty:
            hist_df = all_historical[all_historical['godown_id' if 'godown_id' in all_historical.columns else 'godown_id_x'] == selected_godown].copy()
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            hist_df = hist_df[hist_df['date'] >= (hist_df['date'].max() - pd.Timedelta(days=30))]
            
            # Map product names
            prod_dict = dict(zip(products['product_id'], products['name']))
            hist_df['Product Name'] = hist_df['product_id'].map(prod_dict)
            
            # Selection for drill down
            unique_products = sorted(hist_df['Product Name'].dropna().unique())
            view_option = st.selectbox("Select View:", ["Overall Godown Volume"] + list(unique_products))
            
            if view_option == "Overall Godown Volume":
                trend = hist_df.groupby('date')['daily_demand'].sum().reset_index()
                chart_title = f"Total 30-Day Order Volume across all products at **{selected_locality}**"
            else:
                trend = hist_df[hist_df['Product Name'] == view_option].groupby('date')['daily_demand'].sum().reset_index()
                chart_title = f"30-Day Order Volume for **{view_option}** at **{selected_locality}**"
                
            # Remove the very last day from the visual trend because it is an incomplete partial day and causes an artificial drop
            if not trend.empty:
                trend = trend[trend['date'] < trend['date'].max()]
            
            # Calculate KPIs
            total_orders = int(trend['daily_demand'].sum())
            avg_daily = round(trend['daily_demand'].mean(), 1)
            peak_day = trend.loc[trend['daily_demand'].idxmax()]['date'].strftime('%b %d') if not trend.empty else "N/A"
            peak_vol = int(trend['daily_demand'].max()) if not trend.empty else 0
            
            # Display KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("Total 30-Day Orders", total_orders)
            c2.metric("Daily Average Orders", avg_daily)
            c3.metric(f"Peak Day ({peak_day})", peak_vol, "Highest Volume")
            
            st.markdown(f"**{chart_title}**")
            import plotly.express as px
            fig = px.line(trend, x='date', y='daily_demand', markers=True,
                          line_shape='spline', # Smooth curve
                          color_discrete_sequence=['#00ffcc'])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Daily Order Volume"),
                hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical data not ready. Please run the data generator.")
