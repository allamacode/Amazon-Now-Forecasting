# Amazon Now Forecasting - Enterprise AI Console

This repository contains the **Amazon Now (Bengaluru) - Enterprise AI Console**, a prototype dashboard for predicting hyper-local demand, managing fleet routing, and optimizing overnight restocks. 

## Features & Purpose

### 1. Live Dashboard & Geospatial Fleet Radar
![Real-Time Geospatial View](images/real_time_geospatial_view.png)

**Command Center - Godown Selection:**
![Godown Selection](images/different_godowns.png)

**Active Fleet Telemetry & Rider Tracking:**
![Active Fleet Telemetry](images/active_fleet_telemetry.png)

- **Purpose**: Provides a real-time, bird's-eye view of all godowns and tracks simulated riders delivering orders. It dynamically calculates fleet availability based on live traffic.
- **Why we used it (Free Tools)**: `PyDeck` (via Streamlit) is used for rendering interactive maps, and the `OSRM (Open Source Routing Machine) API` is used to fetch live traffic/routing data. OSRM is a completely free, open-source routing engine. 
- **Production Alternative**: In production, **Google Maps Platform (Routing API)** or **Mapbox** would be used for high-accuracy, enterprise-grade traffic and routing data, as OSRM public servers are not meant for high-scale commercial loads.

### 2. Enterprise AI Forecasting Engine (XGBoost)
![Demand Forecast & XAI Insights](images/demand_forecast_xai.png)
- **Purpose**: Predicts the next 24-hour demand for every single product at a specific godown. This helps prevent out-of-stock scenarios.
- **Why we used it (Free Tools)**: `XGBoost` is a highly efficient, open-source, and free machine learning algorithm that is the industry standard for tabular data forecasting. We also used `Scikit-Learn` for encoding and preprocessing.
- **Production Alternative**: In a massive production environment like Amazon, we would use managed services like **Amazon Forecast**, **AWS SageMaker**, or **Google Cloud Vertex AI** to train models across billions of rows distributed across GPU clusters.

### 3. Explainable AI (XAI) Insights
- **Purpose**: Instead of a "black box" prediction, this feature explicitly explains *why* the AI made a certain forecast by showing Feature Importances (e.g., how much did 'Weather' or 'Promotions' affect the outcome).
- **Why we used it (Free Tools)**: We used `Plotly Express` (free/open-source charting library) and the native feature importance metric inside `XGBoost`.
- **Production Alternative**: For advanced production XAI, libraries like **SHAP (SHapley Additive exPlanations)** or **LIME** would be integrated via cloud services to provide user-level and row-level explanations.

### 4. Overnight Procurement & Restock Manifests
![Overnight Procurement](images/overnight_procurement.png)
- **Purpose**: Automates the decision of what needs to be transferred from the main warehouse to local godowns to meet tomorrow's demand. It auto-generates a CSV packing slip.
- **Why we used it (Free Tools)**: Pure `Pandas` and `Streamlit` logic to compare forecasted demand against simulated current shelf stock.
- **Production Alternative**: This would be directly integrated into an **ERP System (like SAP or Oracle)** or **AWS Supply Chain** APIs, instantly triggering robotic pick-and-pack workflows rather than just downloading a CSV.

### 5. Lost Intent & Catalog Expansion
![Catalog Expansion](images/catalog_expansion.png)
- **Purpose**: Simulates tracking when customers search for items that aren't currently stocked at their local godown. It estimates lost revenue and allows managers to auto-request expansion.
- **Why we used it (Free Tools)**: Custom analytical logic simulating user search logs in `Pandas`. 
- **Production Alternative**: In production, this data would stream live from **Elasticsearch** (search logs) through **Apache Kafka / AWS Kinesis** into a real-time analytics engine to accurately quantify lost intent.

### 6. Historical Demand Trends
![Historical Demand Trends](images/historical_trends.png)
- **Purpose**: Visualizes the last 30 days of order volume for the entire godown or drills down into specific products to spot seasonal patterns.
- **Why we used it (Free Tools)**: `Plotly Express` for interactive charting, rendering free and fluid graphs.
- **Production Alternative**: A dedicated BI tool like **Tableau**, **PowerBI**, or **Amazon QuickSight** would be used to handle massive historical aggregations via data warehouses like **Snowflake** or **Amazon Redshift**.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data generation: `python generate_data.py`
3. Run feature engineering: `python feature_engineering.py`
4. Train the model: `python model_training.py`
5. Start the dashboard: `start_dashboard.bat` or `streamlit run app.py`

## Screenshots
*(Add screenshots of each feature here once available)*
