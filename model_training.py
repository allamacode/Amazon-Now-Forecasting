import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import pickle
import os
import json

print("Loading engineered features...")
df = pd.read_csv('data/engineered_features.csv')
df['date'] = pd.to_datetime(df['date'])

features = [
    'day_of_week', 'is_weekend', 
    'rolling_7d_demand', 'rolling_14d_demand', 
    'lag_1d', 'lag_2d', 'lag_7d', 'EMA_7d',
    'base_price', 'is_perishable', 'locality_type', 'category',
    'Weather_Rain', 'Weather_Heat', 'is_promotion',
    'days_since_launch', 'category_average_7d_demand'
]

# Encode categorical (godown_id and product_id)
print("Encoding godown_id and product_id...")
encoders = {}
for col in ['godown_id', 'product_id']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    features.append(col)

# Sort by date for chronological splitting
df.sort_values(by='date', inplace=True)

X = df[features]
y = df['daily_demand']

# Chronological split (last 10 days for testing to avoid data leakage)
unique_dates = df['date'].unique()
split_date = unique_dates[-10]

train_mask = df['date'] < split_date
test_mask = df['date'] >= split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training XGBoost model on {len(X_train)} samples (Test on {len(X_test)} samples)...")

# Adding L2 regularization (reg_lambda) and tree constraints to prevent overfitting
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    min_child_weight=3,
    reg_lambda=1.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Fix early_stopping_rounds deprecation warning by passing it to fit, or in recent xgboost versions use early_stopping_rounds in XGBRegressor init.
# But just eval_set is enough to show progress.
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=20)

print("Evaluating...")
preds = model.predict(X_test)
preds = np.maximum(0, preds)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
# Mask to avoid division by zero in MAPE
non_zero_mask = y_test > 0
mape = mean_absolute_percentage_error(y_test[non_zero_mask], preds[non_zero_mask])

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape*100:.2f}%")

# Save Metrics
metrics = {
    'rmse': round(float(rmse), 2),
    'mae': round(float(mae), 2),
    'mape': round(float(mape * 100), 2)
}
os.makedirs('models', exist_ok=True)
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save Feature Importances
importance_dict = model.get_booster().get_score(importance_type='gain')
sorted_importances = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
with open('models/feature_importance.json', 'w') as f:
    json.dump(sorted_importances, f)

print("Saving model and encoders...")
model.save_model('models/demand_forecast_model.json')

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model training complete.")
