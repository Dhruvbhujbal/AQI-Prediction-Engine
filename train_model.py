import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

print("🚀 Starting Model Training Process...")

# 1. Load the dataset
df = pd.read_csv('data/aqi_training_data.csv')
df.columns = df.columns.str.replace('.', '_', regex=False)
df.columns = df.columns.str.replace('-', '_', regex=False)

# 2. THE FIX: True Predictive Features
# We only use weather conditions, NOT other pollutants, to predict PM2.5
features = [
    'temperature_celsius', 
    'wind_kph', 
    'humidity', 
    'pressure_mb',
    'precip_mm',
    'visibility_km'
]

# We are now predicting the actual PM2.5 concentration!
target = 'air_quality_PM2_5'

X = df[features]
y = df[target]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

# 4. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("⚙️ Training Random Forest Model... (This will take a bit longer now)")
model.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\n📈 --- REALISTIC MODEL EVALUATION ---")
print(f"R-Squared (Accuracy): {r2:.4f} (A score between 0.40 and 0.85 is normal here!)")
print(f"RMSE (Error Margin):  {rmse:.4f} µg/m³")
print(f"MAE (Avg Error):      {mae:.4f} µg/m³")
print("------------------------------------\n")

# 6. Save the fixed model
joblib.dump(model, 'models/best_aqi_model.pkl')
print("✅ Model successfully saved to 'models/best_aqi_model.pkl'!")