import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AQI Predictor", page_icon="🌍", layout="wide")

# --- LOAD DATA (Cached for speed) ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/aqi_training_data.csv') # Make sure this path is correct!
    df.columns = df.columns.str.replace('.', '_', regex=False)
    df.columns = df.columns.str.replace('-', '_', regex=False)
    
    # NEW: Convert the date column to actual dates and extract the Month name
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df['Month'] = df['last_updated'].dt.month_name()
    
    return df
df = load_data()

# --- SIDEBAR NAVIGATION ---ṇ
st.sidebar.title("Navigation")
st.sidebar.markdown("Research Project: Air Quality Prediction")
page = st.sidebar.radio("Go to:", [
    "🏠 Home (Abstract)", 
    "📅 Mode 1: Monthly Forecast", 
    "🎛️ Mode 2: Scenario Simulation", 
    "📡 Mode 3: Live API Prediction", 
    "📊 Data Dashboard",
    "🧠 Model Methodology"
])

# --- PAGE 1: HOME ---
if page == "🏠 Home (Abstract)":
    # Use a container to group the header visually
    with st.container():
        st.title("🌍 AQI Prediction Engine")
        st.markdown("### Next-Generation Meteorological Air Quality Forecasting")
        st.divider() # Adds a sleek horizontal line
    
    # Asymmetrical columns (The left is wider than the right)
    col_text, col_stats = st.columns([2, 1])
    
    with col_text:
        st.subheader("Research Abstract")
        st.write("""
        This engine investigates the non-linear meteorological factors affecting the Air Quality Index (AQI) across India. 
        By stripping away mathematically derived pollutants and focusing strictly on atmospheric physics (Temperature, Wind Dispersion, Pressure Systems), 
        this model achieves a highly realistic baseline for predicting PM2.5 accumulation.
        """)
        st.info(f"🟢 System Online. Securely loaded {len(df):,} training records.")
    
    with col_stats:
        st.subheader("Global Dataset Averages")
        # Placing metrics in a stylized box
        st.metric(label="Mean PM2.5 Concentration", value=f"{df['air_quality_PM2_5'].mean():.2f} µg/m³")
        st.metric(label="Mean PM10 Concentration", value=f"{df['air_quality_PM10'].mean():.2f} µg/m³")
        st.metric(label="Average Temperature", value=f"{df['temperature_celsius'].mean():.1f} °C")

# --- PAGE 2: MODE 1 (City + Month) ---
elif page == "📅 Mode 1: Monthly Forecast":
    st.title("Mode 1: Historical Monthly Forecasting")
    st.write("Predict the average PM2.5 for a specific city during a specific month based on historical weather patterns.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Sort cities alphabetically for a better user experience
        cities = sorted(df['location_name'].dropna().unique())
        selected_city = st.selectbox("Select City", cities)
    with col2:
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        selected_month = st.selectbox("Select Month", months)
        
    if st.button("Predict Average PM2.5"):
        import joblib
        
        # 1. Filter the dataset for the selected city and month
        city_month_data = df[(df['location_name'] == selected_city) & (df['Month'] == selected_month)]
        
        if city_month_data.empty:
            st.warning(f"⚠️ Not enough historical data for {selected_city} in {selected_month}. Try another combination.")
        else:
            # 2. Calculate the historical average weather for this city/month
            avg_temp = city_month_data['temperature_celsius'].mean()
            avg_wind = city_month_data['wind_kph'].mean()
            avg_humidity = city_month_data['humidity'].mean()
            avg_pressure = city_month_data['pressure_mb'].mean()
            avg_precip = city_month_data['precip_mm'].mean()
            avg_visibility = city_month_data['visibility_km'].mean()
            
            # 3. Load Model and Predict
            try:
                model = joblib.load('models/best_aqi_model.pkl')
                
                # Feed the historical averages into our ML model
                user_input = [[avg_temp, avg_wind, avg_humidity, avg_pressure, avg_precip, avg_visibility]]
                prediction = model.predict(user_input)[0]
                
                # 4. Display the Results Professionally
                st.success(f"Predicted PM2.5 for {selected_city} in {selected_month}: **{prediction:.2f} µg/m³**")
                
                # Show the user the data that drove this prediction (Model Explainability!)
                st.write("### 🔍 How did the model make this prediction?")
                st.write(f"The model used the historical weather averages for {selected_city} in {selected_month}:")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Avg Temp", f"{avg_temp:.1f} °C")
                metrics_col2.metric("Avg Wind", f"{avg_wind:.1f} kph")
                metrics_col3.metric("Avg Rain", f"{avg_precip:.2f} mm")
                
            except FileNotFoundError:
                st.error("Model file not found! Please run train_model.py first.")

# --- PAGE 3: MODE 2 (Manual Input) ---
elif page == "🎛️ Mode 2: Scenario Simulation":
    st.title("🎛️ Scenario Simulation")
    st.markdown("Dial in specific weather conditions to simulate the resulting PM2.5 accumulation.")
    st.divider()
    
    # Create two columns: Left for Inputs (Control Panel), Right for Output (Results)
    input_col, result_col = st.columns([1.5, 1])
    
    with input_col:
        st.subheader("Atmospheric Controls")
        # Group sliders inside a sleek UI container
        with st.container(border=True):
            temp = st.slider("🌡️ Temperature (°C)", -10.0, 50.0, 25.0)
            wind = st.slider("💨 Wind Speed (kph)", 0.0, 100.0, 15.0)
            humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
            
            # Hide the more complex variables in an expander to keep the UI clean!
            with st.expander("⚙️ Advanced Meteorological Settings"):
                pressure = st.slider("Pressure (mb)", 900.0, 1100.0, 1010.0)
                precip = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
                visibility = st.slider("Visibility (km)", 0.0, 20.0, 10.0)
                
            # A dedicated button spanning the column
            simulate_btn = st.button("Initialize Simulation 🚀", use_container_width=True)

    with result_col:
        st.subheader("Simulation Output")
        if simulate_btn:
            import joblib
            try:
                model = joblib.load('models/best_aqi_model.pkl')
                user_input = [[temp, wind, humidity, pressure, precip, visibility]]
                prediction = model.predict(user_input)[0]
                
                # Big, bold visual output
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; border-radius: 10px; background-color: rgba(0, 255, 209, 0.1); border: 2px solid #00FFD1;">
                    <h2 style="color: #E4E4E7; margin-bottom: 0px;">Predicted PM2.5</h2>
                    <h1 style="color: #00FFD1; font-size: 50px; margin-top: 0px;">{prediction:.1f} <span style="font-size: 20px;">µg/m³</span></h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Warning logic
                st.write("") # Spacer
                if prediction <= 50:
                    st.success("Status: Optimal atmospheric dispersion.")
                elif prediction <= 100:
                    st.warning("Status: Moderate accumulation detected.")
                else:
                    st.error("Status: Critical pollution accumulation hazard!")
                    
            except FileNotFoundError:
                st.error("System Error: Predictive model offline. Run train_model.py.")
        else:
            st.info("Awaiting parameters... Adjust the controls and initialize the simulation.")

# --- PAGE 4: MODE 3 (Live API) ---
elif page == "📡 Mode 3: Live API Prediction":
    st.title("Mode 3: Real-Time Prediction")
    st.write("Fetch live weather data from the internet and feed it into our trained ML model. Then, compare our model's prediction against the actual real-world PM2.5 sensors!")
    
    # 1. User Input for Live Prediction
    live_city = st.text_input("Enter a City Name (e.g., Mumbai, Delhi, London)", "Delhi")
    
    # ⚠️ REPLACE THIS WITH YOUR FREE WEATHERAPI KEY
    API_KEY = "323c1ed733ab430582c122347263103"
    
    if st.button("Fetch Live Data & Predict"):
        if API_KEY == "YOUR_API_KEY_HERE":
            st.error("🚨 Please paste your WeatherAPI key in the app.py code!")
        else:
            with st.spinner(f'Fetching live data for {live_city}...'):
                import joblib
                
                # 2. Call the Live Weather API (Asking for Air Quality data too!)
                url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={live_city}&aqi=yes"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 3. Extract the exact features our ML model needs
                    current_weather = data['current']
                    live_temp = current_weather['temp_c']
                    live_wind = current_weather['wind_kph']
                    live_humidity = current_weather['humidity']
                    live_pressure = current_weather['pressure_mb']
                    live_precip = current_weather['precip_mm']
                    live_visibility = current_weather['vis_km']
                    
                    # Extract the ACTUAL real-world PM2.5 for comparison
                    actual_pm25 = current_weather['air_quality']['pm2_5']
                    
                    # 4. Predict using OUR Machine Learning Model
                    try:
                        model = joblib.load('models/best_aqi_model.pkl')
                        user_input = [[live_temp, live_wind, live_humidity, live_pressure, live_precip, live_visibility]]
                        predicted_pm25 = model.predict(user_input)[0]
                        
                        st.success(f"Live data retrieved successfully for {data['location']['name']}, {data['location']['country']}!")
                        
                        # 5. Display the Scientific Comparison (Great for the paper)
                        st.write("### 🧪 Model vs. Reality Comparison")
                        
                        # Calculate how far off our model was (Absolute Error)
                        error_margin = predicted_pm25 - actual_pm25
                        
                        col1, col2 = st.columns(2)
                        
                        # Show Actual
                        col1.metric(
                            label="📡 Actual Real-World PM2.5", 
                            value=f"{actual_pm25:.2f} µg/m³",
                            help="This is the reading from physical IoT sensors right now."
                        )
                        
                        # Show Prediction with Delta
                        col2.metric(
                            label="🤖 Our Model's Prediction", 
                            value=f"{predicted_pm25:.2f} µg/m³",
                            delta=f"{error_margin:.2f} difference",
                            delta_color="inverse", # Red if we over-predicted, Green if under
                            help="This is what our Random Forest calculated based purely on the weather."
                        )
                        
                        # Show the Weather Conditions
                        st.write("---")
                        st.write("#### 🌦️ Current Weather Conditions Driving This Prediction:")
                        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                        w_col1.metric("Temperature", f"{live_temp} °C")
                        w_col2.metric("Wind", f"{live_wind} kph")
                        w_col3.metric("Humidity", f"{live_humidity} %")
                        w_col4.metric("Visibility", f"{live_visibility} km")
                        
                    except FileNotFoundError:
                        st.error("Model file not found! Please run train_model.py first.")
                else:
                    st.error(f"Could not find live data for '{live_city}'. Please check the spelling.")

# --- PAGE 5: DASHBOARD ---
elif page == "📊 Data Dashboard":
    st.title("Analytical Dashboard")
    st.write("Explore the underlying training data to understand the relationships between meteorology and air pollution.")
    
    # 1. Top 10 Most Polluted Cities (Bar Chart)
    st.subheader("1. Top 10 Cities by Average PM2.5")
    # Calculate average PM2.5 per city
    city_avg = df.groupby('location_name')['air_quality_PM2_5'].mean().reset_index()
    top_10_cities = city_avg.sort_values(by='air_quality_PM2_5', ascending=False).head(10)
    
    fig1 = px.bar(top_10_cities, x='location_name', y='air_quality_PM2_5', 
                  color='air_quality_PM2_5', color_continuous_scale='Reds',
                  labels={'location_name': 'City', 'air_quality_PM2_5': 'Average PM2.5 (µg/m³)'})
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("---")
    
    # 2. Wind vs. PM2.5 Scatter Plot (The Dispersion Effect)
    st.subheader("2. Meteorological Impact: Wind Dispersion Effect")
    st.write("Notice how higher wind speeds generally lead to lower PM2.5 concentrations because the wind disperses the pollutants.")
    
    # Taking a sample of 2000 points so the browser doesn't lag
    sample_df = df.sample(2000, random_state=42) if len(df) > 2000 else df
    
    fig2 = px.scatter(sample_df, x="wind_kph", y="air_quality_PM2_5", 
                     color="air_quality_PM2_5", color_continuous_scale='Turbo',
                     opacity=0.6, hover_name="location_name",
                     labels={'wind_kph': 'Wind Speed (kph)', 'air_quality_PM2_5': 'PM2.5 Level'})
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("---")

    # 3. Correlation Matrix (Highly Academic!)
    st.subheader("3. Feature Correlation Matrix")
    st.write("This heatmap shows the mathematical correlation (Pearson) between all variables. Negative numbers (blue) mean as one goes up, the other goes down.")
    
    # Select only the numeric columns that matter for our model
    numeric_cols = ['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'precip_mm', 'visibility_km', 'air_quality_PM2_5']
    corr_matrix = df[numeric_cols].corr()
    
    fig3 = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                     color_continuous_scale='RdBu_r', origin='lower')
    st.plotly_chart(fig3, use_container_width=True)

# --- PAGE 6: METHODOLOGY ---
elif page == "🧠 Model Methodology":
    st.title("Model Methodology & Evaluation")
    st.write("This section details the scientific approach, feature selection, and evaluation of our Machine Learning pipeline.")
    
    st.header("1. Addressing Target Leakage")
    st.info("""
    **The Problem:** Initial models predicting the overall AQI (EPA Index) using pollutant features (PM2.5, PM10) achieved an artificial R² score of 1.0. This was identified as **Target Leakage**, because the EPA Index is mathematically derived directly from those exact pollutants. 
    
    **The Solution:** To create a genuine predictive tool, the model architecture was restructured to predict raw **PM2.5 concentrations** using *strictly meteorological data* (Temperature, Wind, Humidity, Pressure, Precipitation, and Visibility).
    """)
    
    st.header("2. Model Architecture & Metrics")
    st.write("We implemented a **Random Forest Regressor** (an ensemble learning method) to capture the non-linear relationships between weather patterns and particulate matter accumulation.")
    
    # Displaying your actual training metrics!
    col1, col2, col3 = st.columns(3)
    col1.metric("R-Squared (Accuracy)", "0.5236", help="52.36% of PM2.5 variance is explained purely by weather.")
    col2.metric("RMSE", "65.98 µg/m³", help="Root Mean Squared Error")
    col3.metric("MAE", "40.28 µg/m³", help="Mean Absolute Error")
    
    st.write("*Note: The unexplained variance (~48%) is attributed to anthropogenic factors not present in the dataset (e.g., traffic, industrial emissions, crop burning).*")
    
    st.write("---")
    
    st.header("3. Feature Importance (Explainable AI)")
    st.write("What weather condition has the biggest impact on PM2.5? We extracted the decision weights directly from the Random Forest model to find out:")
    
    import joblib
    try:
        # Load the trained model to extract its internal feature importances
        model = joblib.load('models/best_aqi_model.pkl')
        
        # These must match the exact order trained in train_model.py
        feature_names = ['Temperature', 'Wind Speed', 'Humidity', 'Pressure', 'Precipitation', 'Visibility']
        importances = model.feature_importances_
        
        # Create a dataframe for Plotly
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        # Plot the Feature Importance
        fig4 = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Viridis',
                      title="Random Forest Feature Importance Weights")
        st.plotly_chart(fig4, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Model file not found! Please run train_model.py first.")