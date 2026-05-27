import pandas as pd

# 1. Load the dataset
df = pd.read_csv('aqi_training_data.csv')

# 2. Fix the column names by replacing periods and hyphens with underscores
df.columns = df.columns.str.replace('.', '_', regex=False)
df.columns = df.columns.str.replace('-', '_', regex=False)

# Now your column is safely named "air_quality_PM2_5"
print(df.columns.tolist())

# 3. Save the cleaned data for your ML models
df.to_csv('cleaned_aqi_data.csv', index=False)