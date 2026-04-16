import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv(
    r"C:\Users\hp\Downloads\SeoulBikeData.csv",
    encoding="latin1"
)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)

# Encode Seasons
le = LabelEncoder()
df['Seasons'] = le.fit_transform(df['Seasons'])

# Features
X = df[['TemperatureC',
        'Humidity',
        'Wind_speed_ms',
        'Seasons',
        'Hour']]

# Target
y = df['Rented_Bike_Count']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully")