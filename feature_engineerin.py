import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load the datasets
train = pd.read_csv('data/datasets/ais_train.csv', sep='|')
ports = pd.read_csv('data/datasets/ports.csv', sep='|')

# Function to redefine coordinates
def redefine_coordinates(df):
    df['longitude'] = df['longitude'].apply(lambda x: x if x >= 0 else x + 360)
    df['latitude'] = df['latitude'].apply(lambda x: x if x >= 0 else x + 180)
    return df

# Function to handle missing headings
def missing_heading(df):
    df['heading'] = df['heading'].replace(511, np.nan)
    return df

# Function to change time attributes
def change_time_attr(df):
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second
    return df

# Function to add lagged features for latitude and longitude
def add_lat_lon_lag(df, lag_steps=3):
    df = df.sort_values(by=['vesselId', 'time'])
    for vessel_id, group in df.groupby('vesselId'):
        for lag in range(1, lag_steps + 1):
            df.loc[group.index, f'latitude_lag_{lag}'] = group['latitude'].shift(lag)
            df.loc[group.index, f'longitude_lag_{lag}'] = group['longitude'].shift(lag)
    return df

# Function to indicate whether a vessel is underway
def under_way(df): 
    df['under_way'] = df['navstat'].isin([0, 8]).astype(int)
    return df 

# Function to calculate bearing to port
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

# Function to add bearing to port
def add_bearing_to_port(df):
    df['bearing_to_port'] = df.apply(lambda row: calculate_bearing(row['latitude'], row['longitude'], row['latitude_port'], row['longitude_port']), axis=1)
    df['correct_direction'] = df['heading'] - df['bearing_to_port']  
    return df

# Function to convert lat/lon to Cartesian coordinates
def lat_lon_to_cartesian(df):
    df['latitude_rad'] = np.radians(df['latitude'])
    df['longitude_rad'] = np.radians(df['longitude'])
    R = 6371.0
    df['x'] = R * np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
    df['y'] = R * np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
    df['z'] = R * np.sin(df['latitude_rad'])
    df = df.drop(columns=['latitude', 'longitude'])
    return df

def factorize_ids(df):
    # Factorize 'vesselId' and get the integer IDs and the mapping
    vesselID, vesselID_mapping = pd.factorize(df['vesselId'])
    df['vesselId'] = vesselID  # Replace 'vesselId' column in the training data

    # Factorize 'portId' and replace with integer IDs
    df['portId'] = pd.factorize(df['portId'])[0]  # Replace 'portId' column with integer IDs

    return df

# Apply all transformations to the training data
train = redefine_coordinates(train)
train = missing_heading(train)
train = change_time_attr(train)
train = add_lat_lon_lag(train)
train = under_way(train)
train = train.merge(ports[['portId', 'latitude', 'longitude']], how='left', on='portId', suffixes=('', '_port'))
train = add_bearing_to_port(train)
train = lat_lon_to_cartesian(train)
train = factorize_ids(train)


# Include 'vesselId' in the features
X = train.drop(columns=['latitude', 'longitude', 'time', 'VesselID'])  # Keep 'vesselId' in the features
y_lat = train['latitude']  # Target for latitude
y_lon = train['longitude']  # Target for longitude

# Splitting the dataset
X_train, X_test, y_train_lat, y_test_lat = train_test_split(X, y_lat, test_size=0.2, random_state=42)
X_train, X_test, y_train_lon, y_test_lon = train_test_split(X, y_lon, test_size=0.2, random_state=42)


# Training the model for latitude
lat_model = xgb.XGBRegressor(random_state=42)
lat_model.fit(X_train, y_train_lat)

# Training the model for longitude
lon_model = xgb.XGBRegressor(random_state=42)
lon_model.fit(X_train, y_train_lon)
# Training the model for latitude
lat_model = xgb.XGBRegressor(random_state=42)
lat_model.fit(X_train, y_train_lat)

# Training the model for longitude
lon_model = xgb.XGBRegressor(random_state=42)
lon_model.fit(X_train, y_train_lon)

# Function to make predictions
def make_predictions(new_data):
    new_data = redefine_coordinates(new_data)
    new_data = change_time_attr(new_data)
    new_data = add_lat_lon_lag(new_data)
    new_data = under_way(new_data)
    new_data = new_data.merge(ports[['portId', 'latitude', 'longitude']], how='left', on='portId', suffixes=('', '_port'))
    new_data = add_bearing_to_port(new_data)
    new_data = lat_lon_to_cartesian(new_data)

    X_new = new_data.drop(columns=['latitude', 'longitude', 'time', 'vesselId', 'ID'])  # Drop irrelevant columns
    predicted_lat = lat_model.predict(X_new)
    predicted_lon = lon_model.predict(X_new)

    new_data['predicted_latitude'] = predicted_lat
    new_data['predicted_longitude'] = predicted_lon
    return new_data


new_data = pd.read_csv('ais_train.csv')  # Load your new data here
predictions = make_predictions(new_data)
print(predictions[['ID', 'predicted_latitude', 'predicted_longitude']])
