import pandas as pd

def preprocess(df_train,df_test):
    data_train = df_train.copy()
    data_test = df_test.copy()

    #######
    # Format
    #######

    # Format time
    data_train['time'] = pd.to_datetime(data_train['time'])
    data_test['time'] = pd.to_datetime(data_test['time'])

    ####### 
    # Replace vesselID and portID with integers 
    #######

    # Factorize the 'vesselID' column in X_train and get the integer IDs and the mapping
    vesselID, vesselID_mapping = pd.factorize(data_train['vesselId'])

    # Replace 'vessel_ID' column in X_train with integer IDs
    data_train['vesselId'] = vesselID

    # Create a dictionary from the mapping to apply the same to X_test
    vessel_to_ID = {vessel: idx for idx, vessel in enumerate(vesselID_mapping)}

    # Replace 'vesselID' in X_test using the same mapping from X_train
    data_test['vesselId'] = data_test['vesselId'].map(vessel_to_ID)
    
    # Replace 'portId' column with integer IDs
    data_train['portId'] = pd.factorize(data_train['portId'])[0]

    ####### 
    # Remove outliers 
    #######

    # Remove sog outliers
    data_train = data_train[data_train['sog'] <= 40]

    return data_train, data_test

def feature_engineering(df_train,df_test):  
    data_train = df_train.copy()
    data_test = df_test.copy()

    #######
    # Sort
    #######

    # Sort by vesselID then time
    data_train = data_train.sort_values(['vesselId','time'])

    ####### 
    # Create lag features
    #######

    # Create lagged columns for longitude and latitude
    data_train['longitude_lag1'] = data_train['longitude'].shift(1)
    data_train['latitude_lag1'] = data_train['latitude'].shift(1)

    # Insert longitude and latitude lag before longitude and latitude
    data_train.insert(5, 'longitude_lag1', data_train.pop('longitude_lag1'))
    data_train.insert(5, 'latitude_lag1', data_train.pop('latitude_lag1'))

    ####### 
    # One hot encoding (ish)
    #######

    # New feature for if the vessel is moored or not
    data_train['not_under_way'] = data_train['navstat'].apply(lambda x: 1 if x == 5 or x == 1 else 0)
    data_train['under_way'] = data_train['navstat'].apply(lambda x: 1 if x == 0 or x == 8 else 0)
    #data_train.insert(5, 'is_moored', data_train.pop('is_moored'))

    ####### 
    # Create calendar features 
    #######

    # Extract calendar features for 'etaRaw'
    data_train[['etaMonth', 'etaDay', 'etaHour', 'etaMinute']] = data_train['etaRaw'].str.extract(r'(\d{2})-(\d{2}) (\d{2}):(\d{2})')

    # Extract calendar features for 'time'
    data_train[['timeYear', 'timeMonth', 'timeDay', 'timeHour', 'timeMinute', 'timeSecond']] = data_train['time'].str.extract(r'(\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')

    # Convert objects to integers
    data_train[['etaMonth', 'etaDay', 'etaHour', 'etaMinute', 'timeYear', 'timeMonth', 'timeDay', 'timeHour', 'timeMinute', 'timeSecond']] = data_train[['etaMonth', 'etaDay', 'etaHour', 'etaMinute', 'timeYear', 'timeMonth', 'timeDay', 'timeHour', 'timeMinute', 'timeSecond']].astype(int)

    # Drop time and etaRaw columns
    data_train.drop(columns=['time', 'etaRaw'], inplace=True)

    # Extract calendar features for the test data set 'time'
    data_test[['timeYear', 'timeMonth', 'timeDay', 'timeHour', 'timeMinute', 'timeSecond']] = data_test['time'].str.extract(r'(\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')

    # Drop time column
    data_test.drop(columns=['time'], inplace=True)

    return data_train, data_test