import pandas as pd
import numpy as np
import sys
import math

from tqdm import tqdm
import math
import gc

from sklearn.preprocessing import LabelEncoder

# Function importing Dataset
def importdata(): 
    if len(sys.argv) != 2:
        sys.exit("Usage: python feature_engineering.py data/data/Train.csv")

    data = pd.read_csv(sys.argv[1], sep= ',', header = 0) 
      
    # Printing the dataset shape 
    print ("Dataset Length: ", len(data)) 
    print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset observations 
    print ("Dataset: \n",data.head(2)) 
    return data 


# covert features  from string to List of values 
def replace_nan(x):
    if x==" ":
        return np.nan
    else :
        return float(x)


def aggregate_features(x,col_name):
    x["max_"+col_name]=x[col_name].apply(np.max)
    x["min_"+col_name]=x[col_name].apply(np.min)
    x["mean_"+col_name]=x[col_name].apply(np.mean) #average
    x["std_"+col_name]=x[col_name].apply(np.std)
    x["var_"+col_name]=x[col_name].apply(np.var)
    x["median_"+col_name]=x[col_name].apply(np.median) # separe higher from lower values
    x["ptp_"+col_name]=x[col_name].apply(np.ptp) # range of values peak to peak max -min
    return x  

def remove_nan_values(x):
    return [e for e in x if not math.isnan(e)]

    
def encode_location(data):
    enc = LabelEncoder()
    enc.fit(data["location"])
    data["location"]=enc.transform(data["location"])
    return data

#'Unnamed: 0' and Quakers
def drop_column(df:pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df.drop([col_name], axis=1)
    return df

def fill_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    altitude_low_meters_mean=1500.3684210526317
    altitude_high_meters_mean=1505.6315789473683
    altitude_mean_log_mean=7.0571530664031155
    df["altitude_low_meters"] = df["altitude_low_meters"].fillna(altitude_low_meters_mean)
    df["altitude_high_meters"] = df["altitude_high_meters"].fillna(altitude_high_meters_mean)
    df["altitude_mean_log"] = df["altitude_mean_log"].fillna(altitude_mean_log_mean)
    return df

if __name__=="__main__": 
    data = importdata() 
    features=["temp","precip","rel_humidity","wind_dir","wind_spd","atmos_press"]
    for feature in features : 
        data[feature]=data[feature].apply(lambda x: [ replace_nan(X) for X in x.replace("nan"," ").split(",")])

    print(f'removing nan values...')
    for col_name in tqdm(features):
        data[col_name]=data[col_name].apply(remove_nan_values)

    print('aggregate features...')
    for col_name in tqdm(features):
        data=aggregate_features(data,col_name)

    data = data.drop(features, axis=1)
    data = data.drop(['ID'], axis=1)
    print('encoded location')

    data = encode_location(data)
    
    print(f'select location D')
    data = data[(data['location']== 3)]
    print(f'----'*15)
    print(f'Data: {data.shape}, saved in: data/data_prep_reg.csv ')
    print(f'data columns: {data.columns}')
    data.to_csv('data/data_prep_reg.csv')
    del data
