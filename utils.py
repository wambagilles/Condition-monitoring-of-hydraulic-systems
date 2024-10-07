import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sensor_dataset import SensorDataset
import json
import numpy as np

def read_and_parse_dataset():
    target_colnames = ["cooler_condition", "valve_condition", "internal_pump_leakage", "hydraulic_accumulator", "stable"]
    target = pd.read_csv("data_subset/profile.txt", names=target_colnames, sep='\t')[['valve_condition']]
    # Frame the problem as classification
    target['valve_condition'] = target['valve_condition'].apply(lambda x: map_optim(x))
    
    encoder = OneHotEncoder(categories = 'auto')
    encoded_target = encoder.fit_transform(
        target['valve_condition'].values.reshape(-1,1)).toarray()
    target = pd.DataFrame(encoded_target)
    target.columns = ['not_optimal','optimal',]


    pressure = pd.read_csv("data_subset/ps2.txt", header=None, sep='\t')
    # Reduce the sample rate from 100 to 10
    pressure = pd.DataFrame(pressure.apply(downscale_sample_rate, axis=1).tolist())
    volume_flow = pd.read_csv("data_subset/fs1.txt", header=None, sep='\t')

    train_pressure, test_pressure = train_test_split(pressure, train_size=2000, shuffle=False)
    train_volume_flow, test_volume_flow = train_test_split(volume_flow, train_size=2000, shuffle=False)
    train_target, test_target = train_test_split(target, train_size=2000, shuffle=False)

    train_dataset = SensorDataset(train_pressure, train_volume_flow, train_target)
    test_dataset = SensorDataset(test_pressure, test_volume_flow, test_target)
	
    return train_dataset, test_dataset

def map_optim(x):
    if x == 100:
        return 1
    return 0
    



# Function to aggregate every 10 consecutive values
def downscale_sample_rate(row):
    reshaped = row.values.reshape(-1, 10)  # Reshape each row into 600x10
    return reshaped.mean(axis=1)  # Compute the mean of every 10 values



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)