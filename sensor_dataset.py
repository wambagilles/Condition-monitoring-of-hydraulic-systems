from torch.utils.data import Dataset
import numpy as np
import math
import statistics



# Define Dataset class, for sensors data
class SensorDataset(Dataset):
    def __init__(self, pressure, volume_flow, target):
        self.pressure = pressure
        self.volume_flow = volume_flow
        self.target = target
        self.len = len(target)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        volume_flow = self.volume_flow.iloc[idx].to_numpy()
       
        pressure = self.pressure.iloc[idx].to_numpy()

        a_priori_vars = compute_apriori_vars(volume_flow, pressure)
        target = self.target.iloc[idx].values
        return np.array([volume_flow, pressure]), a_priori_vars, target
    
def compute_apriori_vars(volume_flow, pressure):
        vol_flow_mean = statistics.mean(volume_flow)
        vol_flow_median = statistics.median(volume_flow)
        vol_flow_variance = statistics.variance(volume_flow)
        vol_flow_stdev = statistics.stdev(volume_flow)

        pressure_mean = statistics.mean(pressure)
        pressure_median = statistics.median(pressure)
        pressure_variance = statistics.variance(pressure)
        pressure_stdev = statistics.stdev(pressure)

        a_priori_vars = np.array([vol_flow_mean, vol_flow_median, vol_flow_variance, vol_flow_stdev,
                                  pressure_mean, pressure_median, pressure_variance, pressure_stdev])
        return a_priori_vars