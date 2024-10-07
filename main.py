import argparse
from model import AutoLagNet
from train import train
from utils import read_and_parse_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm as tqdm


def main():
    parser = argparse.ArgumentParser(description="Script for the technical exercise")

    args = parser.parse_args()
    device = 'cpu'
    n_var_in = 2 #Multivaluate time series Pressure and volume_flow
    n_neurons_out = 2
    n_var_a_priori = 8

    train_dataset, test_dataset = read_and_parse_dataset() 

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    
    model = AutoLagNet( n_var_in, n_neurons_out, n_var_a_priori)
    model.to(device)
    
    train(model, train_loader, test_loader)

 


if __name__ == '__main__':
    main()
