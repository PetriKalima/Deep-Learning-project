import pandas as pd
import numpy as np
import torch
from mlp import MLP
from sklearn.preprocessing import StandardScaler


def main():
    device = torch.device("cpu")
    traindata = pd.read_csv('train.csv', delimiter=',')
    traintarget = traindata["SalePrice"]
    traindata_x = traindata.drop(columns="SalePrice")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(traindata_x)

    mlp = MLP()
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005)
    n_epochs = 1000





if __name__ == "__main__":
    main()