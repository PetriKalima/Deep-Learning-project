import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR


def main():
    device = torch.device("cpu")
    traindata = pd.read_csv('train_house.csv', delimiter=',')
    traindata = pd.get_dummies(traindata)
    traindata = traindata.fillna(0)
    traintarget = traindata["SalePrice"]
    traindata_x = traindata.drop(columns="SalePrice")
    scaler = StandardScaler()
    #x_train_scaled = scaler.fit_transform(traindata_x)
    #print(pd.get_dummies(traindata_x))
    mlp = MLP(289, 100, 50, 1)
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005)
    n_epochs = 1000

    train_accuracy_history = []
    test_accuracy_history = []
    
    for epoch in range(n_epochs):
        x_tensor = torch.tensor(traindata_x.values, dtype=torch.float32)
        y_tensor = torch.tensor(traintarget, dtype=torch.float32)
        y_tensor = y_tensor.reshape(1460, 1)    
        optimizer.zero_grad()
        outputs = mlp.forward(x_tensor)
        loss = F.mse_loss(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    print(y_tensor.numpy())
    acc = np.mean(np.abs((y_tensor.detach().numpy() - outputs.detach().numpy()) / y_tensor.detach().numpy())) * 100    
    print(outputs, y_tensor)    
    print(loss)
    print(acc)

    
     
                    


if __name__ == "__main__":
    main()