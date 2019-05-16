import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
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
    x_test = pd.read_csv('test_house.csv', delimiter=',')
    x_test = pd.get_dummies(x_test)
    x_test = x_test.fillna(0)
    valuestoremove = list(set(traindata_x.columns) - set(x_test.columns))
    for value in valuestoremove:
        traindata_x = traindata_x.drop(columns=value)
    print(traindata_x.shape)    
    scaler = StandardScaler()
    #x_train_scaled = scaler.fit_transform(traindata_x)
    #print(pd.get_dummies(traindata_x))
    if '--load' in sys.argv:
        txtfile = sys.argv[sys.argv.index('--load') + 1]
        hyperparameters = []
        accuracies = []
        i = 0
        with open(txtfile, 'r') as filetoread:
            for line in filetoread:
                if i % 2 == 0:
                    line = line.strip('\n')
                    hp = line.split(' ')
                    hyperparameters.append(hp[:3])
                else:
                    line = line.strip('\n')    
                    accuracies.append(line)
                i += 1
        maxacc = max(accuracies)
        index = accuracies.index(maxacc)
        hp_best = hyperparameters[index]
        print(hp_best)
        mlp = MLP(271, int(hp_best[0]), int(hp_best[1]), 1)
        mlp.to(device)
        x_tensor_new = torch.tensor(x_test.values, dtype=torch.float32)
        mlp, acc, outputs, loss = train(2500, traindata_x, traintarget, mlp, float(hp_best[2]))
        print(acc)
        
        outputs = mlp.forward(x_tensor_new)
        with open("submission.csv", 'w') as csvfile:
            csvfile.write("Id,SalePrice")
            i = 0
            values = outputs.detach().numpy()
            for value in values:
                value = str(value[0]).replace(',', '.')
                csvfile.write(str(1461+i) + ',' + str(value) + '\n')
                i += 1                                  
    else:
        hyperparameters, accuracies = HPSearch(traindata_x, traintarget, device)    
    

     

    
def HPSearch(traindata_x, traintarget, device):
    hyperparameters = []
    accuracies = []
    n = 100
    range1 = [100, 250]
    range2 = [10, 100]
    lrate_range = [np.log(0.01), np.log(0.05)]
    for (size1, size2, lrate) in random_search(n, range1, range2, lrate_range):
        size1, size2 = int(size1), int(size2)
        lrate = np.exp(lrate)
        hyperparameters.append([size1, size2, lrate])
        print("HP ", hyperparameters[-1]) 
        mlp = MLP(289, size1, size2, 1)
        mlp.to(device)
        mlp, acc, outputs, loss = train(500, traindata_x, traintarget, mlp, lrate)
        print("acc ", acc)
        accuracies.append(acc)

    with open("results.txt", 'w') as txtfile:
        for i in range(len(hyperparameters)):
            for item in hyperparameters[i]:
                txtfile.write(str(item) + ' ')
            txtfile.write('\n')    
            txtfile.write(str(accuracies[i]) + '\n') 
    return hyperparameters, accuracies        

def train(n, traindata_x, traintarget, mlp, lrate):
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lrate)
    n_epochs = n
    for epoch in range(n_epochs):
        x_tensor = torch.tensor(traindata_x.values, dtype=torch.float32)
        y_tensor = torch.tensor(traintarget, dtype=torch.float32)
        y_tensor = y_tensor.reshape(1460, 1)    
        optimizer.zero_grad()
        outputs = mlp.forward(x_tensor)
        loss = F.mse_loss(outputs, y_tensor)
        loss.backward()
        optimizer.step()    
    print(outputs, y_tensor)    
    print(loss)
    acc = 100 - np.mean(np.abs((y_tensor.detach().numpy() - outputs.detach().numpy()) / y_tensor.detach().numpy())) * 100
    return mlp, acc, outputs, loss

def random_search(n, *params):
    values = []
    for i in range(n):
        value = []
        for j in range(len(params)):
            num = np.random.uniform(params[j][0], params[j][1])
            value.append(num)
        values.append(value)
    return values 

if __name__ == "__main__":
    main()