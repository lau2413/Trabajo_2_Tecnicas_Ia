# src/data.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
import torch
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

path = kagglehub.dataset_download("datamunge/sign-language-mnist")
csv_path = os.path.join(path, "sign_mnist_train.csv")
train = pd.read_csv(csv_path)

# visualización de una muestra del dataset
X=train.drop('label',axis=1)
y=train['label'].values
i=np.random.randint(len(X))
img=X.iloc[i].values.reshape(28,28)
plt.imshow(img,cmap='gray')
plt.title(y[i])

# Se Normalizan los datos
X_values = X.values / 255.0
Y_values = y

# Reshape a (1, 28, 28) 
X_tensor = torch.tensor(X_values, dtype=torch.float32).reshape(-1, 1, 28, 28)
Y_tensor  = torch.tensor(Y_values, dtype=torch.long)

# Clase Dataset personalizada 
class SignDataset(Dataset):
    def __init__(self, X, y):
        # X ya llega como tensor (N, 1, 28, 28), y como tensor (N,)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#  DataLoaders con K-Fold
def obtener_dataloaders(batch_size=64, n_splits=5):
    dataset_completo = SignDataset(X_tensor, Y_tensor)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tensor)):
        train_subset = Subset(dataset_completo, train_idx)
        val_subset   = Subset(dataset_completo, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,          
            num_workers=0,
            pin_memory=True        
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,         
            num_workers=0,
            pin_memory=True
        )

        print(f"Fold {fold+1}/{n_splits}"
              f"Train: {len(train_subset)} muestras"
              f"Val: {len(val_subset)} muestras")

        yield train_loader, val_loader


for fold, (train_loader, val_loader) in enumerate(obtener_dataloaders(batch_size=64, n_splits=5)):
    imgs, labels = next(iter(train_loader))
    print(f"  Lote de ejemplo numero imgs: {imgs.shape}, labels: {labels.shape}")
    break 