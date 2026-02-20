import pandas as pd
import numpy as np

import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import dump

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(67)
np.random.seed(67)
torch.cuda.manual_seed(67)

working_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
train_dataset_path = os.path.join(working_path, "csv_files", "SMS_train.csv")
pth_files_path = os.path.join(working_path, "pth_files")

df = pd.read_csv(train_dataset_path, encoding_errors="ignore")
df = df[["Message_body", "Label"]]

def text_preprocessing(text: pd.DataFrame, label: pd.DataFrame):
    tfidf = TfidfVectorizer()
    text = text.str.lower()
    label = label.map({"Non-Spam": 0, "Spam": 1})
    
    text, label = text.to_numpy(), label.to_numpy()
    text = tfidf.fit_transform(text).toarray()
    
    return text, label, tfidf

X, y, tfidf = text_preprocessing(df["Message_body"], df["Label"])

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)

# set class weight
class_count = torch.bincount(y)
class_weight = 1.0 / class_count[y]
sampler = WeightedRandomSampler(class_weight, num_samples=len(class_weight), replacement=True)

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

from nn import NN
model = NN(input_size=len(X[0]), hidden_size=32, num_classes=len(class_count)).to(device)
model.train()

# hyperparameter
lr = 0.01
wd = 0.01
epochs = 100

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

ep_each_loss = []
for ep in range(epochs):
    ep_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        ep_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    if (ep + 1) % 10 == 0:
        print(f"Epoch: {ep + 1}, Loss: {ep_loss / len(train_loader):.4f}")
        
    ep_each_loss.append(ep_loss / len(train_loader))
    
        
print(f"Final loss: {(ep_loss / len(train_loader)):.4f}")
    
torch.save(model.state_dict(), os.path.join(pth_files_path, "model.pt"))
dump(tfidf, os.path.join(pth_files_path, "tfidf.joblib"))

def graph_loss(ep_each_loss):
    import matplotlib.pyplot as plt
    
    plt.plot(ep_each_loss, "-", color="blue", label="Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epochs")
    plt.grid(True)
    plt.savefig(os.path.join(working_path, "images", "train_loss.jpg"), dpi=300, bbox_inches="tight")
    plt.close()
    
graph_loss(ep_each_loss)

