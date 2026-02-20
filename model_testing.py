import torch

import numpy as np
import pandas as pd

import os
working_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import classification_report, confusion_matrix

from joblib import load

torch.manual_seed(67)
np.random.seed(67)
torch.cuda.manual_seed(67)

device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset_path = os.path.join(working_path, "csv_files", "SMS_test.csv")
df_test = pd.read_csv(test_dataset_path, encoding_errors="ignore")
df_test = df_test[["Message_body", "Label"]]

def text_preprocessing(text: pd.DataFrame, label: pd.DataFrame):
    tfidf = load(os.path.join(working_path, "pth_files", "tfidf.joblib"))
    text = text.str.lower()
    label = label.map({"Non-Spam": 0, "Spam": 1})
    
    text, label = text.to_numpy(), label.to_numpy()
    text = tfidf.transform(text).toarray()
    
    return text, label

X, y = text_preprocessing(df_test["Message_body"], df_test["Label"])

X = torch.tensor(X, dtype=torch.float32).to(device)

from nn import NN
model = NN(input_size=len(X[0]), hidden_size=32, num_classes=2).to(device)
model.load_state_dict(torch.load(os.path.join(working_path, "pth_files", "model.pt")))

with torch.no_grad():
    output = model(X)
    y_pred = output.argmax(dim=1)
    
y_pred = y_pred.cpu().numpy()

print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))