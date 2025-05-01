# shigella_resistance_predictor.py (V2 - with AUC single-class fix)
# Deep learning model for predicting antibiotic resistance of XDR-Shigella from genotype data

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --------- Load Input Data ---------
genotype_file = "./XDR_Shigella_AI_Predictor/data/genotype_matrix.csv"
resistance_file = "./XDR_Shigella_AI_Predictor/data/resistance_labels.csv"

X = pd.read_csv(genotype_file, index_col=0).values
y = pd.read_csv(resistance_file, index_col=0).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# --------- Define Model ---------
class ResistancePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResistancePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# --------- Train Model ---------
model = ResistancePredictor(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# --------- Evaluate AUC and Save Results ---------
model.eval()
preds = model(X_test).detach().numpy()
y_true = y_test.numpy()

aucs = []
with open("auc_scores.txt", "w") as f:
    for i in range(y_true.shape[1]):
        # Check if y_true for this antibiotic has only one class
        if len(np.unique(y_true[:, i])) < 2:
            result = f"Antibiotic {i+1} AUC: Cannot compute (only one class in test set)"
            print(result)
            f.write(result + "\n")
            continue

        auc = roc_auc_score(y_true[:, i], preds[:, i])
        aucs.append(auc)
        result = f"Antibiotic {i+1} AUC: {auc:.3f}"
        print(result)
        f.write(result + "\n")

    if aucs:
        f.write(f"Average AUC: {np.mean(aucs):.3f}\n")

# --------- Save Model ---------
torch.save(model.state_dict(), "trained_model.pth")

# --------- Plot ROC AUC per Antibiotic ---------
if aucs:
    plt.bar(range(1, len(aucs)+1), aucs)
    plt.xlabel("Antibiotic")
    plt.ylabel("AUC")
    plt.title("AUC per Antibiotic (XDR Shigella)")
    plt.savefig("auc_plot.png")
    plt.show()

print("Training complete. Model, AUC scores and ROC plot have been saved.")


