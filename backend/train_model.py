import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_service"))
from model_def import MoonClassifier

# --- Data ---
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_ds     = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# --- Model, loss, optimizer ---
model     = MoonClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training loop ---
EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:3d}/{EPOCHS} — loss: {avg_loss:.4f}")

# --- Test accuracy ---
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    accuracy = (preds == y_test_t).float().mean().item()
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# --- Save ---
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
