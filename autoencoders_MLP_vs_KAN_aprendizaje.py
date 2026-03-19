# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:13:30 2026

@author: Propietario
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from PIL import Image
import io

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# 1. Create dataset
X_np, y_np = make_moons(n_samples=400, noise=0.15, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

# 2. Define MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 3. Define Simple KAN (Basis expansion approach)
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=3):
        super(KANLayer, self).__init__()
        self.grid_size = grid_size
        self.base_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)

    def forward(self, x):
        base_output = torch.matmul(x, self.base_weight.t())
        # Using [x, sin(x), cos(x)] as basis functions
        basis = torch.stack([x, torch.sin(x), torch.cos(x)], dim=-1) # (batch, in_dim, 3)
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        return base_output + spline_output

class SimpleKAN(nn.Module):
    def __init__(self):
        super(SimpleKAN, self).__init__()
        self.layer1 = KANLayer(2, 4, grid_size=3)
        self.layer2 = KANLayer(4, 1, grid_size=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return self.sigmoid(x)

# 5. Helper to create a frame
def get_frame(mlp, kan, epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        p1 = mlp(grid).reshape(xx.shape).numpy()
        p2 = kan(grid).reshape(xx.shape).numpy()
    
    # MLP Plot
    ax1.contourf(xx, yy, p1, levels=20, cmap='RdBu', alpha=0.4)
    ax1.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], c='red', label='Clase Roja', edgecolors='k')
    ax1.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], c='blue', label='Clase Azul', edgecolors='k')
    ax1.set_title(f"MLP (Época {epoch})")
    ax1.legend()
    
    # KAN Plot
    ax2.contourf(xx, yy, p2, levels=20, cmap='RdBu', alpha=0.4)
    ax2.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], c='red', label='Clase Roja', edgecolors='k')
    ax2.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], c='blue', label='Clase Azul', edgecolors='k')
    ax2.set_title(f"KAN (Época {epoch})")
    ax2.legend()
    
    plt.tight_layout()
    
    # Save fig to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# 6. Training Loop
mlp = MLP()
kan = SimpleKAN()
opt_mlp = optim.Adam(mlp.parameters(), lr=0.02)
opt_kan = optim.Adam(kan.parameters(), lr=0.02)
criterion = nn.BCELoss()

frames = []
for epoch in range(1, 401):
    # MLP step
    opt_mlp.zero_grad()
    loss_mlp = criterion(mlp(X), y)
    loss_mlp.backward()
    opt_mlp.step()
    
    # KAN step
    opt_kan.zero_grad()
    loss_kan = criterion(kan(X), y)
    loss_kan.backward()
    opt_kan.step()
    
    if epoch % 20 == 0:
        frames.append(get_frame(mlp, kan, epoch))

# Save GIF
frames[0].save('entrenamiento_kan_vs_mlp.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)

import os

# Al final del código...
print(f"¡Proceso finalizado! El archivo se ha guardado en: {os.getcwd()}")

# Si quieres intentar abrirlo automáticamente (en Windows):
os.startfile('entrenamiento_kan_vs_mlp.gif')
