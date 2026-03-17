import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Importamos sklearn para acceder al servidor alternativo fiable (OpenML)
from sklearn.datasets import fetch_openml

# ==========================================
# 1. CARGA DE DATOS: KMNIST (Vía Servidor Alternativo)
# ==========================================
class KMNIST_Alternativo(Dataset):
    def __init__(self, train=True):
        # 1. Descargamos desde OpenML (mucho más estable que el servidor japonés original)
        X, y = fetch_openml('Kuzushiji-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
        
        # 2. Damos formato: Normalizamos a [0, 1] y redimensionamos a (Canal, Alto, Ancho)
        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, 1, 28, 28)
        y = y.astype(np.int64)
        
        # 3. Separamos 60.000 para entrenar y 10.000 para testear
        if train:
            self.X = torch.tensor(X[:60000])
            self.y = torch.tensor(y[:60000])
        else:
            self.X = torch.tensor(X[60000:])
            self.y = torch.tensor(y[60000:])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print("Conectando a servidor alternativo... (Esto puede tardar 1 o 2 minutos la primera vez)")
trainset = KMNIST_Alternativo(train=True)
testset = KMNIST_Alternativo(train=False)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# ==========================================
# 2. DEFINICIÓN DE LA ARQUITECTURA (AE Latente = 16)
# ==========================================
class AutoencoderKMNIST(nn.Module):
    def __init__(self):
        super(AutoencoderKMNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z = self.encoder(x)         
        x_reconst = self.decoder(z)
        return x_reconst, z         

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
modelo = AutoencoderKMNIST()
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

epochs = 20
print("\nIniciando entrenamiento del Ejercicio 3 (KMNIST | Latente = 16)...")

for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in trainloader:
        optimizer.zero_grad()
        
        reconstruccion, _ = modelo(images)
        loss = criterion(reconstruccion, images.view(-1, 28 * 28))
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Época {epoch+1}/{epochs} - Pérdida (BCE): {running_loss/len(trainloader):.4f}")

# ==========================================
# 4. VISUALIZACIÓN: Original, Reconstruida y LATENTE 4x4
# ==========================================
dataiter = iter(testloader)
images, labels = next(dataiter)

imagenes_unicas = []
clases_vistas = set()
for i in range(len(images)):
    lbl = labels[i].item()
    if lbl not in clases_vistas:
        clases_vistas.add(lbl)
        imagenes_unicas.append(images[i])
    if len(clases_vistas) == 10:
        break

imagenes_unicas = torch.stack(imagenes_unicas)

with torch.no_grad():
    reconstrucciones, latentes = modelo(imagenes_unicas)

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("Ejercicio 3: KMNIST y Visualización del Espacio Latente 4x4")

for i in range(10):
    # Original
    axes[0, i].imshow(imagenes_unicas[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Original")
    
    # Reconstruida
    img_recon = reconstrucciones[i].view(28, 28).numpy()
    axes[1, i].imshow(img_recon, cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Reconstruida")
    
    # Espacio Latente 4x4
    img_latente = latentes[i].view(4, 4).numpy()
    axes[2, i].imshow(img_latente, cmap='viridis')
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Latente 4x4")

plt.tight_layout()
plt.show()
