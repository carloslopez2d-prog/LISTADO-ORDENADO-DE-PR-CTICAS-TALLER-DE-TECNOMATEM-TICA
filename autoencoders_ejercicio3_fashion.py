import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""
# ==========================================
# 1. CARGA DE DATOS: ALTERNATIVA FASHION-MNIST
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])

# Cambiamos KMNIST por FashionMNIST
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
"""
# ==========================================
# 1. CARGA DE DATOS: KMNIST (Caracteres Japoneses)
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])

# Fíjate que ahora usamos torchvision.datasets.KMNIST
trainset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# ==========================================
# 2. DEFINICIÓN DE LA ARQUITECTURA (AE Latente = 16)
# ==========================================
class AutoencoderKMNIST(nn.Module):
    def __init__(self):
        super(AutoencoderKMNIST, self).__init__()
        # Codificador: 784 -> 128 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        # Decodificador: 16 -> 128 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z = self.encoder(x)         # Guardamos el vector latente (z) para dibujarlo luego
        x_reconst = self.decoder(z)
        return x_reconst, z         # Ahora devolvemos también 'z'

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
modelo = AutoencoderKMNIST()
criterion = nn.BCELoss()

# Usamos Adam en vez de SGD para que aprenda mejor los trazos complejos
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

epochs = 20
print("Iniciando entrenamiento del Ejercicio 3 (KMNIST | Latente = 16)...")

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

# Cogemos 10 ejemplos únicos
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
    # Obtenemos las imágenes y los vectores latentes de 16 valores
    reconstrucciones, latentes = modelo(imagenes_unicas)

# Preparamos la figura con 3 filas
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("Ejercicio 3: KMNIST y Visualización del Espacio Latente 4x4")

for i in range(10):
    # Fila 1: Original
    axes[0, i].imshow(imagenes_unicas[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Original")
    
    # Fila 2: Reconstruida
    img_recon = reconstrucciones[i].view(28, 28).numpy()
    axes[1, i].imshow(img_recon, cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Reconstruida")
    
    # Fila 3: Espacio Latente (Convertimos el vector de 16 en una matriz de 4x4)
    # Le ponemos color (viridis) para que se vean bien las activaciones
    img_latente = latentes[i].view(4, 4).numpy()
    axes[2, i].imshow(img_latente, cmap='viridis')
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Latente 4x4")

plt.tight_layout()
plt.show()
