import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Evita fallos en el núcleo al graficar

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. CARGA Y PREPROCESADO DE DATOS (MNIST)
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# ==========================================
# 2. DEFINICIÓN DE LA ARQUITECTURA (Latente = 16)
# ==========================================
class AutoencoderEj1(nn.Module):
    def __init__(self):
        super(AutoencoderEj1, self).__init__()
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
        z = self.encoder(x)         # Guardamos los 16 valores comprimidos
        x_reconst = self.decoder(z) 
        return x_reconst, z         # Devolvemos la imagen y el vector latente

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
modelo = AutoencoderEj1()
criterion = nn.BCELoss() 
optimizer = optim.SGD(modelo.parameters(), lr=0.5, momentum=0.9)

epochs = 20
print("Iniciando entrenamiento del Ejercicio 1 (Dimensión latente = 16)...")

for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in trainloader:
        optimizer.zero_grad()
        
        reconstruccion, _ = modelo(images) # Extraemos la reconstrucción
        
        loss = criterion(reconstruccion, images.view(-1, 28 * 28))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Época {epoch+1}/{epochs} completada - Pérdida (BCE): {running_loss/len(trainloader):.4f}")

# ==========================================
# 4. VISUALIZACIÓN: ORIGINAL, RECONSTRUIDA Y LATENTE 4x4
# ==========================================
dataiter = iter(testloader)
images, labels = next(dataiter)

# Buscar un ejemplo de cada dígito (0 al 9)
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

# Preparamos la figura con 3 FILAS ahora
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("Ejercicio 1: MNIST | Original, Reconstruida y Espacio Latente (16 neuronas)")

for i in range(10):
    # Fila 1: Original
    axes[0, i].imshow(imagenes_unicas[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Original")
    
    # Fila 2: Reconstruida
    img_recon = reconstrucciones[i].view(28, 28).numpy()
    axes[1, i].imshow(img_recon, cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("AE (16)")
    
    # Fila 3: Espacio Latente (Convertimos los 16 valores en una matriz 4x4)
    # Usamos el mapa de color 'viridis' para que destaquen las neuronas más activas
    img_latente = latentes[i].view(4, 4).numpy()
    axes[2, i].imshow(img_latente, cmap='viridis')
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title("Latente 4x4")

plt.tight_layout()
plt.show()
# 5. VISUALIZACIÓN DE RESULTADOS Y ARQUITECTURA
model.eval()

# --- NUEVA SECCIÓN: DIBUJO DE LA ARQUITECTURA ---
def dibujar_arquitectura():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Definimos las capas y sus tamaños para el dibujo
    capas = [784, 128, 16, 128, 784]
    nombres = ['X', 'f_enc', 'z', 'f_dec', 'X_hat']
    x_pos = [1, 3, 5, 7, 9]
    
    for i, (tam, nombre) in enumerate(zip(capas, nombres)):
        # Dibujamos rectángulos o líneas que representen la densidad
        # El alto del bloque es proporcional (logarítmicamente) al tamaño de la capa
        height = torch.log(torch.tensor(tam)).item() * 0.5
        color = 'gold' if i < 2 else ('lightblue' if i > 2 else 'lightgrey')
        
        # Dibujar el bloque de neuronas
        rect = plt.Rectangle((x_pos[i]-0.4, -height/2), 0.8, height, 
                             color=color, alpha=0.3, ec='black')
        ax.add_patch(rect)
        
        # Etiquetas de tamaño y nombre
        plt.text(x_pos[i], height/2 + 0.2, f"Dim: {tam}", ha='center')
        plt.text(x_pos[i], -height/2 - 0.5, nombre, ha='center', fontweight='bold', fontsize=12)

    # Dibujar líneas de conexión (esquema de cuello de botella)
    for i in range(len(x_pos)-1):
        plt.plot([x_pos[i]+0.4, x_pos[i+1]-0.4], [0, 0], 'k--', alpha=0.2)

    plt.title("Esquema de la Arquitectura Autoencoder", fontsize=15)
    plt.xlim(0, 10)
    plt.ylim(-5, 5)
    plt.axis('off')
    plt.show()

# Ejecutar dibujos
dibujar_arquitectura()

with torch.no_grad():
    images, _ = next(iter(test_loader))
    images = images.to(device)
    reconstructed = model(images).cpu().numpy()
    images = images.cpu().numpy()

    plt.figure(figsize=(15, 4))
    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Originales")
        ax = plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Reconstrucciones")
    plt.show()
