# APARTADO (A)

import os
# --- FIX 1: Evita el conflicto de librerías que tira el núcleo ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --- 1. PREPARAR DATOS ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- FIX 2: num_workers=0 para evitar problemas de hilos en Windows/Jupyter ---
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, 
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, 
                                         shuffle=True, num_workers=0)

# --- 2. DEFINIR ARQUITECTURA ---
class RedCNN(nn.Module):
    def __init__(self):
        super(RedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciar
net = RedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# --- 3. ENTRENAMIENTO ---
print("Entrenando... (Espera un momento)")
j = 5 #numero de épocas
for epoch in range(j):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Época {epoch+1} completada. Loss: {running_loss/len(trainloader):.3f}')

print("¡Entrenamiento OK!")

# --- 4. MOSTRAR RESULTADO (CON CUIDADO) ---
try:
    # Obtener un lote
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Predecir
    with torch.no_grad(): # FIX 3: Desactiva gradientes para inferencia
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

    # Mostrar por consola primero (por si falla la gráfica)
    print(f"La red dice que es un: {predicted.item()}")
    print(f"En realidad es un: {labels.item()}")

    # Graficar
    img = images[0].numpy().squeeze()
    plt.figure() # Crear nueva figura explícitamente
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicción: {predicted.item()} (Real: {labels.item()})")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Error al mostrar la imagen: {e}")
