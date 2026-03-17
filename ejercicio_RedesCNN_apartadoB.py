# APARTADO (B)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 11:23:54 2026

@author: Propietario
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1️⃣ Transformaciones (Igual que antes)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2️⃣ CAMBIO: Cargar dataset FashionMNIST en lugar de MNIST
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data_fashion', train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data_fashion', train=False, download=True, transform=transform
)

# Etiquetas para Fashion-MNIST (para que el gráfico sea legible)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# 3️⃣ Definir la CNN (La arquitectura sirve igual porque las imágenes miden lo mismo)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
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

# 4️⃣ Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5️⃣ Entrenamiento
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Entrenamiento finalizado")

# 6️⃣ Probar con una imagen de ropa
model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    image, label = next(dataiter)

    image_dev = image.to(device)
    output = model(image_dev)
    _, predicted = torch.max(output, 1)

    img_show = image.squeeze() * 0.5 + 0.5 
    
    plt.figure(figsize=(5,5))
    plt.imshow(img_show.numpy(), cmap='gray')
    
    # Usamos la lista 'classes' para mostrar el nombre en lugar del número
    plt.title(f"Real: {classes[label.item()]} \nPredicción: {classes[predicted.item()]}")
    plt.axis("off")
    plt.show()
