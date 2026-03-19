import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import imageio
import ssl


# Parche de seguridad para Mac
ssl._create_default_https_context = ssl._create_unverified_context


# ==========================================
# 1. CARGAR DATOS
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# ==========================================
# 2. ARQUITECTURA DEL AUTOENCODER
# ==========================================
class AutoencoderGenerativo(nn.Module):
   def __init__(self):
       super(AutoencoderGenerativo, self).__init__()
       # Encoder: 784 -> 256 -> 32 (Espacio Latente)
       self.encoder = nn.Sequential(
           nn.Linear(28 * 28, 256),
           nn.ReLU(),
           nn.Linear(256, 32)
       )
       # Decoder: 32 -> 256 -> 784
       self.decoder = nn.Sequential(
           nn.Linear(32, 256),
           nn.ReLU(),
           nn.Linear(256, 28 * 28),
           nn.Sigmoid()
       )


   def encode(self, x):
       return self.encoder(x.view(-1, 28 * 28))


   def decode(self, z):
       return self.decoder(z).view(-1, 28, 28)


   def forward(self, x):
       z = self.encode(x)
       return self.decode(z)


# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
modelo = AutoencoderGenerativo()
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)


epochs = 15 # Con 15 épocas ya entiende bastante bien las formas
print("Entrenando la IA (esto tardará un par de minutos)...")


for epoch in range(epochs):
   modelo.train()
   running_loss = 0.0
   for images, _ in trainloader:
       optimizer.zero_grad()
       out = modelo(images)
       loss = criterion(out.view(-1, 28 * 28), images.view(-1, 28 * 28))
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
   print(f"Época {epoch+1}/{epochs} completada - Pérdida: {running_loss/len(trainloader):.4f}")


# ==========================================
# 4. MAGIA GENERATIVA: INTERPOLACIÓN Y GIF
# ==========================================
print("\nEntrenamiento terminado. Buscando un '2' y un '7' para mezclarlos...")
modelo.eval()


# Buscar un número 2 y un número 7 reales en el dataset
img_A, img_B = None, None
for img, label in testset:
   if label == 2 and img_A is None:
       img_A = img
   if label == 7 and img_B is None:
       img_B = img
   if img_A is not None and img_B is not None:
       break


with torch.no_grad():
   # Pasamos ambas imágenes al Espacio Latente (vectores de 32 dimensiones)
   z_A = modelo.encode(img_A)
   z_B = modelo.encode(img_B)


   frames_gif = []
   pasos = 60 # Número de fotogramas del GIF de ida
  
   print("Generando fotogramas de transición...")
   alphas = np.linspace(0, 1, pasos)
  
   for alpha in alphas:
       # FÓRMULA DE INTERPOLACIÓN LINEAL
       z_intermedio = (1 - alpha) * z_A + alpha * z_B
      
       # Le pedimos al Decoder que dibuje este vector inventado
       img_generada = modelo.decode(z_intermedio).squeeze().numpy()
      
       # Convertimos la imagen a formato de píxeles (0-255) para el GIF
       img_generada_8bit = (img_generada * 255).astype(np.uint8)
       frames_gif.append(img_generada_8bit)


# Para que el GIF se detenga un momento al principio y al final
frames_gif = [frames_gif[0]] * 15 + frames_gif + [frames_gif[-1]] * 15


# Añadimos los mismos fotogramas marcha atrás para hacer efecto "boomerang"
frames_gif = frames_gif + frames_gif[::-1]


# ==========================================
# 5. GUARDAR EL GIF EN LA RUTA ESPECÍFICA
# ==========================================
ruta_guardado = '/Users/molab/Documents/ttec'
# Nos aseguramos de que la carpeta exista por si acaso
os.makedirs(ruta_guardado, exist_ok=True)


nombre_gif = os.path.join(ruta_guardado, 'transformacion_2_a_7.gif')


# Guardar el GIF (loop=0 hace que se repita infinitamente)
imageio.mimsave(nombre_gif, frames_gif, fps=15, loop=0)
print(f"\n¡Éxito absoluto! Tu GIF en bucle te está esperando en: {nombre_gif}")
