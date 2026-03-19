import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ssl


# Parche de seguridad para Mac
ssl._create_default_https_context = ssl._create_unverified_context


# ==========================================
# 1. CARGA DE DATOS (MNIST)
# ==========================================
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# ==========================================
# 2. ARQUITECTURA DEL VAE
# ==========================================
class VAE(nn.Module):
   def __init__(self):
       super(VAE, self).__init__()
       # ENCODER: Reduce la imagen a una capa oculta de 400
       self.fc1 = nn.Linear(784, 400)
      
       # EL TRUCO DEL VAE: En lugar de un solo vector latente, saca DOS vectores de 20 dimensiones:
       self.fc21 = nn.Linear(400, 20) # Para la Media (mu)
       self.fc22 = nn.Linear(400, 20) # Para el Logaritmo de la Varianza (logvar)
      
       # DECODER: Reconstruye desde el espacio latente de 20 dimensiones
       self.fc3 = nn.Linear(20, 400)
       self.fc4 = nn.Linear(400, 784)


   def encode(self, x):
       h1 = F.relu(self.fc1(x))
       return self.fc21(h1), self.fc22(h1) # Devuelve media y varianza


   def reparameterize(self, mu, logvar):
       # Aquí es donde "tira los dados" añadiendo ruido estadístico
       std = torch.exp(0.5 * logvar) # Desviación típica
       eps = torch.randn_like(std)   # Ruido aleatorio (epsilon)
       return mu + eps * std         # Ecuación fundamental del VAE: z = mu + epsilon * sigma


   def decode(self, z):
       h3 = F.relu(self.fc3(z))
       return torch.sigmoid(self.fc4(h3)) # Sigmoide para que los píxeles estén entre 0 y 1


   def forward(self, x):
       mu, logvar = self.encode(x.view(-1, 784))
       z = self.reparameterize(mu, logvar) # Muestreo latente
       return self.decode(z), mu, logvar


# ==========================================
# 3. FUNCIÓN DE PÉRDIDA ESPECÍFICA DEL VAE
# ==========================================
def loss_function(recon_x, x, mu, logvar):
   # 1. Pérdida de Reconstrucción (BCE): ¿Se parece el número generado al original?
   BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
  
   # 2. Divergencia Kullback-Leibler (KLD): Obliga a la distribución a parecerse a una Campana de Gauss normal
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  
   return BCE + KLD


# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
modelo_vae = VAE()
optimizer = optim.Adam(modelo_vae.parameters(), lr=1e-3)


epochs = 15
print("Entrenando el VAE (aprox. 1-2 minutos)...")
modelo_vae.train()


for epoch in range(epochs):
   train_loss = 0
   for data, _ in trainloader:
       optimizer.zero_grad()
       recon_batch, mu, logvar = modelo_vae(data)
       loss = loss_function(recon_batch, data, mu, logvar)
       loss.backward()
       train_loss += loss.item()
       optimizer.step()
      
   print(f'Época {epoch+1}/{epochs} | Pérdida VAE: {train_loss / len(trainloader.dataset):.4f}')


# ==========================================
# 5. GENERAR NUEVAS IMÁGENES A PARTIR DE UN NÚMERO EXISTENTE
# ==========================================
print("\nBuscando un '3' para generar nuevas versiones de él...")
modelo_vae.eval()


# Buscar un número '3' en el set de prueba
img_original = None
for img, label in testset:
   if label == 3:
       img_original = img
       break


with torch.no_grad():
   # 1. Pasamos el 3 original por el Encoder para sacar su Media y Varianza
   mu, logvar = modelo_vae.encode(img_original.view(-1, 784))
  
   # 2. Vamos a "tirar los dados" 5 veces distintas usando esa misma Media y Varianza
   # Esto generará 5 números "3" que son parecidos pero no idénticos (diferentes grosores, inclinaciones...)
   imagenes_generadas = []
   for _ in range(5):
       z_muestreado = modelo_vae.reparameterize(mu, logvar)
       img_gen = modelo_vae.decode(z_muestreado).view(28, 28).numpy()
       imagenes_generadas.append(img_gen)


# ==========================================
# 6. VISUALIZACIÓN
# ==========================================
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
fig.suptitle("VAE Generativo: Original vs 5 Variaciones Inventadas", fontsize=16)


# Dibujar el original
axes[0].imshow(img_original.squeeze().numpy(), cmap='gray')
axes[0].set_title("3 Original", fontweight='bold')
axes[0].axis('off')


# Dibujar las 5 variaciones generadas
for i in range(5):
   axes[i+1].imshow(imagenes_generadas[i], cmap='gray')
   axes[i+1].set_title(f"3 Generado v{i+1}")
   axes[i+1].axis('off')


plt.tight_layout()
plt.show()
