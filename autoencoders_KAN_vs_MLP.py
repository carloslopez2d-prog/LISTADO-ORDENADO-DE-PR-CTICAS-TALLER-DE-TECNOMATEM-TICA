import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import imageio


# =====================================================================
# 1. GENERACIÓN DE DATOS: TABLERO DE AJEDREZ (PUNTOS ÚNICOS)
# =====================================================================
print("Generando malla de 8x8 puntos únicos alternados (Alta Frecuencia)...")
X_list = []
y_list = []


# Creamos una cuadrícula de 8x8 (64 puntos en total)
# Centramos los datos restando 3.5 para que vayan aprox de -4 a 4
rango = 8
for i in range(rango):     # Coordenada Y (filas)
   for j in range(rango): # Coordenada X (columnas)
      
       # Coordenadas exactas (sin ruido aleatorio, es un punto fijo)
       cx = j - 3.5
       cy = i - 3.5
      
       X_list.append([cx, cy])
      
       # La clase alterna como en un tablero de ajedrez: par=0, impar=1
       clase = (i + j) % 2
       y_list.append(clase)


# Convertimos a tensores
X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.long)


# Límites para dibujar el fondo (un poco más amplio que los datos)
x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
puntos_fondo = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)


# =====================================================================
# 2. ARQUITECTURAS: KAN vs MLP
# =====================================================================
# --- RED KAN (Kolmogorov-Arnold Network) ---
class CapaKAN(nn.Module):
   def __init__(self, in_dim, out_dim, puntos_red=5):
       super().__init__()
       # Ajustamos la rejilla para cubrir el rango de nuestros datos (-4 a 4)
       self.grid = nn.Parameter(torch.linspace(-5, 5, puntos_red), requires_grad=False)
       self.coeficientes = nn.Parameter(torch.randn(out_dim, in_dim, puntos_red) / puntos_red)


   def forward(self, x):
       distancias = x.unsqueeze(-1) - self.grid
       base_gaussiana = torch.exp(-distancias**2)
       salida = torch.einsum('big,oig->bo', base_gaussiana, self.coeficientes)
       return salida


class MiniKAN(nn.Module):
   def __init__(self):
       super().__init__()
       # 16 neuronas ocultas y mayor resolución (puntos_red=15) para captar la alta frecuencia
       self.kan1 = CapaKAN(2, 16, puntos_red=15)
       self.kan2 = CapaKAN(16, 2, puntos_red=15)


   def forward(self, x):
       x = self.kan1(x)
       x = self.kan2(x)
       return x


# --- RED MLP CLÁSICA (Perceptrón Multicapa) ---
class MLP_Clasico(nn.Module):
   def __init__(self):
       super().__init__()
       # Necesita ser profunda y ancha para simular un tablero de ajedrez
       self.red = nn.Sequential(
           nn.Linear(2, 64),
           nn.ReLU(),
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Linear(64, 2)
       )


   def forward(self, x):
       return self.red(x)


# =====================================================================
# 3. ENTRENAMIENTO Y GENERACIÓN DEL GIF
# =====================================================================
modelo_kan = MiniKAN()
modelo_mlp = MLP_Clasico()


criterion = nn.CrossEntropyLoss()
# Learning rates ajustados para estabilidad
opt_kan = optim.Adam(modelo_kan.parameters(), lr=0.02)
opt_mlp = optim.Adam(modelo_mlp.parameters(), lr=0.02)


epochs = 600 # Necesitamos más épocas para converger en puntos individuales
frames_gif = []


print("Entrenando... KAN vs MLP en el problema de los Puntos Aislados")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.tight_layout(pad=4.0)


for epoch in range(epochs):
   # Entrenar KAN
   opt_kan.zero_grad()
   salida_kan = modelo_kan(X)
   loss_kan = criterion(salida_kan, y)
   loss_kan.backward()
   opt_kan.step()
  
   # Entrenar MLP
   opt_mlp.zero_grad()
   salida_mlp = modelo_mlp(X)
   loss_mlp = criterion(salida_mlp, y)
   loss_mlp.backward()
   opt_mlp.step()
  
   # Capturar fotograma cada 15 épocas
   if epoch % 15 == 0 or epoch == epochs - 1:
       ax1.clear()
       ax2.clear()
      
       with torch.no_grad():
           Z_kan = modelo_kan(puntos_fondo).argmax(dim=1).numpy().reshape(xx.shape)
           Z_mlp = modelo_mlp(puntos_fondo).argmax(dim=1).numpy().reshape(xx.shape)
      
       # Gráfico KAN
       ax1.contourf(xx, yy, Z_kan, alpha=0.3, cmap='bwr')
       # Pintamos los puntos un poco más grandes para verlos bien
       ax1.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='bwr', edgecolors='k', s=80)
       ax1.set_title(f"Red KAN (Ondular)\nPérdida: {loss_kan.item():.4f}")
       ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
      
       # Gráfico MLP
       ax2.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='bwr')
       ax2.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='bwr', edgecolors='k', s=80)
       ax2.set_title(f"Red MLP (Poligonal)\nPérdida: {loss_mlp.item():.4f}")
       ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
      
       # Extraer imagen
       fig.canvas.draw()
       rgba = np.asarray(fig.canvas.buffer_rgba())
       imagen_frame = rgba[..., :3].copy()
       frames_gif.append(imagen_frame)
      
   if (epoch + 1) % 50 == 0:
       print(f"Época {epoch+1}/{epochs} | Loss KAN: {loss_kan.item():.4f} | Loss MLP: {loss_mlp.item():.4f}")


plt.close(fig)


# =====================================================================
# 4. GUARDAR RESULTADO
# =====================================================================
ruta_guardado = '/Users/molab/Documents/ttec'
os.makedirs(ruta_guardado, exist_ok=True)
nombre_gif = os.path.join(ruta_guardado, 'batalla_puntos_ajedrez.gif')


frames_gif = frames_gif + [frames_gif[-1]] * 20


imageio.mimsave(nombre_gif, frames_gif, fps=10, loop=0)


print(f"\n¡GIF generado! Mira la diferencia de patrones en: {nombre_gif}")
