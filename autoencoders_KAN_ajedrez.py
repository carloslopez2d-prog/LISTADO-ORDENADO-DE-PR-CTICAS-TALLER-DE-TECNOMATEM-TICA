import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import imageio


# =====================================================================
# 1. GENERACIÓN DE DATOS (TABLERO DE AJEDREZ / PUNTOS RODEADOS)
# =====================================================================
print("Generando puntos altamente mezclados (patrón de tablero de ajedrez)...")
X_list = []
y_list = []


# Creamos una cuadrícula de 4x4 grupos de puntos
for i in range(4):
   for j in range(4):
       cx, cy = i * 2.0, j * 2.0
       # 20 puntos por cada grupo con un poco de ruido para que se toquen
       puntos = torch.randn(20, 2) * 0.4 + torch.tensor([cx, cy])
       X_list.append(puntos)
       # La clase alterna dependiendo de la fila y la columna (0, 1, 0, 1...)
       clase = (i + j) % 2
       y_list.append(torch.full((20,), clase, dtype=torch.long))


X = torch.cat(X_list)
y = torch.cat(y_list)


# Limites para dibujar el fondo
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
puntos_fondo = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)


# =====================================================================
# 2. ARQUITECTURA KAN AUMENTADA PARA ALTA COMPLEJIDAD
# =====================================================================
class CapaKAN(nn.Module):
   def __init__(self, in_dim, out_dim, puntos_red=5):
       super().__init__()
       # Ampliamos la red neuronal para que cubra todo el rango del tablero
       self.grid = nn.Parameter(torch.linspace(-3, 9, puntos_red), requires_grad=False)
       self.coeficientes = nn.Parameter(torch.randn(out_dim, in_dim, puntos_red) / puntos_red)


   def forward(self, x):
       distancias = x.unsqueeze(-1) - self.grid
       base_gaussiana = torch.exp(-distancias**2)
       salida = torch.einsum('big,oig->bo', base_gaussiana, self.coeficientes)
       return salida


class MiniKAN(nn.Module):
   def __init__(self):
       super().__init__()
       # Aumentamos a 16 neuronas para este problema tan difícil
       self.kan1 = CapaKAN(2, 16, puntos_red=10)
       self.kan2 = CapaKAN(16, 2, puntos_red=10)


   def forward(self, x):
       x = self.kan1(x)
       x = self.kan2(x)
       return x


# =====================================================================
# 3. ENTRENAMIENTO Y CAPTURA DE FOTOGRAMAS (CORREGIDO)
# =====================================================================
modelo = MiniKAN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.05)


epochs = 500
frames_gif = []


print("Entrenando la red KAN y grabando fotogramas...")
fig = plt.figure(figsize=(8, 6))


for epoch in range(epochs):
   optimizer.zero_grad()
   salida = modelo(X)
   loss = criterion(salida, y)
   loss.backward()
   optimizer.step()
  
   # Grabamos un fotograma cada 10 épocas
   if epoch % 10 == 0 or epoch == epochs - 1:
       plt.clf() # Limpiar lienzo
      
       with torch.no_grad():
           predicciones_fondo = modelo(puntos_fondo).argmax(dim=1).numpy()
           Z = predicciones_fondo.reshape(xx.shape)
      
       plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
       plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='bwr', edgecolors='k')
      
       plt.title(f"Separación Compleja KAN (Tablero de Ajedrez)\nÉpoca: {epoch} | Pérdida: {loss.item():.4f}")
       plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
      
       # EL ARREGLO DEL GIF: Añadir .copy() evita que se sobreescriba en la memoria RAM de tu Mac
       fig.canvas.draw()
       rgba = np.asarray(fig.canvas.buffer_rgba())
       imagen_frame = rgba[..., :3].copy()
       frames_gif.append(imagen_frame)
      
   if (epoch + 1) % 50 == 0:
       print(f"Época {epoch+1}/{epochs} completada... Pérdida: {loss.item():.4f}")


plt.close(fig)


# =====================================================================
# 4. GUARDAR EL GIF ANIMADO EN TU CARPETA
# =====================================================================
ruta_guardado = '/Users/molab/Documents/ttec'
os.makedirs(ruta_guardado, exist_ok=True)
nombre_gif = os.path.join(ruta_guardado, 'entrenamiento_ajedrez_kan.gif')


# Añadimos una pausa en el último fotograma
frames_gif = frames_gif + [frames_gif[-1]] * 15


# Guardamos el GIF en bucle
imageio.mimsave(nombre_gif, frames_gif, fps=10, loop=0)
print(f"\n¡Éxito absoluto! Tu GIF con todos los fotogramas guardados está en: {nombre_gif}")
