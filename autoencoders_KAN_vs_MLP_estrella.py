import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import imageio


# =====================================================================
# 1. GENERACIÓN DE DATOS: "ESTRELLA DE MAR" (Complejidad Radial)
# =====================================================================
print("Generando patrón de Estrella/Ameba para forzar fronteras onduladas...")


# 1. Nube Interior (Roja / Clase 1)
n_inner = 200
# Aumentamos un poco la dispersión del centro para que se acerque a los bordes
X_inner = torch.randn(n_inner, 2) * 1.2
y_inner = torch.ones(n_inner, dtype=torch.long)


# 2. Nube Exterior Ondulada (Azul / Clase 0)
n_outer = 400
theta = torch.rand(n_outer) * 2 * np.pi


# --- LA MATEMÁTICA DE LA COMPLEJIDAD ---
# Definimos una forma de estrella usando una onda sinusoidal sobre el radio
frecuencia_onda = 7   # Número de "brazos" o puntas de la estrella
amplitud_onda = 1.4   # Cuánto entran y salen los brazos
radio_base = 4.5      # El tamaño medio del anillo


# Fórmula del radio variable: Base + Onda(ángulo) + Ruido
# Esto es lo que fuerza la frontera no circular
r_onda = radio_base + amplitud_onda * torch.sin(frecuencia_onda * theta)
r = r_onda + torch.randn(n_outer) * 0.3 # Ruido para darle grosor


# Convertimos a cartesianas
x_outer = r * torch.cos(theta)
y_outer = r * torch.sin(theta)
X_outer = torch.stack([x_outer, y_outer], dim=1)
y_outer = torch.zeros(n_outer, dtype=torch.long)


# Juntamos los datos
X = torch.cat([X_inner, X_outer])
y = torch.cat([y_inner, y_outer])


# Ampliamos los límites visuales porque la estrella es más grande
rango_visual = 7.0
x_min, x_max = -rango_visual, rango_visual
y_min, y_max = -rango_visual, rango_visual
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
puntos_fondo = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)


# =====================================================================
# 2. ARQUITECTURAS: KAN vs MLP
# =====================================================================
# --- RED KAN ---
class CapaKAN(nn.Module):
   def __init__(self, in_dim, out_dim, puntos_red=5):
       super().__init__()
       # Grid más amplio para cubrir la estrella
       self.grid = nn.Parameter(torch.linspace(-7, 7, puntos_red), requires_grad=False)
       self.coeficientes = nn.Parameter(torch.randn(out_dim, in_dim, puntos_red) / puntos_red)


   def forward(self, x):
       distancias = x.unsqueeze(-1) - self.grid
       base_gaussiana = torch.exp(-distancias**2)
       salida = torch.einsum('big,oig->bo', base_gaussiana, self.coeficientes)
       return salida


class MiniKAN(nn.Module):
   def __init__(self):
       super().__init__()
       # Aumentamos un poco la resolución de la rejilla (puntos_red=12) para las curvas
       self.kan1 = CapaKAN(2, 12, puntos_red=12)
       self.kan2 = CapaKAN(12, 2, puntos_red=12)


   def forward(self, x):
       x = self.kan1(x)
       x = self.kan2(x)
       return x


# --- RED MLP CLÁSICA ---
class MLP_Clasico(nn.Module):
   def __init__(self):
       super().__init__()
       # Le damos más neuronas para que intente hacer los picos de la estrella
       self.red = nn.Sequential(
           nn.Linear(2, 48),
           nn.ReLU(),
           nn.Linear(48, 48),
           nn.ReLU(),
           nn.Linear(48, 2)
       )


   def forward(self, x):
       return self.red(x)


# =====================================================================
# 3. ENTRENAMIENTO Y GENERACIÓN DEL GIF
# =====================================================================
modelo_kan = MiniKAN()
modelo_mlp = MLP_Clasico()


criterion = nn.CrossEntropyLoss()
opt_kan = optim.Adam(modelo_kan.parameters(), lr=0.025)
opt_mlp = optim.Adam(modelo_mlp.parameters(), lr=0.025)


epochs = 500
frames_gif = []


print("Entrenando... KAN (Curvas) vs MLP (Rectas) en la Estrella")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
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
  
   # Capturar fotograma cada 10 épocas
   if epoch % 10 == 0 or epoch == epochs - 1:
       ax1.clear()
       ax2.clear()
      
       with torch.no_grad():
           Z_kan = modelo_kan(puntos_fondo).argmax(dim=1).numpy().reshape(xx.shape)
           Z_mlp = modelo_mlp(puntos_fondo).argmax(dim=1).numpy().reshape(xx.shape)
      
       # Gráfico KAN
       ax1.contourf(xx, yy, Z_kan, alpha=0.3, cmap='bwr')
       ax1.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='bwr', edgecolors='k', s=50)
       ax1.set_title(f"Red KAN (Adaptación Curva)\nPérdida: {loss_kan.item():.4f}")
       ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
       ax1.set_aspect('equal', adjustable='box')
      
       # Gráfico MLP
       ax2.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='bwr')
       ax2.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='bwr', edgecolors='k', s=50)
       ax2.set_title(f"Red MLP (Aproximación Poligonal)\nPérdida: {loss_mlp.item():.4f}")
       ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
       ax2.set_aspect('equal', adjustable='box')
      
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
nombre_gif = os.path.join(ruta_guardado, 'batalla_estrella_compleja.gif')


frames_gif = frames_gif + [frames_gif[-1]] * 25


imageio.mimsave(nombre_gif, frames_gif, fps=12, loop=0)
print(f"\n¡GIF generado! Observa las diferencias en las puntas de la estrella: {nombre_gif}")
