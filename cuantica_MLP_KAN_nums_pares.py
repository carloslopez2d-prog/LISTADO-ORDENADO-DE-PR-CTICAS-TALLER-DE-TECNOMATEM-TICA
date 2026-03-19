import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from kan import KAN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

# ==========================================
# 1. Datos: Alternando del 0 al 10 (Paridad)
# ==========================================
X_train = torch.arange(0, 11, dtype=torch.float32).unsqueeze(1)
# Si es impar es clase 1, si es par es clase 0
Y_train = (X_train % 2 == 1).float() 

# ==========================================
# 2. Las 3 Arquitecturas
# ==========================================
# A) MLP: CAPADO (Pocas neuronas para que fracase a propósito)
model_mlp = nn.Sequential(
    nn.Linear(1, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1), nn.Sigmoid()
)

# B) KAN: Kolmogorov-Arnold adaptativo
model_kan = KAN(width=[1, 5, 1], grid=10, k=3)

# C) Qubit: Simulador Cuántico Variacional
class QubitModelParity(nn.Module):
    def __init__(self):
        super().__init__()
        # Inicializamos cerca de pi/2 para ayudarle a pillar la frecuencia inicial
        self.w = nn.Parameter(torch.tensor([1.5])) 
        self.b = nn.Parameter(torch.tensor([0.0]))
    def forward(self, x):
        return torch.sin(self.w * x + self.b)**2

model_qubit = QubitModelParity()

opt_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.01)
opt_kan = torch.optim.Adam(model_kan.parameters(), lr=0.01)
opt_qubit = torch.optim.Adam(model_qubit.parameters(), lr=0.02)

# ==========================================
# 3. Entrenamiento y Generación del GIF
# ==========================================
epochs = 1500
frame_step = 30 # Guardamos 1 fotograma cada 30 épocas (50 frames en total)
filenames = []
os.makedirs("frames_paridad", exist_ok=True)

x_plot = torch.linspace(-1, 11, 200).unsqueeze(1)

print("Grabando la película de las 3 IA resolviendo la Paridad...")

for epoch in range(epochs + 1):
    # --- Entrenar MLP ---
    opt_mlp.zero_grad()
    loss_mlp = torch.mean((model_mlp(X_train) - Y_train)**2)
    loss_mlp.backward()
    opt_mlp.step()
    
    # --- Entrenar KAN ---
    opt_kan.zero_grad()
    loss_kan = torch.mean((model_kan(X_train) - Y_train)**2)
    loss_kan.backward()
    opt_kan.step()
    
    # --- Entrenar Qubit ---
    opt_qubit.zero_grad()
    loss_qubit = torch.mean((model_qubit(X_train) - Y_train)**2)
    loss_qubit.backward()
    opt_qubit.step()
    
    # --- Guardar Fotograma ---
    if epoch % frame_step == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # TITULO CORREGIDO: Le quitamos el y=1.05 y lo hacemos más grande
        fig.suptitle(f"Clasificación Alterna (0 a 10) | Época: {epoch}", fontsize=16)
        
        nombres = ["MLP (Capado - Fracasa)", "KAN (Se adapta por tramos)", "Qubit (Es su naturaleza)"]
        modelos = [model_mlp, model_kan, model_qubit]
        colores = ['blue', 'green', 'purple']
        
        for ax, nombre, mod, color in zip(axes, nombres, modelos, colores):
            with torch.no_grad():
                y_pred = mod(x_plot).numpy()
            
            ax.plot(x_plot.numpy(), y_pred, color=color, linewidth=2.5)
            
            # Puntos pares (Clase 0)
            ax.scatter(X_train.numpy()[::2], Y_train.numpy()[::2], c='red', s=80, zorder=5, label='Clase 0')
            # Puntos impares (Clase 1)
            ax.scatter(X_train.numpy()[1::2], Y_train.numpy()[1::2], c='blue', s=80, zorder=5, label='Clase 1')
            
            ax.set_title(nombre)
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(-1, 11)
            ax.grid(True, linestyle='--', alpha=0.6)
            if nombre == nombres[0]:
                ax.legend(loc='lower right')

        filename = f"frames_paridad/frame_{epoch}.png"
        
        # MARGENES CORREGIDOS: Dejamos espacio arriba para el título
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        plt.savefig(filename, dpi=80)
        filenames.append(filename)
        plt.close(fig)

# ==========================================
# 4. Ensamblar GIF
# ==========================================
print("Ensamblando el GIF animado...")
frames_img = [Image.open(f) for f in filenames]
ruta_gif = os.path.join(os.path.expanduser('~'), 'Downloads', 'evolucion_paridad_final.gif')

# Añadir pausa al final para poder ver el resultado
for _ in range(15):
    frames_img.append(frames_img[-1])

frames_img[0].save(ruta_gif, format='GIF', append_images=frames_img[1:], save_all=True, duration=80, loop=0)
print(f"[OK] Película guardada en: {ruta_gif}")
