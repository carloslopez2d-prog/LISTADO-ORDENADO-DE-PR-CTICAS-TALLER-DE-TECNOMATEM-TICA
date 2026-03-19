import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ==========================================
# 1. DATOS DE ENTRENAMIENTO
# ==========================================
# Puntos 1 y 2 -> Clase 0
# Puntos 3 y 4 -> Clase 1
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)


# Puntos para dibujar la curva (alta resolución)
x_test = torch.linspace(0, 5, 100).unsqueeze(1)


# ==========================================
# 2. DEFINICIÓN DE LOS MODELOS
# ==========================================


# A) Perceptrón Multicapa Clásico (MLP)
mlp = nn.Sequential(
   nn.Linear(1, 16),
   nn.ReLU(),
   nn.Linear(16, 1),
   nn.Sigmoid()
)


# B) KAN Simplificado (1D RBF-KAN)
# En lugar de pesos fijos y activaciones en nodos, aprende una combinación lineal de funciones (Gaussianas)
class SimpleKAN1D(nn.Module):
   def __init__(self, num_bases=8):
       super().__init__()
       # Centros de las gaussianas repartidos en el dominio
       self.mu = nn.Parameter(torch.linspace(0, 5, num_bases))
       self.w = nn.Parameter(torch.randn(num_bases))
       self.sigma = 1.0
       self.bias = nn.Parameter(torch.zeros(1))
      
   def forward(self, x):
       diff = x - self.mu  # Distancia a los centros
       rbf = torch.exp(-diff**2 / self.sigma**2) # Activación RBF
       out = rbf @ self.w + self.bias
       return torch.sigmoid(out).unsqueeze(1)


kan = SimpleKAN1D()


# C) Perceptrón Cuántico (1 Cúbit)
class QuantumPerceptron(nn.Module):
   def __init__(self):
       super().__init__()
       # Inicializamos w y b cerca de la solución teórica para ayudar a que no caiga en mínimos locales bruscos
       self.w = nn.Parameter(torch.tensor([1.0]))
       self.b = nn.Parameter(torch.tensor([-2.5]))
      
   def forward(self, x):
       z = (self.w * x + self.b) / 2.0
       # Evitamos ceros y unos absolutos para la inestabilidad del logaritmo (Cross Entropy)
       prob = torch.sin(z)**2
       return torch.clamp(prob, 1e-6, 1.0 - 1e-6)


quantum = QuantumPerceptron()


# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
epochs = 300
criterion = nn.BCELoss() # Entropía Cruzada Binaria


# Optimizadores separados
opt_mlp = torch.optim.Adam(mlp.parameters(), lr=0.05)
opt_kan = torch.optim.Adam(kan.parameters(), lr=0.05)
opt_q = torch.optim.Adam(quantum.parameters(), lr=0.05)


# Listas para guardar los frames del GIF
history_mlp = []
history_kan = []
history_q = []


print("Entrenando los 3 modelos simultáneamente...")


for ep in range(epochs):
   # --- Forward y Loss ---
   pred_mlp = mlp(x_train)
   loss_mlp = criterion(pred_mlp, y_train)
  
   pred_kan = kan(x_train)
   loss_kan = criterion(pred_kan, y_train)
  
   pred_q = quantum(x_train)
   loss_q = criterion(pred_q, y_train)
  
   # --- Backward y Optimización ---
   opt_mlp.zero_grad()
   loss_mlp.backward()
   opt_mlp.step()
  
   opt_kan.zero_grad()
   loss_kan.backward()
   opt_kan.step()
  
   opt_q.zero_grad()
   loss_q.backward()
   opt_q.step()
  
   # --- Guardar Frame para el GIF ---
   with torch.no_grad():
       history_mlp.append(mlp(x_test).numpy())
       history_kan.append(kan(x_test).numpy())
       history_q.append(quantum(x_test).numpy())
      
   if ep % 50 == 0 or ep == epochs - 1:
       print(f"Época {ep:3d} | L_MLP: {loss_mlp:.4f} | L_KAN: {loss_kan:.4f} | L_Q: {loss_q:.4f}")


# ==========================================
# 4. GENERACIÓN DEL GIF ANIMADO
# ==========================================
print("\nGenerando la animación... (esto puede tardar unos segundos)")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(top=0.85)


titles = ["Perceptrón Multicapa (MLP)", "KAN (RBF 1D)", "Perceptrón Cuántico (1 Cúbit)"]
lines = []


x_np = x_test.numpy()


# Configurar cada subplot
for i, ax in enumerate(axes):
   # Puntos de entrenamiento
   ax.scatter([1, 2], [0, 0], color='red', label='Clase 0', s=100, zorder=5, edgecolors='black')
   ax.scatter([3, 4], [1, 1], color='blue', label='Clase 1', s=100, zorder=5, edgecolors='black')
  
   ax.set_xlim(0, 5)
   ax.set_ylim(-0.1, 1.1)
   ax.set_title(titles[i], fontsize=12, fontweight='bold')
   ax.set_xlabel("x")
   ax.grid(True, linestyle='--', alpha=0.6)
   if i == 0:
       ax.set_ylabel("Probabilidad P(y=1)")
       ax.legend(loc='lower right')
      
   # Inicializar las curvas (vacías al principio)
   line, = ax.plot(x_np, np.zeros_like(x_np), color='orange', lw=3)
   lines.append(line)


# Función que actualiza cada frame
def update(frame):
   lines[0].set_ydata(history_mlp[frame])
   lines[1].set_ydata(history_kan[frame])
   lines[2].set_ydata(history_q[frame])
   fig.suptitle(f"Evolución de Fronteras de Decisión - Época: {frame:03d} / {epochs}", fontsize=16)
   return lines


ani = animation.FuncAnimation(fig, update, frames=epochs, interval=30, blit=False)


# Detectar el escritorio automáticamente
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(desktop_path):

   desktop_path = os.path.join(os.path.expanduser("~"), "Escritorio")


gif_filepath = os.path.join(desktop_path, "comparativa_clasificacion_modelos.gif")
# Guardar el GIF
ani.save(gif_filepath, writer='pillow', fps=30)
print(f"¡Animación completada y guardada en: {gif_filepath}!")








  
