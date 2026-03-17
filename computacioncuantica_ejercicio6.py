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

%2ª PARTE
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ==========================================
# 1. DATOS DE ENTRENAMIENTO ACTUALIZADOS
# ==========================================
# Clase 0: x = 1, 2, 5, 6
# Clase 1: x = 3, 4
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)


# Puntos de test ampliados para dibujar la curva hasta x=7
x_test = torch.linspace(0, 7, 150).unsqueeze(1)


# ==========================================
# 2. DEFINICIÓN DE LOS MODELOS
# ==========================================


# A) Perceptrón Multicapa Clásico (MLP)
# Ahora necesita formar un "bache" (0 -> 1 -> 0)
mlp = nn.Sequential(
   nn.Linear(1, 16),
   nn.ReLU(),
   nn.Linear(16, 1),
   nn.Sigmoid()
)


# B) KAN Simplificado (1D RBF-KAN)
class SimpleKAN1D(nn.Module):
   def __init__(self, num_bases=10):  # Aumentamos un poco las bases para cubrir hasta x=7
       super().__init__()
       self.mu = nn.Parameter(torch.linspace(0, 7, num_bases))
       self.w = nn.Parameter(torch.randn(num_bases))
       self.sigma = 1.0
       self.bias = nn.Parameter(torch.zeros(1))
      
   def forward(self, x):
       diff = x - self.mu
       rbf = torch.exp(-diff**2 / self.sigma**2)
       out = rbf @ self.w + self.bias
       return torch.sigmoid(out).unsqueeze(1)


kan = SimpleKAN1D()


# C) Perceptrón Cuántico (1 Cúbit)
class QuantumPerceptron(nn.Module):
   def __init__(self):
       super().__init__()
       # Inicializamos w y b para ayudar al descenso de gradiente a encontrar
       # la frecuencia de onda correcta más rápido y evitar mínimos locales.
       self.w = nn.Parameter(torch.tensor([1.5]))
       self.b = nn.Parameter(torch.tensor([-2.5]))
      
   def forward(self, x):
       z = (self.w * x + self.b) / 2.0
       prob = torch.sin(z)**2
       return torch.clamp(prob, 1e-6, 1.0 - 1e-6)


quantum = QuantumPerceptron()


# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
epochs = 400  # Aumentamos a 400 para asegurar que el MLP converja la "montaña"
criterion = nn.BCELoss()


opt_mlp = torch.optim.Adam(mlp.parameters(), lr=0.05)
opt_kan = torch.optim.Adam(kan.parameters(), lr=0.05)
opt_q = torch.optim.Adam(quantum.parameters(), lr=0.05)


history_mlp = []
history_kan = []
history_q = []


print("Entrenando los 3 modelos simultáneamente en el problema 0-1-0...")


for ep in range(epochs):
   # Forward y Loss
   pred_mlp = mlp(x_train)
   loss_mlp = criterion(pred_mlp, y_train)
  
   pred_kan = kan(x_train)
   loss_kan = criterion(pred_kan, y_train)
  
   pred_q = quantum(x_train)
   loss_q = criterion(pred_q, y_train)
  
   # Backward
   opt_mlp.zero_grad()
   loss_mlp.backward()
   opt_mlp.step()
  
   opt_kan.zero_grad()
   loss_kan.backward()
   opt_kan.step()
  
   opt_q.zero_grad()
   loss_q.backward()
   opt_q.step()
  
   # Guardar frame
   with torch.no_grad():
       history_mlp.append(mlp(x_test).numpy())
       history_kan.append(kan(x_test).numpy())
       history_q.append(quantum(x_test).numpy())
      
   if ep % 50 == 0 or ep == epochs - 1:
       print(f"Época {ep:3d} | L_MLP: {loss_mlp:.4f} | L_KAN: {loss_kan:.4f} | L_Q: {loss_q:.4f}")


# ==========================================
# 4. GENERACIÓN DEL GIF ANIMADO
# ==========================================
print("\nGenerando la animación... (esto tomará unos segundos)")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(top=0.85)


titles = ["Perceptrón Multicapa (MLP)", "KAN (RBF 1D)", "Perceptrón Cuántico (1 Cúbit)"]
lines = []
x_np = x_test.numpy()


for i, ax in enumerate(axes):
   # Pintamos los nuevos puntos
   ax.scatter([1, 2, 5, 6], [0, 0, 0, 0], color='red', label='Clase 0', s=100, zorder=5, edgecolors='black')
   ax.scatter([3, 4], [1, 1], color='blue', label='Clase 1', s=100, zorder=5, edgecolors='black')
  
   ax.set_xlim(0, 7)
   ax.set_ylim(-0.1, 1.1)
   ax.set_title(titles[i], fontsize=12, fontweight='bold')
   ax.set_xlabel("x")
   ax.grid(True, linestyle='--', alpha=0.6)
   if i == 0:
       ax.set_ylabel("Probabilidad P(y=1)")
       ax.legend(loc='upper right')
      
   line, = ax.plot(x_np, np.zeros_like(x_np), color='orange', lw=3)
   lines.append(line)


def update(frame):
   lines[0].set_ydata(history_mlp[frame])
   lines[1].set_ydata(history_kan[frame])
   lines[2].set_ydata(history_q[frame])
   fig.suptitle(f"Adaptación a patrón 0-1-0 | Época: {frame:03d} / {epochs}", fontsize=16)
   return lines


ani = animation.FuncAnimation(fig, update, frames=epochs, interval=30, blit=False)


# Detectar el escritorio
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(desktop_path):
   desktop_path = os.path.join(os.path.expanduser("~"), "Escritorio")


gif_filepath = os.path.join(desktop_path, "comparativa_modelos_0_1_0.gif")


# Guardar GIF
ani.save(gif_filepath, writer='pillow', fps=30)
print(f"¡Animación completada y guardada en: {gif_filepath}!")

%3ª PARTE
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





  
