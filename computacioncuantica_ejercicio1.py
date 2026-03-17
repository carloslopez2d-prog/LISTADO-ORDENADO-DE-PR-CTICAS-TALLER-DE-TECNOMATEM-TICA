
import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. HIPERPARÁMETROS ---
epochs = 800         # Número de épocas 
lr = 0.01            # Tasa de aprendizaje recomendada 
N = 100              # 100 puntos equiespaciados 

# --- 2. GENERACIÓN DE DATOS ---
# Intervalo [-1, 1] 
X = [-1 + 2 * i / (N - 1) for i in range(N)]
Y = [x**2 for x in X]  # Función objetivo f(x) = x^2 

# --- 3. INICIALIZACIÓN DE PARÁMETROS ---
w = 2.0 
b = 0.0

# Lista para guardar la "historia" y hacer el GIF luego
history = []

print("Entrenando el Perceptrón Cuántico (MSE)...")

# --- 4. BUCLE DE ENTRENAMIENTO "A MANO" ---
for epoch in range(epochs):
    grad_w = 0.0
    grad_b = 0.0
    total_loss = 0.0
    
    # Guardamos las predicciones de esta época para el frame del GIF
    current_preds = []
  
    for x, y in zip(X, Y):
        # Forward pass (Circuito cuántico Ry) [cite: 45, 51]
        z = (w * x + b) / 2.0
        y_hat = math.sin(z)**2  # Función de activación natural 
        current_preds.append(y_hat)
      
        # Función de pérdida: Error Cuadrático Medio (MSE) 
        loss = (y_hat - y)**2
        total_loss += loss
      
        # Backward pass (Derivadas analíticas exactas para MSE)
        # dL/dyhat = 2 * (yhat - y)
        # dyhat/dz = 2 * sin(z) * cos(z) = sin(2*z) = sin(wx + b)
        dL_dyhat = 2 * (y_hat - y)
        dyhat_dz = math.sin(w * x + b)
      
        grad_w += dL_dyhat * dyhat_dz * (x / 2.0)
        grad_b += dL_dyhat * dyhat_dz * (1.0 / 2.0)
      
    # Guardamos los datos de la época actual (Frame completo)
    history.append((current_preds, epoch, total_loss / N, w, b))
  
    # Actualización de pesos [cite: 9]
    w -= lr * (grad_w / N)
    b -= lr * (grad_b / N)
  
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Época {epoch:3d} | Pérdida (MSE): {total_loss/N:.5f} | w: {w:.5f} | b: {b:.5f}")

print("\nEntrenamiento finalizado. Generando el GIF...")

# --- 5. GENERACIÓN DEL GIF ANIMADO ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X, Y, label="$f(x) = x^2$ (Objetivo)", color="C0", linewidth=2.5)
line, = ax.plot(X, history[0][0], label="Perceptrón Cuántico", color="C1", linewidth=2.5)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="lower center")
ax.grid(True, linestyle='--', alpha=0.6)

def update(frame_idx):
    preds, epoch, loss, current_w, current_b = history[frame_idx]
    line.set_ydata(preds)
    ax.set_title(f"Aproximación MSE | Época: {epoch:03d} / {epochs}\nLoss: {loss:.4f} | w: {current_w:.4f} | b: {current_b:.4f}")
    return line,

# Creamos la animación con todos los frames
ani = animation.FuncAnimation(fig, update, frames=len(history), blit=True)

# Buscar la ruta del escritorio
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(desktop_path):
    desktop_path = os.path.join(os.path.expanduser("~"), "Escritorio")
# Si no encuentra escritorio, intenta Descargas
if not os.path.exists(desktop_path):
    desktop_path = os.path.join(os.path.expanduser("~"), "Downloads")

gif_filepath = os.path.join(desktop_path, "ejercicio1_mse_cuantico.gif")

print(f"Guardando GIF en: {gif_filepath} ...")
ani.save(gif_filepath, writer='pillow', fps=30)
print("¡GIF guardado con éxito!")
