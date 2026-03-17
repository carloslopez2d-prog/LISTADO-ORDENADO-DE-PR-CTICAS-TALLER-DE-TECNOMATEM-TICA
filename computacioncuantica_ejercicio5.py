# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:45:29 2026

@author: Propietario
"""

# -*- coding: utf-8 -*-
import math
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. HIPERPARÁMETROS ---
epochs = 2000        # Aumentamos un poco las épocas para un ajuste perfecto
lr = 0.05            # Tasa de aprendizaje ajustada
N = 100              # Puntos
C = 1.5              # Amplitud para cubrir el rango
D = -1.5             # D opuesto a C

# --- 2. GENERACIÓN DE DATOS ---
X = [-1 + 2 * i / (N - 1) for i in range(N)]
Y = [x * (1 - x**2) for x in X]  # Función objetivo: f(x) = x(1 - x^2)

# --- 3. INICIALIZACIÓN DE PARÁMETROS (¡EL TRUCO ESTÁ AQUÍ!) ---
random.seed(42)      # Para asegurar que salga bien a la primera
w = random.uniform(-0.5, 0.5)  # Inicialización aleatoria pequeña
b = random.uniform(-0.5, 0.5)

history = []

print("Entrenando Perceptrón Cuántico para f(x) = x(1 - x^2)...")

# --- 4. BUCLE DE ENTRENAMIENTO (MSE) ---
for epoch in range(epochs):
    grad_w = 0.0
    grad_b = 0.0
    total_loss = 0.0
    current_preds = []
    
    for x, y in zip(X, Y):
        # Forward pass
        z = (w * x + b) / 2.0
        prob1 = math.sin(z)**2  # Probabilidad estado |1>
        prob0 = math.cos(z)**2  # Probabilidad estado |0>
        
        # Activación: combinación lineal
        y_hat = C * prob1 + D * prob0
        current_preds.append(y_hat)
        
        # Pérdida MSE
        loss = (y_hat - y)**2
        total_loss += loss
        
        # Backward pass
        dL_dyhat = 2 * (y_hat - y)
        dyhat_dz = (C - D) * math.sin(w * x + b)
        
        grad_w += dL_dyhat * dyhat_dz * (x / 2.0)
        grad_b += dL_dyhat * dyhat_dz * (1.0 / 2.0)
        
    history.append((current_preds, epoch, total_loss / N, w, b))
    
    # Actualización de pesos
    w -= lr * (grad_w / N)
    b -= lr * (grad_b / N)
    
    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"Época {epoch:4d} | Pérdida: {total_loss/N:.5f} | w: {w:.5f} | b: {b:.5f}")

# --- 5. GENERACIÓN DEL GIF ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X, Y, label="$f(x) = x(1 - x^2)$", color="C0", linewidth=2.5)
line, = ax.plot(X, history[0][0], label="Aproximación Cuántica", color="C1", linewidth=2.5)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper right")
ax.grid(True, linestyle='--', alpha=0.6)

def update(frame_idx):
    preds, epoch, loss, current_w, current_b = history[frame_idx]
    line.set_ydata(preds)
    ax.set_title(f"Aproximación Cuántica | Época: {epoch:04d}\n$f(x) = x(1 - x^2)$")
    return line,

# Saltamos frames para que el GIF no pese demasiado y sea rápido
frames_a_mostrar = history[::10]
ani = animation.FuncAnimation(fig, lambda i: update(i*10), frames=len(frames_a_mostrar), interval=30, blit=True)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(desktop_path):
    desktop_path = os.path.join(os.path.expanduser("~"), "Escritorio")
if not os.path.exists(desktop_path):
    desktop_path = os.getcwd()

gif_filepath = os.path.join(desktop_path, "aproximacion_funcion_compleja.gif")

print(f"\nGuardando GIF en: {gif_filepath}...")
ani.save(gif_filepath, writer='pillow', fps=30)
print("¡Listo!")
