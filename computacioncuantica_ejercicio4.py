# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:30:25 2026

@author: Propietario
"""

import math
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. HIPERPARÁMETROS ---
epochs = 1000        # Aumentamos un poco las épocas para la función identidad
lr = 0.01            # Tasa de aprendizaje
N = 100              # Número de puntos
C = 2.0              # Parámetro de amplitud C
D = -2.0             # Parámetro de amplitud D (C y D opuestos para maximizar pendiente)

# --- 2. GENERACIÓN DE DATOS ---
# Intervalo [-2, 2] para la función identidad f(x) = x
X = [-2 + 4 * i / (N - 1) for i in range(N)]
Y = [x for x in X]   # f(x) = x

# --- 3. INICIALIZACIÓN DE PARÁMETROS ---
w = 0.5              # Inicialización pequeña para la zona lineal del seno/coseno
b = 0.0

history = []

print("Entrenando Perceptrón Cuántico (Función Identidad con Activación Combinada)...")

# --- 4. BUCLE DE ENTRENAMIENTO (MSE) ---
for epoch in range(epochs):
    grad_w = 0.0
    grad_b = 0.0
    total_loss = 0.0
    current_preds = []
  
    for x, y in zip(X, Y):
        # Forward pass
        z = (w * x + b) / 2.0
        prob1 = math.sin(z)**2  # P|1>
        prob0 = math.cos(z)**2  # P|0>
        
        # Nueva función de activación: Ax = C*P|1> + D*P|0>
        y_hat = C * prob1 + D * prob0
        current_preds.append(y_hat)
      
        # Pérdida MSE
        loss = (y_hat - y)**2
        total_loss += loss
      
        # Backward pass (Derivada analítica)
        # La derivada de Ax respecto a z es: (C-D) * sin(2z) = (C-D) * sin(wx + b)
        dL_dyhat = 2 * (y_hat - y)
        dyhat_dz = (C - D) * math.sin(w * x + b)
      
        grad_w += dL_dyhat * dyhat_dz * (x / 2.0)
        grad_b += dL_dyhat * dyhat_dz * (1.0 / 2.0)
      
    history.append((current_preds, epoch, total_loss / N, w, b))
  
    w -= lr * (grad_w / N)
    b -= lr * (grad_b / N)
  
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Época {epoch:3d} | Pérdida: {total_loss/N:.5f} | w: {w:.5f} | b: {b:.5f}")

# --- 5. GENERACIÓN DEL GIF ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(X, Y, label="$f(x) = x$ (Identidad)", color="C0", linewidth=2.5)
line, = ax.plot(X, history[0][0], label="Activación $C \cdot P|1> + D \cdot P|0>$", color="C1", linewidth=2.5)

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper left")
ax.grid(True, linestyle='--', alpha=0.6)

def update(frame_idx):
    preds, epoch, loss, current_w, current_b = history[frame_idx]
    line.set_ydata(preds)
    ax.set_title(f"Ejercicio 4 | Época: {epoch:03d}\nIdentidad con Activación Combinada")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(history), blit=True)

# Ruta de guardado (Escritorio/Descargas)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(desktop_path):
    desktop_path = os.path.join(os.path.expanduser("~"), "Escritorio")
if not os.path.exists(desktop_path):
    desktop_path = os.getcwd()

gif_filepath = os.path.join(desktop_path, "ejercicio4_identidad_cuantica.gif")

print(f"\nGuardando GIF en: {gif_filepath}...")
ani.save(gif_filepath, writer='pillow', fps=30)
print("¡Listo!")
