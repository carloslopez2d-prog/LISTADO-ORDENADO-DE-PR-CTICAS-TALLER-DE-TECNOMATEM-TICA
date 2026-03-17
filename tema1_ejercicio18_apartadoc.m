clear; clc; close all;
% Datos
X = [0, 1, 3, 4];
Y = [0, 2, 2, 5];
learning_rate = 0.01; % Tasa de aprendizaje
epochs = 30;         % Número de iteraciones
filename = 'evolucion_perceptron.gif';
% Inicialización de pesos (w) y sesgo (b)
w = 0;
b = 0;
% Historial de pérdida para graficar
loss_history = zeros(1, epochs);
% figura
fig = figure('Position', [100, 100, 1000, 500], 'Color', 'w');
x_line = linspace(-1, 5, 100); % Dominio para dibujar la recta suavemente
%% 2. Bucle de Entrenamiento
for epoch = 1:epochs
  
   % Predicción del modelo: y_hat = w*x + b
   Y_hat = w * X + b;
  
   % Cálculo del error
   error = Y_hat - Y;
  
   % Función de Pérdida
   loss = 0.5 * sum(error.^2);
   loss_history(epoch) = loss;
  
   % Cálculo de gradiente
   % dL/dw = sum(error * x)
   grad_w = sum(error .* X);
  
   % dL/db = sum(error)
   grad_b = sum(error);
  
   % Actualización de parámetros
   w = w - learning_rate * grad_w;
   b = b - learning_rate * grad_b;
  
   % VISUALIZACIÓN
   % Solo actualizamos el GIF cada ciertas épocas para que no sea muy lento,
   % o todas si son pocas épocas.
  
   clf(fig); % Limpiar figura
  
   % -- SUBPLOT 1: Datos y Recta de Regresión --
   subplot(1, 2, 1);
   scatter(X, Y, 100, 'b', 'filled'); % Puntos originales
   hold on;
  
   % Calcular la recta actual para graficar
   y_line_plot = w * x_line + b;
   plot(x_line, y_line_plot, 'r-', 'LineWidth', 2);
  
   title(sprintf('Regresión Lineal (Época %d)\n y = %.2fx + %.2f', epoch, w, b));
   xlabel('X'); ylabel('Y');
   axis([-1 5 -1 6]); % Ejes fijos para ver el movimiento
   grid on;
   legend('Datos', 'Perceptrón', 'Location', 'northwest');
  
   % -- SUBPLOT 2: Evolución de la Pérdida --
   subplot(1, 2, 2);
   plot(1:epoch, loss_history(1:epoch), 'k-', 'LineWidth', 1.5);
   hold on;
   plot(epoch, loss, 'ro', 'MarkerFaceColor', 'r'); % Punto actual
  
   title(sprintf('Función de Pérdida L\nLoss actual: %.4f', loss));
   xlabel('Época'); ylabel('Pérdida (SSE)');
   xlim([0 epochs]);
   ylim([0 max(loss_history(1))*1.1]); % Ajustar Y al error inicial
   grid on;
  
   drawnow;
  
   % --- GENERACIÓN DEL GIF ---
   frame = getframe(fig);
   im = frame2im(frame);
   [imind, cm] = rgb2ind(im, 256);
  
   if epoch == 1
       imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
   else
       imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
   end
end
fprintf('Entrenamiento finalizado.\nPesos finales: w = %.4f, b = %.4f\n', w, b);







