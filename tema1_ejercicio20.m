clear; clc; close all;
%% 1. Configuración de Datos
% Datos del enunciado
X = [1, 2, 7, 9];       % Entradas
Y_target = [0, 0, 1, 1]; % Clases (0 o 1)
% Parámetros de entrenamiento
lr = 0.1;           % Tasa de aprendizaje (learning rate)
epochs = 200;       % Número de épocas
filename = 'clasificacion_sigmoide.gif';
% Inicialización (según enunciado)
w = 1;
b = 0;
% Preparar visualización
figure('Color', 'white', 'Position', [100 100 1000 500]);
loss_history = zeros(1, epochs);
% Rango para pintar la curva suavemente
xx = linspace(0, 10, 200);
%% 2. Bucle de Entrenamiento
for epoch = 1:epochs
  
   % --- FORWARD PASS ---
   % 1. Calcular Z lineal
   z = w * X + b;
  
   % 2. Aplicar función de activación Sigmoide
   % A es la probabilidad predicha (y gorro)
   A = 1 ./ (1 + exp(-z));
  
   % --- CÁLCULO DE PÉRDIDA (Entropía Cruzada) ---
   % L = -Sum( y*log(y_hat) + (1-y)*log(1-y_hat) )
   % Añadimos un pequeño eps para evitar log(0) si la predicción es perfecta
   epsilon = 1e-10;
   loss = -sum( Y_target .* log(A + epsilon) + (1 - Y_target) .* log(1 - A + epsilon) );
   loss_history(epoch) = loss;
  
   % --- BACKPROPAGATION ---
   % La derivada de la Entropía Cruzada con Sigmoide se simplifica a: (A - Y)
   error = A - Y_target;
  
   % Gradientes (Acumulados para todos los datos)
   grad_w = sum(error .* X);
   grad_b = sum(error);
  
   % --- ACTUALIZACIÓN (Descenso del Gradiente) ---
   w = w - lr * grad_w;
   b = b - lr * grad_b;
  
   % --- VISUALIZACIÓN Y GIF ---
   if mod(epoch, 5) == 0 || epoch == 1
       clf;
      
       % -- Subplot 1: Clasificación Sigmoide --
       subplot(1, 2, 1); hold on;
      
       % Dibujar la curva sigmoide actual
       z_plot = w * xx + b;
       sig_plot = 1 ./ (1 + exp(-z_plot));
       plot(xx, sig_plot, 'b-', 'LineWidth', 2);
      
       % Dibujar los puntos de datos
       % Pintamos los de Clase 0 en rojo y Clase 1 en verde para distinguir
       scatter(X(Y_target==0), Y_target(Y_target==0), 100, 'r', 'filled', 'DisplayName', 'Clase 0');
       scatter(X(Y_target==1), Y_target(Y_target==1), 100, 'g', 'filled', 'DisplayName', 'Clase 1');
      
       % Línea de decisión (0.5)
       yline(0.5, 'k--', 'Umbral 0.5');
      
       title(sprintf('Época %d\nw=%.2f, b=%.2f', epoch, w, b));
       ylabel('Probabilidad P(y=1|x)'); xlabel('Entrada x');
       axis([0 10 -0.1 1.1]); grid on; legend('Modelo', 'Clase 0', 'Clase 1', 'Location', 'southeast');
      
       % -- Subplot 2: Función de Pérdida --
       subplot(1, 2, 2);
       plot(1:epoch, loss_history(1:epoch), 'k-', 'LineWidth', 1.5);
       hold on;
       plot(epoch, loss, 'ro', 'MarkerFaceColor', 'r');
      
       title(sprintf('Entropía Cruzada (Loss)\nValor actual: %.4f', loss));
       xlabel('Época'); ylabel('Loss');
       xlim([0 epochs]); ylim([0 max(loss_history(1))*1.1]);
       grid on;
      
       drawnow;
      
       % Guardar frame para GIF
       frame = getframe(gcf);
       im = frame2im(frame);
       [imind, cm] = rgb2ind(im, 256);
       if epoch == 1
           imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
       else
           imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
       end
   end
end
fprintf('Entrenamiento terminado.\n');
fprintf('Predicción final para x=[1, 2, 7, 9]:\n');
disp(A);





