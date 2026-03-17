clear; clc; close all;
%% 1. Configuración
x_dom = linspace(0, 3, 100);   
x_bc = [0, 3];                 
y_bc = [2, 5];                 
y_exacta = x_dom + 2; % Solución exacta y = x + 2
%% 2. Arquitectura (4 Neuronas + Sesgo c)
N = 4;
a = randn(1,N);      
w = randn(1,N);      
b = randn(1,N);      
c_glob = randn(1,1); 
lr = 0.0025;          
epochs = 2000;       
filename = 'PINN_Colores_Final.gif';
loss_history = zeros(1,epochs);
figure('Color','white','Position',[50 50 1200 500]); % Ventana ancha
%% 3. Bucle de Entrenamiento
for epoch = 1:epochs
  
   % --- Forward Pass y Loss ---
   L = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
   loss_history(epoch) = L;
   % --- Gradientes (Diferencias Finitas) ---
   eps = 1e-5;
  
   % Actualización a
   for i = 1:N
       a_old = a(i);
       a(i) = a_old + eps; Lp = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       a(i) = a_old - eps; Lm = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       grad = (Lp - Lm) / (2*eps);
       a(i) = a_old - lr * grad;
   end
   % Actualización w
   for i = 1:N
       w_old = w(i);
       w(i) = w_old + eps; Lp = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       w(i) = w_old - eps; Lm = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       grad = (Lp - Lm) / (2*eps);
       w(i) = w_old - lr * grad;
   end
   % Actualización b
   for i = 1:N
       b_old = b(i);
       b(i) = b_old + eps; Lp = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       b(i) = b_old - eps; Lm = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
       grad = (Lp - Lm) / (2*eps);
       b(i) = b_old - lr * grad;
   end
   % Actualización c_glob
   c_old = c_glob;
   c_glob = c_old + eps; Lp = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
   c_glob = c_old - eps; Lm = custom_loss(a, w, b, c_glob, x_dom, x_bc, y_bc);
   grad = (Lp - Lm) / (2*eps);
   c_glob = c_old - lr * grad;
   %% 4. Visualización
   if mod(epoch, 40) == 0 || epoch == 1
       clf;
      
       % === SUBPLOT 1: GEOMETRÍA Y COLORES ===
       subplot(1,2,1); hold on;
      
       % Calcular predicción
       y_pred = zeros(size(x_dom));
       for i=1:N
           y_pred = y_pred + a(i)*tanh(w(i)*x_dom + b(i));
       end
       y_pred = y_pred + c_glob;
      
       % 1. Colorear Zonas (Verde Arriba / Roja Abajo)
       % Creamos polígonos para 'fill'. El límite superior es 10, inferior -5
       x_poly = [x_dom, fliplr(x_dom)];
       y_upper = [10*ones(size(x_dom)), fliplr(y_pred)]; % Zona Verde
       y_lower = [-5*ones(size(x_dom)), fliplr(y_pred)]; % Zona Roja
      
       fill(x_poly, y_upper, [0.7 1 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5); % Verde
       fill(x_poly, y_lower, [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5); % Rojo
      
       % 2. Recta exacta (Discontinua)
       plot(x_dom, y_exacta, 'b--', 'LineWidth', 2, 'DisplayName', 'Solución Exacta');
      
       % 3. Curva de la Red (Negra Continua)
       plot(x_dom, y_pred, 'k-', 'LineWidth', 3, 'DisplayName', 'PINN (Red)');
      
       % 4. Puntos de Condiciones Iniciales
       scatter(x_bc, y_bc, 120, 'k', 'filled', 'MarkerEdgeColor','w', 'LineWidth', 1.5, 'DisplayName', 'Condiciones BC');
      
       % Estética
       xlabel('x'); ylabel('y');
       title(['Época: ', num2str(epoch)]);
       legend('Zona superiror', 'Zona inferior', 'Exacta', 'Red Neuronal', 'CC', 'Location', 'northwest');
       ylim([0 7]); xlim([0 3]);
       grid on; box on;
      
       % === SUBPLOT 2: PÉRDIDA ===
       subplot(1,2,2);
       plot(1:epoch, loss_history(1:epoch), 'r-', 'LineWidth', 1.5);
       set(gca, 'YScale', 'log');
       xlim([0 epochs]); % Eje fijo
      
       title(['Pérdida L = ', num2str(L, '%.4e')]);
       xlabel('Épocas'); ylabel('L (Escala Log)'); grid on;
      
       % GIF
       drawnow;
       frame = getframe(gcf);
       im = frame2im(frame);
       [imind, map] = rgb2ind(im, 256);
       if epoch == 1
           imwrite(imind, map, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
       else
           imwrite(imind, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
       end
   end
end
%% Funciones Auxiliares
function L = custom_loss(a, w, b, c, x, x_bc, y_bc)
   y_xx = zeros(size(x));
   for i = 1:length(a)
       z = w(i)*x + b(i);
       t = tanh(z);
       y_xx = y_xx + a(i) * (w(i)^2) * (-2 * t .* (1 - t.^2));
   end
  
   L_ode = trapz(x, abs(y_xx).^2);
  
   y_pred_bc = zeros(size(x_bc));
   for k = 1:length(x_bc)
       val = 0;
       for i = 1:length(a)
           val = val + a(i) * tanh(w(i)*x_bc(k) + b(i));
       end
       y_pred_bc(k) = val + c;
   end
  
   L_bc = (y_pred_bc(1) - y_bc(1))^2 + (y_pred_bc(2) - y_bc(2))^2;
   L = L_ode + L_bc;end
