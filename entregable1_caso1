clear; clc; close all;
X = [2, 3, 2, 4;   %Coordenada x1
    4, 6, 0, -5]; %Coordenada x2
Y_target = [1, 1, 0, 0];
% Parámetros
lr = 0.2;
epochs = 200;
filename = 'entrenamiento_perceptron_sigmoide.gif';
% Inicialización
w = [0.1, -0.1];
b = 0;
% Preparación de la malla para el coloreado
[gridX1, gridX2] = meshgrid(linspace(1, 5, 100), linspace(-6, 8, 100));
grid_pts = [gridX1(:)'; gridX2(:)'];
figure('Color', 'white', 'Position', [100 100 1100 500]);
loss_history = zeros(1, epochs);
%% 2. Bucle de Entrenamiento
for epoch = 1:epochs
   z = w * X + b;
   A = 1 ./ (1 + exp(-z)); %Sigmoide
    % Función pérdida(Entropía Cruzada)
   loss = -sum(Y_target .* log(A + 1e-10) + (1 - Y_target) .* log(1 - A + 1e-10));
   loss_history(epoch) = loss;
   error = A - Y_target;
   grad_w = error * X';
   grad_b = sum(error);
    % Actualización
   w = w - lr * grad_w;
   b = b - lr * grad_b;
    % Visualización
   if mod(epoch, 5) == 0 || epoch == 1
       subplot(1, 2, 1); hold off;
       % --- COLOREADO DEL PLANO ---
       z_grid = w * grid_pts + b;
       A_grid = reshape(1 ./ (1 + exp(-z_grid)), size(gridX1));
       imagesc([1 5], [-6 8], A_grid);
       set(gca, 'YDir', 'normal');
       hold on;
       % Rojo (Clase 0) a Verde (Clase 1)
       colormap(subplot(1,2,1), [linspace(1,0.5,32)', linspace(0.5,1,32)', linspace(0.5,0.5,32)']);
       alpha(0.5);
       % Dibujar la frontera de decisión
       x_plot = linspace(1, 5, 10);
       y_plot = (-w(1) * x_plot - b) / w(2);
       plot(x_plot, y_plot, 'k-', 'LineWidth', 2, 'DisplayName', 'Frontera');
       scatter(X(1, Y_target==1), X(2, Y_target==1), 100, 'g', 'filled', 'MarkerEdgeColor', 'k');
       scatter(X(1, Y_target==0), X(2, Y_target==0), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
    
       title(sprintf('Época %d: Activación Sigmoide (Coloreado)', epoch));
       xlabel('x1'); ylabel('x2'); axis([1 5 -6 8]); grid on;
       % Subplot
       subplot(1, 2, 2);
       plot(1:epoch, loss_history(1:epoch), 'k-', 'LineWidth', 1.5);
       title('Error de Entropía Cruzada');
       xlabel('Época'); ylabel('Loss'); grid on;
       drawnow;
       % --- GENERACIÓN DEL GIF ---
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
