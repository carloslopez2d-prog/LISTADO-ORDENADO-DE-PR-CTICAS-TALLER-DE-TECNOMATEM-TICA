clc; clear; close all;
%% 1. Datos de entrenamiento
X = [0  2;
    3  3;
    3 -3;
    5  0];
% Etiquetas: azul = 0, verde = 1
y = [0; 0; 1; 1];
N = size(X,1);
%% 2. Inicialización de parámetros
rng(1);                 % Semilla para reproducibilidad
w = randn(2,1);         % Pesos
b = randn;              % Sesgo
apr = 0.1;              % Tasa de aprendizaje
epochs = 2000;          % Número de épocas
%% 3. Entrenamiento (descenso por gradiente)
for epoch = 1:epochs
   for i = 1:N
       x = X(i,:)';
       yi = y(i);
       % Salida del perceptrón
       z = w' * x + b;
       y_gorro = 1 / (1 + exp(-z));   % Sigmoide
       % Derivadas
       error = y_gorro - yi;
       d_sigma = y_gorro * (1 - y_gorro);
       % Gradientes
       grad_w = 2 * error * d_sigma * x;
       grad_b = 2 * error * d_sigma;
       % Actualización
       w = w - apr * grad_w;
       b = b - apr * grad_b;
   end
end
%% 4. Generación de la malla del plano
[x1, x2] = meshgrid(-20:0.5:20, -20:0.5:20);
Z = w(1)*x1 + w(2)*x2 + b;
Y_gorro = 1 ./ (1 + exp(-Z));
%% 5. Clasificación
class = Y_gorro >= 0.5;
%% 6. Visualización
figure;
hold on;
% Clasificación de la malla
scatter(x1(class==0), x2(class==0), 8, 'b', 'filled');
scatter(x1(class==1), x2(class==1), 8, 'g', 'filled');
% Frontera de decisión
contour(x1, x2, Y_gorro, [0.5 0.5], 'k', 'LineWidth', 3);
% Puntos de entrenamiento
scatter(X(y==0,1), X(y==0,2), 150, 'b', 'filled', ...
   'MarkerEdgeColor','k', 'LineWidth',2);
scatter(X(y==1,1), X(y==1,2), 150, 'g', 'filled', ...
   'MarkerEdgeColor','k', 'LineWidth',2);
xlabel('x_1');
ylabel('x_2');
title('Perceptrón: frontera de decisión y datos de entrenamiento');
axis equal;
grid on;
legend('Clase azul','Clase verde','Frontera decisión',...
      'Datos azul','Datos verde');
hold off;
