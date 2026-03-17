%tema 1. ejercicio 19. apartado c
clear all; clc; close all;

% DATOS DEL EJERCICIO
A_exp = [0.20, 0.44, 0.79, 1.23, 1.77];
T_exp = [4.05, 5.49, 6.68, 8.34, 9.21];
T_teorico = [3.94, 5.43, 7.06, 8.70, 10.36];

% PARÁMETROS SUGERIDOS
tasa_aprendizaje = 10^-5;
epocas = 60;
C1 = 1/3;
C2 = 2;
filename = 'entrenamiento_perceptron.gif';

T_func = @(A, C1, C2) (C1 ./ sqrt(A)) .* log(exp(C2 .* A) + sqrt(exp(2 .* C2 .* A) - 1));

% 2. CONFIGURACIÓN INICIAL DE LA FIGURA
fig = figure('Color', 'w', 'Position', [100, 100, 800, 600]);
hold on;
grid on;
xlabel('Área del paracaídas (A)');
ylabel('Tiempo de caída (T(A))');
axis([0 2 0 12]);

plot(A_exp, T_teorico, 'b-o', 'LineWidth', 1, 'DisplayName', 'Valores Teóricos');
plot(A_exp, T_exp, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Valores Experimentales');
legend('Location', 'northwest');

% 3. BUCLE DE ENTRENAMIENTO Y GENERACIÓN DE GIF
for epoca = 1:epocas
    
    T_pred = T_func(A_exp, C1, C2);
    error = T_pred - T_exp;

    % Descenso del gradiente
    grad_C1 = mean(error);
    grad_C2 = mean(error .* A_exp) * 0.5;
    
    C1 = C1 - tasa_aprendizaje * grad_C1;
    C2 = C2 - tasa_aprendizaje * grad_C2;

    % --- Actualización del gráfico ---
    if exist('h_perc', 'var'), delete(h_perc); end

    % Dibujamos la nueva curva
    A_suave = linspace(0.15, 1.9, 100);
    T_suave = T_func(A_suave, C1, C2);
    h_perc = plot(A_suave, T_suave, 'k--', 'LineWidth', 2, ...
        'DisplayName', 'Ajuste Perceptrón');

    title(['Época: ', num2str(epoca), ...
           ' | C1: ', num2str(C1,4), ...
           ' | C2: ', num2str(C2,4)]);
    drawnow;

    % --- Captura y guardado en GIF ---
    frame = getframe(fig);
    im = frame2im(frame);
    [A_gif, map] = rgb2ind(im, 256);

    if epoca == 1
        imwrite(A_gif, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A_gif, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end

hold off;
fprintf('GIF guardado correctamente como: %s\n', filename);
