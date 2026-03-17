%% --- EJERCICIO 3: Canales RGB y Ruido de Imagen ---
% ==========================================
% 1. PARÁMETROS CONFIGURABLES
% ==========================================
% Intensidad del ruido Gaussiano (Varianza, 0 a 1)
VAR_GAUSSIAN = 0.05;
% Cantidad de ruido Sal y Pimienta (Densidad, 0 a 1)
DENSIDAD_SP = 0.05; 
% ==========================================
% 2. CARGA DE IMAGEN
% ==========================================
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg";
filename = "dog.jpg";
if ~exist(filename, 'file')
  websave(filename, url);
end
img = imread(filename);
% ==========================================
% 3. MANIPULACIÓN DE CANALES
% ==========================================
% A) Separar canales (Matlab los muestra en grises si son 2D)
canal_R = img(:, :, 1); % Capa Roja
canal_G = img(:, :, 2); % Capa Verde
canal_B = img(:, :, 3); % Capa Azul
% B) Reconstruir imagen SOLO con el canal ROJO
% Copiamos la imagen original para mantener tamaño y tipo de dato
img_solo_rojo = img;
img_solo_rojo(:, :, 2) = 0; % Poner canal Verde a cero
img_solo_rojo(:, :, 3) = 0; % Poner canal Azul a cero
% ==========================================
% 4. GENERACIÓN DE RUIDO
% ==========================================
% A) Ruido Gaussiano
% 'gaussian' añade ruido blanco. El segundo parámetro es la media (0)
% y el tercero es la varianza (VAR_GAUSSIAN).
img_gauss = imnoise(img, 'gaussian', 0, VAR_GAUSSIAN);
% B) Ruido Sal y Pimienta
% 'salt & pepper' añade píxeles blancos y negros aleatorios.
img_sp = imnoise(img, 'salt & pepper', DENSIDAD_SP);
% ==========================================
% 5. MOSTRAR RESULTADOS
% ==========================================
% --- FIGURA 1: Canales y Reconstrucción Roja ---
figure('Name', 'Ejercicio 3 - Parte 1: Canales');
subplot(2, 2, 1);
imshow(canal_R);
title('Canal R (Visualizado en Grises)');
subplot(2, 2, 2);
imshow(canal_G);
title('Canal G (Visualizado en Grises)');
subplot(2, 2, 3);
imshow(canal_B);
title('Canal B (Visualizado en Grises)');
subplot(2, 2, 4);
imshow(img_solo_rojo);
title('Reconstrucción: Solo Canal Rojo');
% --- FIGURA 2: Comparación de Ruidos ---
figure('Name', 'Ejercicio 3 - Parte 2: Ruido');
subplot(1, 3, 1);
imshow(img);
title('Imagen Original');
subplot(1, 3, 2);
imshow(img_gauss);
title(['Ruido Gaussiano (Var: ' num2str(VAR_GAUSSIAN) ')']);
subplot(1, 3, 3);
imshow(img_sp);
title(['Ruido Sal y Pimienta (Dens: ' num2str(DENSIDAD_SP) ')']);
