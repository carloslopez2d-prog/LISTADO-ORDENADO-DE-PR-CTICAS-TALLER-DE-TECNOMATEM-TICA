%% --- EJERCICIO 2 
% 1. PARÁMETROS DE AJUSTE
UMBRAL = 115;
% Tamaño del filtro de mediana 
K_SIZE = [13 13];
% ==========================================
% 2. Carga y Procesamiento
% ==========================================
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg";
filename = "dog.jpg";
if ~exist(filename, 'file')
  websave(filename, url);
end
img = imread(filename);     
img_gray = rgb2gray(img);   
% Aplicamos Filtro de Mediana para limpiar el ruido del césped
img_procesada = medfilt2(img_gray, K_SIZE);
% ==========================================
% 3. Lógica de Bordes (Sobel)
% ==========================================
img_binaria = zeros(size(img_procesada));
img_binaria(img_procesada > UMBRAL) = 255;
img_binaria(img_procesada <= UMBRAL) = 0;
% Filtros Sobel
Kx = [-1 0 1; -2 0 2; -1 0 1];
Ky = [-1 -2 -1; 0 0 0; 1 2 1];
img_bin_double = double(img_binaria);
Gx = imfilter(img_bin_double, Kx);
Gy = imfilter(img_bin_double, Ky);
Bordes = sqrt(Gx.^2 + Gy.^2);
figure('Name', 'Ejercicio 2: Resultados Finales', 'NumberTitle', 'off');
subplot(1,3,1);
imshow(img);
title('Original');
% 2. Escala de Grises 
subplot(1,3,2);
imshow(img_procesada);
title(['Grises (Filtro Mediana ' num2str(K_SIZE(1)) ')']);
subplot(1,3,3);
imshow(uint8(Bordes));
title(['Bordes (Umbral: ' num2str(UMBRAL) ')']);
