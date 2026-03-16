% Ejercicio 1
%URL de la imagen
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg";
% Nombre local temporal
filename = "dog.jpg";
% Descargar la imagen desde Internet
websave(filename, url);
% Leer la imagen
img = imread(filename);
% Mostrar la imagen original
figure;
imshow(img);
title("Imagen Original");
% Convertir a escala de grises
img_gray = rgb2gray(img);
% Mostrar la imagen en escala de grises
figure;	
imshow(img_gray);
title("Imagen en Escala de Grises");
% Invertir la escala de grises
img_gray_inv = imcomplement(img_gray);
% Mostrar la imagen con la escala de grises invertida
figure;
imshow(img_gray_inv);
title("Escala de Grises Invertida");
