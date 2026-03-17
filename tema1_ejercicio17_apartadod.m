eta = 0.1;
epoca = 10;
w = 0;   %peso inicial
for k = 0:epoca
   %calculamos pérdida
   L = (w^3 - w - 1)^2;
  
   %mostramos resultados
   fprintf('Época %2d : w = % .6f  L = %.6f\n', k, w, L);
  
   %derivada de L respecto w
   grad = 2*(w^3 - w - 1)*(3*w^2 - 1);
  
   %actualización (evitamos actualizar tras la última impresión)
   w = w - eta*grad;
end

