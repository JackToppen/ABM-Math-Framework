function [pres, den] = updateFields( yl, rd, bl, y_max )
%UPDATEFIELDS 

pres = zeros(1, length(yl)+3);

% Initialize pressure
pres(3:(length(pres)-1)) = (1/y_max)*(yl + rd + bl);

% Diffuse pressure
for loop=1:10
    pres_next = zeros(1,length(pres));

    % Pressure diffusion for most of the environment
    for i=2:(length(pres) - 1)
        pres_next(i-1) = pres_next(i-1) + 0.5*(((y_max - 2)*(3/8) + 2*(2/8))/y_max)*pres(i);
        
        pres_next(i) = pres_next(i) + 0.5*(1 + (((y_max - 2)*(2/8) + 2*(4/8))/y_max))*pres(i);
        
        pres_next(i+1) = pres_next(i+1) + 0.5*(((y_max - 2)*(3/8) + 2*(2/8))/y_max)*pres(i);
            
    end

    % Pressure diffusion for the leftmost / minimum column
    pres_next(1) = pres_next(1) + 0.5*(1 + (((y_max - 2)*(5/8) + 2*(6/8))/y_max))*pres(1);
    
    pres_next(2) = pres_next(2) + 0.5*(((y_max - 2)*(3/8) + 2*(2/8))/y_max)*pres(1);
    
    % Pressure diffusion for the rightmost / maximum column
    pres_next(length(pres)-1) = pres_next(length(pres)-1) + 0.5*(((y_max - 2)*(3/8) + 2*(2/8))/y_max)*pres(length(pres));
    
    pres_next(length(pres)) = pres_next(length(pres)) + 0.5*(1 + (((y_max - 2)*(5/8) + 2*(6/8))/y_max))*pres(length(pres));
    
    % Loop update
    pres = pres_next;

end

% Set density gradient
den = zeros(1,length(pres));

for i=3:(length(den) - 1)
    % We follow the description in the supplementary material
    % Note that we add the biasing 
    den(i) = (pres(i-1) - pres(i+1))/2 + 0.1;
end

end

