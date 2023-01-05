function [cellType] = moveCells( cellType_prev, pres, den )
% Function written by Daniel A. Cruz
% Used in GRR approximation for rib ABM from https://doi.org/10.7554/eLife.29144
% See associated manuscript for more details: https://arxiv.org/abs/2211.00630
%MOVECELLS 

cellType_temp=zeros(1,length(cellType_prev));
cellType = cellType_temp;

% Move cells
for i=1:length(cellType)
    if pres(i+2) <= 4
        cellType_temp(i) = cellType_temp(i) + cellType_prev(i);
    else 

        % Move right
        if den(i+2) > 0
            d = ceil(den(i+2));

            if (floor(d/2) + i) >= length(cellType)
                cellType_temp(length(cellType)) = cellType_temp(length(cellType)) + cellType_prev(i);

            elseif (d + i) > length(cellType)
                part = cellType_prev(i)/(length(cellType) - floor(d/2) - i + 1);
                cellType_temp((i + floor(d/2)):length(cellType)) = cellType_temp((i + floor(d/2)):length(cellType)) + part;
            
            else
                part = cellType_prev(i)/(d - floor(d/2) + 1);
                cellType_temp((i + floor(d/2)):(i + d)) = cellType_temp((i + floor(d/2)):(i + d)) + part;

            end
        
        % Move left
        elseif den(i+2) < 0
            d = floor(den(i+2));

            if (ceil(d/2) + i) <= 1
                cellType_temp(1) = cellType_temp(1) + cellType_prev(i);

            elseif (d + i) < 1
                part = cellType_prev(i)/(i + ceil(d/2));
                cellType_temp(1:(i + ceil(d/2))) = cellType_temp(1:(i + ceil(d/2))) + part;
            
            else
                part = cellType_prev(i)/(ceil(d/2) - d + 1);
                cellType_temp((i + d):(i + ceil(d/2))) = cellType_temp((i + d):(i + ceil(d/2))) + part;

            end
        end
    end
end

% Jiggle cells

% Random noise adjustment for most of the environment
for i=2:(length(cellType)-1)
    cellType(i - 1) = cellType(i-1) + 0.16 * cellType_temp(i);
    cellType(i) = cellType(i) + 0.68 * cellType_temp(i);
    cellType(i + 1) = cellType(i+1) + 0.16 * cellType_temp(i);
end

% Random noise adjustment for the leftmost / minimum column
cellType(1) = cellType(1) + 0.84 * cellType_temp(1);
cellType(2) = cellType(2) + 0.16 * cellType_temp(1);

% Random noise adjustment for the rightmost / maximum column
cellType(length(cellType) - 1) = cellType(length(cellType)-1) + 0.16 * cellType_temp(length(cellType));
cellType(length(cellType)) = cellType(length(cellType)) + 0.84 * cellType_temp(length(cellType));

end

