% Script written by Daniel A. Cruz
% GRR approximation for rib ABM from https://doi.org/10.7554/eLife.29144
% See associated manuscript for more details

%% Parameters
clear;
close all;

y_max = 17;
x_max = 64;

localfate = true;

initsizemult = 1;

shh_xport = 12;

shh_intensity_log = 0.6;

% nticks
t_max = 46;

% pRed1
pRed = 0.4;

% pBlue1
pBlue = 0.4;

celldeathmult = 0.3;

prolifratemult = 1;

cdduration = 30;

%% State variables

Yellow = zeros(t_max, x_max+1);
Red = zeros(t_max, x_max+1);
Blue = zeros(t_max, x_max+1);
Cells = zeros(t_max, x_max+1);
Pressure = zeros(1, x_max+4);

% vx and dPdx
x_Density = zeros(1, x_max+1);

%% Set up

% shhC
shh = zeros(1, x_max+1);
% Set up SHH intensity gradient
for i=1:length(shh)
    shh(i)=exp(shh_intensity_log)*(exp(0 - ((((i-1) + 10)/shh_xport)^2)))/exp(0 - ((10/shh_xport)^2));
end
% Yellow to Red probabilities
shhR = pRed * shh;
shhR = min(shhR, 1);
% Yellow to Blue probabilities
shhB = pBlue * (1 - shh);
shhB = max(shhB, 0);

% Set up initial array of "Yellow" cells
init_Cells=initsizemult*1200;
init_x=ceil(14*sqrt(initsizemult));
init_Yellow=zeros(1,x_max+1);
for i=1:init_x
    init_Yellow(i)=init_Cells/init_x;
end

%% Go

% Time step or tick
t = 1;

while (Pressure(x_max-1) < 3) && t < t_max
    % Decide cells
    if t == 1
        [Yellow(t,:), Red(t,:), Blue(t,:)] = decideCells( init_Yellow, zeros(1,x_max+1), zeros(1,x_max+1), prolifratemult, shhR, shhB, celldeathmult, cdduration, localfate, t );
    else 
        [Yellow(t,:), Red(t,:), Blue(t,:)] = decideCells( Yellow(t-1,:), Red(t-1,:), Blue(t-1,:), prolifratemult, shhR, shhB, celldeathmult, cdduration, localfate, t );
    end
    
    % Update fields
    [Pressure, x_Density] = updateFields( Yellow(t,:), Red(t,:), Blue(t,:), y_max );
    
    i=0;
    while (max(Pressure) > 6) && (i < 100)
        % Move cells
        Yellow(t,:) = moveCells( Yellow(t,:), Pressure, x_Density );
        Red(t,:) = moveCells( Red(t,:), Pressure, x_Density );
        Blue(t,:) = moveCells( Blue(t,:), Pressure, x_Density );
        
        % Update fields
        [Pressure, x_Density] = updateFields( Yellow(t,:), Red(t,:), Blue(t,:), y_max );
        
        i = i + 1;
    end

    Cells(t,:) = Yellow(t,:) + Red(t,:) + Blue(t,:);
    t = t + 1;
end

%% Compute and plot totals

yTotal=zeros((t-1),1);
rTotal=yTotal;
bTotal=yTotal;

for j=1:(t-1)
    yTotal(j)=sum(Yellow(j,:));
    rTotal(j)=sum(Red(j,:));
    bTotal(j)=sum(Blue(j,:));
end

Total = [bTotal rTotal yTotal];
plot(Total);
