% Script written by Daniel A. Cruz
% GRR approximation for GoL-like ABM
% See associated manuscript for more details

%% Parameters
clear;
close all;

% (Square) Environment length
w = 20;

% Ending time step
enT = 25;

% Starting Population
n = 500;

% Lower bound for surviving (transition)
ls = 2;

% Upper bound for reproducing (transition)
us = 8;

% Lower bound for production
lr = 2;

% Upper bound for production
ur = 4;

%% State variables
% Remaining patch array
Rpatch = zeros([ enT 1]);
% Occupied patch array
Opatch = zeros([ enT 1]);

% Probability of an adjacent neighbor being occupied
NgPrb = zeros([ enT 1]);

% Survivor array
Surv = zeros([ enT 1]);
% Offspring array
Ofsp = zeros([ enT 1]);
% Population array
Pop = zeros([ enT 1]);

% Loop variables
envS = w^2;
l = min(ls,lr);
u = max(us,ur);

%% Initialization
for i = l:u
    temp = subs(evalin(symengine, 'envS * nchoosek(n,(i+1)) * ((1/(envS))^(i+1)) * ((1-(1/(envS)))^(n-i-1))'));
    Rpatch(1) = Rpatch(1) + temp;
    if (i >= ls) && (i <= us)
        Surv(1) = Surv(1) + (i+1)*temp;
    end
    if (i >= lr) && (i <= ur)
        Ofsp(1) = Ofsp(1) + (i+1)*temp;
    end
end
Pop(1) = Surv(1) + Ofsp(1);
NgPrb(1) = (Rpatch(1)-1)/(envS - 1);
Opatch(1) = Rpatch(1)*(2-NgPrb(1));

%% Iteration
for t = 2:enT
    tempPop = Pop(t-1);
    tempPatch = Opatch(t-1);
    for i = l:u
        temp = subs(evalin(symengine, 'tempPatch * nchoosek(tempPop,(i+1)) * ((1/(tempPatch))^(i+1)) * ((1-(1/(tempPatch)))^(tempPop-i-1))'));
        Rpatch(t) = Rpatch(t) + temp;
        if (i >= ls) && (i <= us)
            Surv(t) = Surv(t) + (i+1)*temp;
        end
        if (i >= lr) && (i <= ur)
            Ofsp(t) = Ofsp(t) + (i+1)*temp;
        end
    end
    Pop(t) = Surv(t) + Ofsp(t);
    NgPrb(t) = (Rpatch(t)-1)/(envS - 1);
    Opatch(t) = Rpatch(t)*(2-NgPrb(t));
end

%% Plot population total

plot(Pop);
