% Script written by Daniel A. Cruz
% GRR approximation for GoL-like ABM
% See associated manuscript for more details

%% Parameters
clear;
close all;

% (Square) Environment length
env_length = 20;
envS = env_length^2;

% Ending time step
enT = 25;

% Starting Population
stPop = 800;

% Lower bound for living (transition)
lt = 2;

% Upper bound for living (transition)
ut = 8;

% Lower bound for production
lp = 2;

% Upper bound for production
up = 4;

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

%% Initialization
for i = lt:ut
    temp = subs(evalin(symengine, 'envS * nchoosek(stPop,(i+1)) * ((1/(envS))^(i+1)) * ((1-(1/(envS)))^(stPop-i-1))'));
    Rpatch(1) = Rpatch(1) + temp;
    Surv(1) = Surv(1) + (i+1)*temp;
    if (i >= lp) && (i <= up)
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
    for i = lt:ut
        temp = subs(evalin(symengine, 'tempPatch * nchoosek(tempPop,(i+1)) * ((1/(tempPatch))^(i+1)) * ((1-(1/(tempPatch)))^(tempPop-i-1))'));
        Rpatch(t) = Rpatch(t) + temp;
        Surv(t) = Surv(t) + (i+1)*temp;
        if (i >= lp) && (i <= up)
            Ofsp(t) = Ofsp(t) + (i+1)*temp;
        end
    end
    Pop(t) = Surv(t) + Ofsp(t);
    NgPrb(t) = (Rpatch(t)-1)/(envS - 1);
    Opatch(t) = Rpatch(t)*(2-NgPrb(t));
end

%% Compute and plot totals

plot(Pop);
