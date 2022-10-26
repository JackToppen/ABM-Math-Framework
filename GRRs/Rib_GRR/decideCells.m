function [yl, rd, bl] = decideCells( yl_prev, rd_prev, bl_prev, proMult, shhR, shhB, cdMult, cdDuration, fate, t )
% Function written by Daniel A. Cruz
% Used in GRR approximation for rib ABM from https://doi.org/10.7554/eLife.29144
% See associated manuscript for more details
%DECIDECELLS 

% Cells divide first
ylHatch = proMult * 0.05 * yl_prev;
rdHatch = proMult * 0.05 * rd_prev;
blHatch = proMult * 0.05 * bl_prev;

YtoR = ylHatch.*shhR;
YtoB = ylHatch.*shhB;

ylTemp = yl_prev + ylHatch - 2*YtoR - 2*YtoB;
rdTemp = rd_prev + rdHatch + 2*YtoR;
blTemp = bl_prev + blHatch + 2*YtoB;

% Cells then die
ylTemp = ylTemp - 0.05 * cdMult * exp (0 - (t/cdDuration)^2) * ylTemp;
rdTemp = rdTemp -  0.05 * cdMult * exp (0 - (t/cdDuration)^2) * rdTemp;
blTemp = blTemp -  0.05 * cdMult * exp (0 - (t/cdDuration)^2) * blTemp;

% And finally cells may change type
RtoB = zeros(1,length(yl_prev));
BtoR = zeros(1,length(yl_prev));
if fate
    for i=1:length(yl_prev)
        if (blTemp(i) / (rdTemp(i) + blTemp(i))) > 0.6
            RtoB(i) = rdTemp(i);
        end
        if (rdTemp(i) / (rdTemp(i) + blTemp(i))) > 0.6
            BtoR(i) = blTemp(i);
        end
    end
end

yl = ylTemp;
rd = rdTemp - RtoB + BtoR;
bl = blTemp + RtoB - BtoR;

end

