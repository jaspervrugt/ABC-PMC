function S = calc_metrics(y,P)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Calculate the summary metrics of measured/simulated discharge record    %
%                                                                         %
% SYNOPSIS: S = calc_metrics(y,P)                                         %
% where                                                                   %
%  y           [input] nx1 vector of discharge values                     %
%  P           [input] nx1 vector of precipitation values                 %
%  S           [outpt] 1x4 vector of summary metrics streamflow record, y %
%   S(1)           Annual runoff index                                    %
%   S(2)           Annual baseflow index                                  %
%   S(3)           Air-entry (= par 1) of 2-parmtr VG-inspired FDC exp.   %
%   S(4)           Slope (= par 2) of 2-parmtr VG-inspired FDC expression %
%                  --> FDCFIT toolbox has many more options & functnlties %
%                                                                         %
% Check the following paper                                               %
%   Vrugt, J.A. (2018), FDCFIT: Software toolbox for fitting a large      %
%       class of flow duration curves to streamflow data, manual          %
%   Sadegh, M., J.A. Vrugt, H.V. Gupta, and C. Xu (2016), The soil water  %
%       characteristic as new class of closed-form parametric expressions %
%       for the flow duration curve, J. Hydrol., 535, pp. 438–456         %
%                                                                         %
% (c) Written by Jasper A. Vrugt, March 2014                              %
% Los Alamos National Laboratory 			        	                  %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Determine the size of the data
ny = numel(y);
% Sort flows & calc exceedance probs using Weibull plotting position
[y_s,E_p] = calc_FDC(y,ny,1); 
x0 = [1 1];                         % Initial parameter values Nelder-Mead
S(3:4) = fminsearch(@(x) ...        % Fit the FDC using van Genuchten WRF
    VG_Ret(x,y_s,E_p,ny),x0);       % --> see Sadegh et al., JofH, 2015

% Now derive the annual runoff coefficient
S(1) = sum(y)/sum(P);
% Initialize baseflow components
yb = nan(ny,1); yb(1) = 0.25 * y(1); 
% Derive annual baseflow coefficient with low-pass filter
phi = 0.925;
% Now loop
for j = 2:ny
    yb(j,1) = min( phi * yb(j-1,1) + (1/2) * (1 - phi) * ...
        ( y(j,1) + y(j-1,1) ) , y(j,1));
end
% Annual baseflow coefficient
S(2) = sum(yb)/sum(y);

end
% <><><><><><><><><><><><> End of primary function <><><><><><><><><><><><>

% <><><><><><><><><><><><><> Secondary functions <><><><><><><><><><><><><>
% SUBROUTINE 1
function [y_s,E_p] = calc_FDC(Y,nY,method)
% Calculate exceedance probability of each flow level

switch method
    case 1
        y_s = sort(Y);			        % Sort discharge in ascending order
        E_p = 1 - ((1:nY) - .5)'./nY;   % Weibull exceedance probabilities
    otherwise
        [F,y_s] = ecdf(Y);              % Built-in empirical CDF
        E_p = 1 - F;                    % Exceedance probabilities
                                        % --> see Vrugt, WRR, 2024
end

end

% <><><><><><><><><><><><><> Secondary functions <><><><><><><><><><><><><>
% SUBROUTINE 2
function [RMSE,E_p] = VG_Ret(pars,y_s,E_ptrue,nY)
% Now compute exceedance probabilities according to VG WRF and values of 
% pars. Next, compute the RMSE between the actual, E_ptrue, and the VG 
% predicted, E_ppred, exceedance probabilities. The FDCFIT toolbox gives
% access to a much larger class of FDC functions. 

alpha = pars(1); n = pars(2);               % Extract parameter values
E_p = (1 + (alpha * y_s).^n).^(1/n - 1);	% E_p according to VG WRF
RMSE = sqrt(sum((E_p - E_ptrue).^2)/nY);	% Calculate RMSE

end
