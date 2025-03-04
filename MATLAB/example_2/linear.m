function [S,y] = linear(theta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Linear regression: Toy example of Vrugt and Sadegh (2013)               %
%                                                                         %
% SYNOPSIS: [S,y] = linear(theta)                                         %
% where                                                                   %
%  theta       [input] 1x2 vector of parameter values                     %
%  S           [outpt] mean and std. summary metrics of regression data   %
%  y           [outpt] 100x1 vector of values of regression data          %
%                                                                         %
% Check the following paper                                               %
%  Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model             %
%      calibration and evaluation: Approximate Bayesian computation,      %
%      Water Resources Research, 49, pp. 4335–4345,                       %
%          https://doi.org/10.1002/wrcr.20354                             %
%                                                                         %
% (c) Written by Jasper A. Vrugt, Feb. 2010                               %
% University of California Irvine                                         %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

persistent x                        % Store in local memory

if isempty(x)
    x = linspace(0,10)';            % 100 equally spaced-values x in [0,10]
end
y = theta(1) * x + theta(2);        % linear transformation
y = normrnd(y,0.5);                 % add random error

S = [mean(y) std(y)];               % compute summary metrics

end
