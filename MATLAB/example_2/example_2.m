% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE      222222   %
%   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE          22 22    %
%   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE         22     %
%   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE           22      %
%   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE      222222   %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% Example 1: Linear regression example from Vrugt and Sadegh (2013)       %
%  Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model             %
%      calibration and evaluation: Approximate Bayesian computation,      %
%      Water Resources Research, 49, pp. 4335–4345,                       %
%          https://doi.org/10.1002/wrcr.20354                             %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

clc; clear; close all hidden;           % clear workspace and figures

ABCPar.N = 50;                          % Population size
ABCPar.method = 'RWM';                  % Which method to create proposals

Par_info.min = [ 0 -10 ];               % Minimum values of parameters
Par_info.max = [ 5  10 ];               % Maximum values of parameters
Par_info.boundhandling  = 'reflect';    % Boundary handling 
Func_name = 'linear';                   % Define modelname
y = 0.5 * linspace(0,10)' + 5;          % Generate synthetic data
y = y + normrnd(0,0.5,100,1);           % Add a random error
Meas_info.S = [ mean(y) std(y) ];       % Define observed summary metrics
err = [ 1.00 0.75 0.50 0.25 ...         % Define the error tolerance
    0.10 0.05 0.025 ];

% Now run ABC population Monte carlo sampler
[theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err, ...
    Par_info,Meas_info);

% Generate screen output
ABC_PMC_postproc
