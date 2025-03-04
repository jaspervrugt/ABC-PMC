% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE      333333   %
%   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE              33   %
%   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE          333   %
%   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              33   %
%   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE      333333   %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% Example 3: Hydrologic modeling with hydrograph functions                %
%  Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model             %
%      calibration and evaluation: Approximate Bayesian computation,      %
%      Water Resources Research, 49, pp. 4335–4345,                       %
%          https://doi.org/10.1002/wrcr.20354                             %
%                                                                         %
% Please check example 14 of DREAM_Package                                %
% = more efficient/better implementation                                  %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

clc; clear; close all hidden;           % clear workspace and figures

ABCPar.N = 50;                          % Population size
ABCPar.method = 'RWM';                  % Which method to create proposals

% parno         1      2      3      4     5      6      7    
% parname:      Imax  Smax  Qsmax   alE   alF   Kfast  Kslow  
Par_info.min = [  0    10     0    1e-6   -10     0      0  ];
Par_info.max = [ 10   1000   100   100     10     10    150 ];

Par_info.boundhandling  = 'reflect';    % Boundary handling 
Func_name = 'rainfall_runoff';          % Define modelname
daily_data = load('03451500.dly');      % Load daily discharge data
data.idx = (731:size(daily_data,1))';   % Two year warm-up period
Meas_info.S = ...                       % Compute measured summary metrics
    calc_metrics(daily_data ...
    (data.idx,6),daily_data(data.idx,4));
err = [ 1.00 0.75 0.50 0.25 ...         % Define the error tolerance
    0.10 0.05 0.025 ];

% Now run ABC population Monte carlo sampler
[theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err, ...
    Par_info,Meas_info);
