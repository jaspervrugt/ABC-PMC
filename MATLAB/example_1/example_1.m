% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE        1111   %
%   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE           11 11   %
%   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE       11  11   %
%   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              11   %
%   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE          11   %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% Example 1: Mixture toy example from Sisson et al. (2007)                %
%  Sisson, S.A., Y. Fan, and M.M. Tanaka (2007), Sequential Monte Carlo   %
%      without likelihoods, Proceedings of the National Academy of        %
%      Sciences of the United States of America, 104(6), pp. 1760 - 1765, %
%          https://doi.org/10.1073/pnas.0607208104                        %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

clc; clear; close all hidden;           % clear workspace and figures

ABCPar.N = 50;                          % Population size
ABCPar.method = 'RWM';                  % Which method to create proposals

Par_info.min = -10;                     % Minimum value of parameter
Par_info.max = 10;                      % Maximum value of parameter
Par_info.boundhandling = 'reflect';     % Boundary handling 

Meas_info.S = 0;                        % Define observed summary metrics
Func_name = 'mixture';                  % Define modelname
    
err = [ 1.00 0.75 0.50 0.25 ...         % Define the error tolerance
    0.10 0.05 0.025 ];

% Now run ABC population Monte carlo sampler
[theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err, ...
    Par_info,Meas_info);

% Generate screen output
ABC_PMC_postproc
