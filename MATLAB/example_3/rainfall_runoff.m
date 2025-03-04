function [S_mod,Y] = rainfall_runoff(theta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Rainfall runoff model (= hmodel) using C++ executable with time         %
% variable integration time step. This C-code was written by Gerrit       %
% Schoups and modified by J.A. Vrugt. Similar implementations are         %
% available for Hymod, GR4J, NAM, HyMOD and the SAC-SMA model             %
%                                                                         %
% MATLAB CODE                                                             %
%  © Written by Jasper A. Vrugt                                           %
%    University of California Irvine                                      %
%  Version 1.0    July 2012                                               %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

persistent data options y0 tout

if isempty(data)
    % Load the French Broad data
    daily_data = load('03451500.dly');
    % First two years are warm-up
    data.idx = (731:size(daily_data,1))';
    % Define the PET, Measured Streamflow and Precipitation.
    data.P = daily_data(:,4); data.Ep = daily_data(:,5);
    % percolation coefficient
    data.aS = 1e-6;                  
    % Integration options
    options.InitialStep = 1;                 % initial time-step (d)
    options.MaxStep     = 1;                 % maximum time-step (d)
    options.MinStep     = 1e-6;              % minimum time-step (d)
    options.RelTol      = 1e-3;              % relative tolerance
    options.AbsTol      = 1e-3*ones(5,1);    % absolute tolerances (mm)
    options.Order       = 2;                 % 2nd order accurate method (Heun)
    % Initial conditions
    y0 = 1e-5 * ones(5,1);
    % Running time
    tout = 0:size(data.P,1);
end

% Assign parameters
data.Imax  = theta(1);      % interception storage capacity (mm)
data.Sumax = theta(2);      % unsaturated zone storage capacity (mm)
data.Qsmax = theta(3);      % maximum percolation rate (mm/d)
data.aE    = theta(4);      % evaporation coefficient
data.aF    = theta(5);      % runoff coefficient
data.Kf    = theta(6);      % fast-flow response time (d)
data.Ks    = theta(7);      % slow-flow response time (d)

% Run model C
Y = crr_model(tout,y0,data,options);
% Now compute discharge from infinite reservoir
y = diff(Y(5,1:end))'; y = y(data.idx);
% Now calculate the summary metrics
S_mod = calc_metrics(y,data.P(data.idx));

end
