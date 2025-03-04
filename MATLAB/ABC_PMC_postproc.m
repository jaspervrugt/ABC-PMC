% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%       A       BBBBBB     CCCCCCC    PPPPPPPPP   MMM       MMM  CCCCCCC  %
%      AA      BBBBBBBB   CCCCCCCCC   PPPPPPPPPP  MMM       MMM CCCCCCCCC %
%     AAAA     BBB   BBB  CCC         PPP     PPP MMM       MMM CCC       %
%    AAAAAA    BBB    BBB CC          PPP     PPP MMMM     MMMM CCC       %
%   AAA  AAA   BBB    BBB CCC         PPP     PPP MMMMM   MMMMM CCC       %
%   AAA  AAA   BBB   BBB  CCC      == PPPPPPPPPP  MMMMMM MMMMMM CCC       %
%   AAAAAAAA   BBBBBBBB   CCC      == PPPPPPPPP   MMMMMMMMMMMMM CCC       %
%  AAA    AAA  BBB   BBB  CCC         PPP         MMM       MMM CCC       %
%  AAA    AAA  BBB    BBB CCC         PPP         MMM       MMM CCC       %
% AAA      AAA BBB    BBB CCC         PPP         MMM       MMM CCC       %
% AAA      AAA BBB   BBB  CCCCCCCCC   PPP         MMM       MMM CCCCCCCCC %
% AAA      AAA BBBBBBBB    CCCCCCC    PPP         MMM       MMM  CCCCCCC  %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Postprocessing of the results of ABC_PMC                                %
%                                                                         %
% SYNOPSIS                                                                %
%  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err)           %
%  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin)    %
%  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin,... %
%      Par_info)                                                          %
%  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin,... %
%      Par_info,Meas_info)                                                %
% where                                                                   %
%  ABCPar       [input] Structure of algorithmic parameters               %
%   .N              Population size                                       %
%   .d              # decision variables (= # parameters)   [= from code] %
%   .T              # generations                           [= from code] %
%   .method         Method used to create proposals                       %
%     = 'RWM'       Random Walk Metropolis algorithm                      %
%     = 'DREAM'     Differential Evolution Adaptive Metropolis            %
%  Func_name    [input] Function (string) returns summary metrics         %
%  err          [input] 1xT vector with decreasing error tolerances PMC   %
%                        → # entries of this vector equals ABCPar.T       %
%  Par_info     [input] Parameter structure: ranges, initial/prior & bnd  %
%   .min            1xd-vector of min parameter values    DEF: -inf(1,d)  %
%   .max            1xd-vector of max parameter values    DEF: inf(1,d)   %
%   .boundhandling  Treat the parameter bounds or not?                    %
%     = 'reflect'   Reflection method                                     %
%     = 'bound'     Set to bound                                          %
%     = 'fold'      Folding [Vrugt&Braak: doi:10.5194/hess-15-3701-2011]  %
%     = 'none'      No boundary handling                  DEFault         %
%  Meas_info    [input] Structure with measurement information (fitting)  %
%   .S              Scalar/vector with summary metrics                    %
%  theta        [outpt] NxdxT array: N popsize, d paramtrs, T generations %
%  S            [outpt] NxnsxT array: N popsize, ns summary met, T genrts %
%  sigma_theta  [outpt] Σ matrix of proposal distribution each generation %
%  output       [outpt] Structure of fields summarizing PMC performance   %
%   .RunTime        Required CPU time                                     %
%   .func_eval      # total model evaluations each PMC pop./generation    %
%   .AR             Acceptance rate each PMC pop./generation              %
%   .ESS            Effective sample size each PMC pop./generation        %
%   .tot_func_eval  Total number of function evaluations                  %
%   .tot_AR         Acceptance rate last PMC population/generation        %
%   .w              NxT matrix with weights of samples each generation    %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%  BUILT-IN CASE STUDIES                                                  %
%   Example 1   Toy example from Sisson et al. (2007)                     %
%   Example 2   Linear regression example from Vrugt and Sadegh (2013)    %
%   Example 3   Hydrologic modeling using hydrograph functionals          %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% MATLAB CODE                                                             %
%  © Written by Jasper A. Vrugt                                           %
%    University of California Irvine                                      %
%  Version 1.0    July 2012                                               %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Find at which iteration the code terminated
ABCPar.d = size(sigma_theta,2);                     % # parameters
ABCPar.T = size(sigma_theta,3);                     % # generations
theta = theta(1:ABCPar.N,1:ABCPar.d,1:ABCPar.T);    % remove empty cells
err = err(1:ABCPar.T);                              % same for error thrhld
fntsize_labels = 18;                                % fontsize labels
fig_number = 1;                                     % set figure number

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% PRINT TO SCREEN SOME SIMPLE STATISTICS
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

AR = output.AR                                      % acceptance rate
TotFuncEval = output.tot_func_eval                  % # total func evals
RunTime = output.RunTime                            % Total CPU time
CORR = corrcoef(theta(1:ABCPar.N, ...               % Posterior corr matrix
    1:ABCPar.d,ABCPar.T))

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% RELATIONSHIP BETWEEN THE VALUE OF ERR AND POSTERIOR STANDARD DEVIATION
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% Plot relationship between "err" and posterior moments
STD = reshape(std(theta),ABCPar.d,ABCPar.T)';
% Now plot (each parameter a different color -- scaled according to prior)
figure(fig_number),plot(err',STD./repmat(Par_info.max - ...
    Par_info.min,ABCPar.T,1),'linewidth',2);
% Add a legend
evalstr = strcat('legend(''par. 1''');
% Each parameter a different color
for jj = 2:ABCPar.d
    % Add the other parameters
    evalstr = strcat(evalstr,',''par.',{' '},num2str(jj),'''');
end
% And now conclude with a closing bracket
evalstr = strcat(evalstr,');');
% Now evaluate the legend
eval(char(evalstr));
% Add xlabel
xlabel('$\epsilon$ value','fontsize',fntsize_labels,'interpreter','latex');
% Then add a y-label
ylabel('Posterior standard deviation','fontsize',fntsize_labels);
% Appropriate fontsize
set(gca,'fontsize',fntsize_labels);
% Update fig_number
fig_number = fig_number + 1;

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% HISTORGRAMS OF MARGINAL DENSITIES OF PARAMETERS
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% Plot the histograms (marginal density) of each parameter;
% What lay out of marginal distributions is desired subplot(r,t)
r = 3; t = 2;

% How many figures do we need to create with this layout?
N_fig = ceil( ABCPar.d / (r * t) ); counter = 1; j = 1;

% Open new figure
figure(fig_number);

% Now plot each parameter
while counter <= ABCPar.d

    % Check whether to open a new figure?
    if j == (r * t) + 1

        % Update fig_number
        fig_number = fig_number + 1;

        % Open new figure
        figure(fig_number);

        % Reset j to 1
        j = 1;

    end

    % Now create histogram
    [N,X] = hist(theta(1:ABCPar.N,counter,ABCPar.T));

    % And plot histogram in red
    subplot(r,t,j),bar(X,N/sum(N),'r'); hold on;
    % --> can be scaled to 1 if using "trapz(X,N)" instead of "sum(N)"!

    if j == 1
        % Add title
        title(['Histograms of marginal distributions of ' ...
            'individual parameters'],'fontsize',fntsize_labels);
    end

    % Add x-labels
    evalstr = strcat('Par',{' '},num2str(counter));
    xlabel(evalstr,'fontsize',fntsize_labels);

    % Then add y-label (only if j == 1 or j = r;
    if j == 1 || ( min(abs(j - ([1:r]*t+1))) == 0 )
        ylabel('Marginal density','fontsize',fntsize_labels);
    end

    % Now determine the min and max X values of the plot
    minX = min(X); maxX = max(X); minY = 0; maxY = max(N/sum(N));

    % Now determine appropriate scales
    deltaX = 0.1*(maxX - minX);

    % Calculate x_min and x_max
    x_min = minX - deltaX; x_max = maxX + deltaX;

    % Now determine the min and max Y values of the plot
    y_min = 0; y_max = 1.1*maxY;

    % Adjust the axis
    axis([x_min x_max y_min y_max]);

    % Check if counter = 1,
    if counter == 1 % --> add a title for first figure
        % Add title
        title(['Histograms of marginal distributions of ' ...
            'individual parameters'],'fontsize',fntsize_labels);
    end

    % Now update the counter
    counter = counter + 1;

    % Update j
    j = j + 1;

    % Appropriate fontsize
    set(gca,'Fontsize',fntsize_labels);

end

% Update fig_number
fig_number = fig_number + 1;

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% CORRELATION PLOTS OF THE POSTERIOR PARAMETER SAMPLES
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% Open a new plot
figure(fig_number); fig_number = fig_number + 1;

% Plot a matrix (includes unscaled marginals on main diagonal!
plotmatrix(theta(1:ABCPar.N,1:ABCPar.d,ABCPar.T),'r+');

% Add title
title(['Marginal distributions and two-dimensional correlation plots ' ...
    'of posterior parameter samples'],'fontsize',fntsize_labels);

% Appropriate fontsize
set(gca,'Fontsize',fntsize_labels);

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% CALCULATE THE RMSE OF THE BEST SOLUTION AND PLOT
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

% Now compute the simulations of theta values and RMSE, only if Meas_info
if exist('Meas_info','var')
    if isfield(Meas_info,'Y')
        % Initalize matrices
        Ysim = nan(Meas_info.n,ABCPar.N); RMSE = nan(1,ABCPar.N);
        for j = 1:ABCPar.N

            % Now evaluate the model and return summary statistics
            evalstr = ['[S_mod(j,:),Ysim(:,j)] = ',Func_name,...
                '(theta(j,1:ABCPar.d,ABCPar.T),Extra);']; eval(evalstr);

            % Compute the RMSE of the maximum aposteriori solution
            RMSE(1,j) = sqrt(sum((Ysim(:,j) - Meas_info.Y).^2) ...
                / Meas_info.n );
        end

        % What is now the best solution
        [dummy,ii] = min(RMSE);

        % And store parameter values
        MAP = theta(ii(1),1:ABCPar.d,ABCPar.T); disp(MAP)

        % Now plot histogram of RMSE values
        [N,X] = hist(RMSE,10); figure(fig_number),bar(X,N/sum(N),'r');

        % Now determine the min and max X values of the plot
        minX = min(X); maxX = max(X); minY = 0; maxY = max(N/sum(N));

        % Now determine appropriate scales
        deltaX = 0.1*(maxX - minX);

        % Calculate x_min and x_max
        x_min = minX - deltaX; x_max = maxX + deltaX;

        % Now determine the min and max Y values of the plot
        y_min = 0; y_max = 1.1*maxY;

        % Adjust the axis
        axis([x_min x_max y_min y_max]);

        % Add a title
        title('Histogram of posterior RMSE values', ...
            'fontsize',fntsize_labels);

        % Add x-label
        xlabel('RMSE [unit?]','fontsize',fntsize_labels);

        % Add y-label
        ylabel('Marginal density','fontsize',fntsize_labels);

        % Appropriate fontsize
        set(gca,'fontsize',fntsize_labels);

        % Update fig_number
        fig_number = fig_number + 1;
    end
end

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% PLOT THE 95% POSTERIOR SIMULATION UNCERTAINTY
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

if exist('Ysim','var')

    % Define significance level
    alfa = 0.05;
    % confidence level
    gamma = 1 - alfa;
    % lower bound (= integer) of sample for 100*gamma pred. limits
    lb = floor(alfa/2 * ABCPar.N); ub = ABCPar.N - lb;
    % Add RMSE to create total uncertainty (homoscedastic error!)
    Stot = Ysim + normrnd(0,min(RMSE),Meas_info.n,ABCPar.N);
    [par_unc,tot_unc] = deal(nan(N,1));
    % Now sort to get desired ranges
    for j = 1:Meas_info.n
        % Sort the model output from jth parameter vector low to high
        a = sort(Ysim(j,1:ABCPar.N));
        % And take the desired prediction uncertainty ranges
        par_unc(j,1) = a(lb); par_unc(j,2) = a(Ub);
        % Same with total uncertainty
        a = sort(Stot(j,1:ABCPar.N));
        % And take the desired prediction uncertainty ranges
        tot_unc(j,1) = a(lb); tot_unc(j,2) = a(Ub);
    end

    % Open new figure
    figure(fig_number),

    % Update fig_number
    fig_number = fig_number + 1;

    % We start with the total uncertainty
    Fill_Ranges(1:Meas_info.n,tot_unc(:,1),tot_unc(:,2), ...
        [0.75 0.75 0.75]);
    hold on;

    % And then plot the parameter uncertainty
    Fill_Ranges(1:Meas_info.n,par_unc(:,1),par_unc(:,2), ...
        [0.25 0.25 0.25]);

    % Now add the observations
    plot([1:Meas_info.n],Measurement.MeasData,'r.'); hold on;

    % Fit axes
    axis([0 Meas_info.n 0 1.1 * max(max(tot_unc))])

    % Add x-label
    xlabel('x','fontsize',fntsize_labels);

    % Add y-label
    ylabel('y','fontsize',fntsize_labels);

    % Add title
    title(['95% Posterior simulation uncertainty ranges ' ...
        '(homoscedastic error!)'],'fontsize',fntsize_labels,...
        'fontweight','bold');

    % Add a legend
    legend('total uncertainty','parameter uncertainty','observed data');

    % Which percentage is included in confidence limits
    Contained_pars = 100 * sum ( Meas_info.Y < par_unc(:,1) | ...
        Meas_info.Y > par_unc(:,2) ) / Meas_info.n;
    disp(Contained_Pars);
    % Percentage inside prediction limits
    Contained_tot = 100 * sum ( Meas_info.Y < tot_unc(:,1) | ...
        Meas_info.Y > tot_unc(:,2) ) / Meas_info.n;
    disp(Contained_tot);
    % This should be close to 100*gamma
    set(gca,'fontsize',fntsize_labels);

end
