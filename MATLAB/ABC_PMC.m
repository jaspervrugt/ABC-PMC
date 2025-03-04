function [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err, ...
    Par_info,Meas_info)
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
% Approximate Bayesian Computation avoids the use of an explicit          %
% likelihood function in favor a (number of) summary statistics that      %
% measure the distance between the model simulation and the data. This    %
% ABC approach is a vehicle for diagnostic model calibration and          %
% evaluation for the purpose of learning and model correction. The PMC    %
% algorithm is not particularly efficient and hence we have alternative   %
% implementations that adaptively selects the sequence of epsilon values. %
% I recommend using the DREAM_(ABC) algorithm developed by Sadegh and     %
% Vrugt (2014). This code is orders of magnitude more efficient than the  %
% ABC-PMC method                                                          %
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
% ALGORITHM HAS BEEN DESCRIBED IN                                         %
%   Turner, B.M, and T. van Zandt (2012), A tutorial on approximate       %
%       Bayesian computation, Journal of Mathematical Psychology, 56,     %
%       pp. 69-85                                                         %
%   Sadegh, M., and J.A. Vrugt (2014), Approximate Bayesian computation   %
%       using Markov chain Monte Carlo simulation: DREAM_(ABC), Water     %
%       Resources Research,                                               %
%           https://doi.org/10.1002/2014WR015386                          %
%   Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model            %
%       calibration and evaluation: Approximate Bayesian computation,     %
%       Water Resources Research, 49, pp. 4335–4345,                      %
%           https://doi.org/10.1002/wrcr.20354                            %
%   Sadegh, M., and J.A. Vrugt (2013), Bridging the gap between GLUE and  %
%       formal statistical approaches: approximate Bayesian computation,  %
%       Hydrology and Earth System Sciences, 17, pp. 4831–4850, 2013      %
%                                                                         %
% FOR MORE INFORMATION, PLEASE READ                                       %
%   Vrugt, J.A., R. de Punder, and P. Grünwald, A sandwich with water:    %
%       Bayesian/Frequentist uncertainty quantification under model       %
%       misspecification, Submitted to Water Resources Research,          %
%       May 2024, https://essopenarchive.org/users/597576/articles/...    %
%           937008-a-sandwich-with-water-bayesian-frequentist-...         %
%           uncertainty-quantification-under-model-misspecification       %
%   Vrugt, J.A., R. de Punder, and P. Grünwald, A sandwich with water:    %
%       Bayesian/Frequentist uncertainty quantification under model       %
%       misspecification, Submitted to Water Resources Research,          %
%       May 2024, https://essopenarchive.org/users/597576/articles/...    %
%           937008-a-sandwich-with-water-bayesian-frequentist-...         %
%           uncertainty-quantification-under-model-misspecification       %
%   Vrugt, J.A. (2024), Distribution-Based Model Evaluation and           %
%       Diagnostics: Elicitability, Propriety, and Scoring Rules for      %
%       Hydrograph Functionals, Water Resources Research, 60,             %
%       e2023WR036710                                                     %
%           https://doi.org/10.1029/2023WR036710                          %
%   Vrugt, J.A., D.Y. de Oliveira, G. Schoups, and C.G.H. Diks (2022),    %
%       On the use of distribution-adaptive likelihood functions:         %
%       Generalized and universal likelihood functions, scoring rules     %
%       and multi-criteria ranking, Journal of Hydrology, 615, Part B,    %
%       2022, doi:10.1016/j.jhydrol.2022.128542.                          %
%           https://www.sciencedirect.com/science/article/pii/...         %
%           S002216942201112X                                             %
%   Vrugt, J.A. (2016), Markov chain Monte Carlo simulation using the     %
%       DREAM software package: Theory, concepts, and MATLAB              %
%       implementation, Environmental Modeling and Software, 75,          %
%       pp. 273-316, doi:10.1016/j.envsoft.2015.08.013                    %
%   Laloy, E., and J.A. Vrugt (2012), High-dimensional posterior          %
%       exploration of hydrologic models using multiple-try DREAM_(ZS)    %
%       and high-performance computing, Water Resources Research, 48,     %
%       W01526, doi:10.1029/2011WR010608                                  %
%   Vrugt, J.A., C.J.F. ter Braak, H.V. Gupta, and                        %
%       B.A. Robinson (2009), Equifinality of formal (DREAM) and          %
%       informal (GLUE) Bayesian approaches in                            %
%       hydrologic modeling? Stochastic Environmental Research and Risk   %
%       Assessment, 23(7), 1011-1026, doi:10.1007/s00477-008-0274-y       %
%   Vrugt, J.A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon,                %
%       B.A. Robinson, and J.M. Hyman (2009), Accelerating Markov chain   %
%       Monte Carlo simulation by differential evolution with             %
%       self-adaptive randomized subspace sampling, International         %
%       Journal of Nonlinear Sciences and Numerical Simulation, 10(3),    %
%       271-288                                                           %
%   Vrugt, J.A., C.J.F. ter Braak, M.P. Clark, J.M. Hyman, and            %
%       B.A. Robinson (2008), Treatment of input uncertainty in           %
%       hydrologic modeling: Doing hydrology backward with Markov chain   %
%       Monte Carlo simulation, Water Resources Research, 44, W00B09,     %
%       doi:10.1029/2007WR006720                                          %
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

% Now define the number of data points we have
Meas_info.nS = size(Meas_info.S,2);

% Algorithmic values
ABCPar.d                = size(Par_info.min,2);         % # parameters 
ABCPar.T                = size(err,2);                  % # "err" values
% Matrix DREAMPar.R: Each chain stores indices other chains available DREAM
for i = 1:ABCPar.N
    ABCPar.R(i,1:ABCPar.N-1) = setdiff(1:ABCPar.N,i); 
end

% Now create the function handle
f_handle = eval(['@(x)',char(Func_name),'(x)']);

% Define rho(S,S_mod) to start with so we go in first while loop
rho = 2 * err(1);

% Set initial theta, weights and summary statistic matrix (array)
theta = NaN(ABCPar.N,ABCPar.d,ABCPar.T); 
w = NaN(ABCPar.N,ABCPar.d); 
S = NaN(ABCPar.N,Meas_info.nS,ABCPar.T);

% Set t and ndraw to 0;
t = 1; ndraw = zeros(1,ABCPar.T);

% Define the scale factor of the proposal distribution
scale_factor = 2.38/sqrt(ABCPar.d);
% Note "2" is optimal scaling factor according to Kullback measure 
% (but with "1" dimension in paper)
% See Robert, Beaumont et al., 10.1.1.153.7886.pdf "Adaptivity for A
% BC algorithms: the ABC-PMC scheme"
scale_factor_DREAM = 2.38/sqrt(2 * ABCPar.d); 
% scale_factor * sqrt( 1 * ABCPar.d) / sqrt( 2 * ABCPar.d ); 
%   --> (2.38/sqrt(2)) / (2.38/sqrt(1))
% scale_factor_DREAM = 1;

% We use a simple prior distribution, so pi(theta) is simply
pi_theta = prod(1./(Par_info.max - Par_info.min));

% Now start with timer!
start_time = cputime;

% Now iterate --> t = 1;
for i = 1:ABCPar.N
    disp(i)
    % Continue sampling until rho is smaller than error tolerance
    while rho > err(1)  
        % Draw theta_0 from prior distribution of theta
        theta_prop = Par_info.min + (Par_info.max - ...
            Par_info.min).* rand(1,ABCPar.d);
        % Now evaluate theta_prop
        S_mod = f_handle(theta_prop);
        % Now calculate the distance metric
        rho = max( abs ( S_mod - Meas_info.S + ...
            normrnd( 0 , err(ABCPar.T) ) ) );
        % Update ndraw
        ndraw(t) = ndraw(t) + 1;
    end
    % Now store theta, because this theta has a rho < err(1),
    theta(i,1:ABCPar.d,t) = theta_prop;
    % Store the S_mod values in S
    S(i,1:Meas_info.nS,t) = S_mod;
    % Set the weight --> random sampling all weights are equal to 1 
    % (basically importance sampling with poor prior)
    w(i,t) = 1/ABCPar.N; % --> you can actually define this vector of 
    % weights prior to the loop!
    % Reset rho to larger than err(1) so we go back in while loop!
    rho = 2 * err(1);

end

% Update the covariance of the proposal distribution (for RWM)
sigma_theta(:,:,t) = scale_factor * cov(theta(1:ABCPar.N,1:ABCPar.d,1));
% To create proposals with RWM
R = chol(sigma_theta(:,:,t) + 1e-5 * eye(ABCPar.d));
% Calculate the effective sample size
ESS(t) = 1 / ( w(:,t)' * w(:,t) );

% Now do sequential monte carlo sampling
for t = 2 : ABCPar.T
    % Save file
    evalstr = strcat('save old_results_',num2str(t)); eval(evalstr);
    % Loop from 1 to N
    for i = 1 : ABCPar.N
        % Continue sampling while rho is larger than error tolerance
        while rho > err(t)
            % Draw from accepted theta's from previous iteration
            idx = find ( mnrnd ( 1 , w ( 1 : ABCPar.N , t - 1 ) ) == 1 );
            % Now determine which sampling method to use?
            switch ABCPar.method
                case 'RWM'
                    % Take this theta value and perturb it with var_theta 
                    % using normal distribution
                    jump = randn ( 1 , ABCPar.d ) * R;
                case 'DREAM'
                    % Now sample r1 and r2 from [1..N] without idx
                    rr = randsample(ABCPar.R(idx,1:N-1),2);
                    % Now create proposal point
                    jump = scale_factor_DREAM * ( theta(rr(1), ...
                        1:ABCPar.d,t-1) - theta(rr(2),1:ABCPar.d,t-1) ) ...
                        + eps * normrnd ( 1 , ABCPar.d );
                otherwise
                    % Define error
                    disp(['ABC_PMC:ERROR - UNKNOWN SAMPLING METHOD: ' ...
                        'DEFINE ABCPAR.method']); return;
            end
            % Now create proposal
            theta_prop = theta(idx,1:ABCPar.d,t-1) + jump;
            % Boundary handling to make sure parameters stay within bound
            theta_prop = boundary_handling(theta_prop,Par_info);
            % Now evaluate theta_prop
            S_mod = f_handle(theta_prop);
            % Now calculate the distance metric
            rho = max( abs ( S_mod - Meas_info.S + ...
                normrnd( 0 , err(ABCPar.T) ) ) );
            % Update the number of draws
            ndraw(t) = ndraw(t) + 1;
        end % proposal > err loop

        % Now store theta, because this theta has a rho < err(1),
        theta(i,1:ABCPar.d,t) = theta_prop;
        % Store the S_mod values in S
        S(i,1:Meas_info.nS,t) = S_mod;
        % Print t and i to screen
        disp([t i])
        % Calculate the weight of this new theta --> no longer uniform
        % weight as we have directed theta search; no longer equal chance
        % for each theta draw from the prior as in first step
        switch ABCPar.method
            case 'RWM'
                % Vectorization --> calculate the jump probability of
                % theta_prop given the accepted samples from
                % previous iteration
                q = w(1:ABCPar.N,t-1) .* mvnpdf(theta(1:ABCPar.N, ...
                    1:ABCPar.d,t-1),repmat(theta(i,1:ABCPar.d,t), ...
                    ABCPar.N,1),sigma_theta(:,:,t-1));
            case 'DREAM'
                % Vectorized solution --> much faster; scaling with prior
                % is needed to make sure that all dimensions count equally
                % delta = ( theta(1:ABCPar.N,1:ABCPar.d,t-1) - ...
                %     repmat(theta(i,1:ABCPar.d,t),ABCPar.N,1) ) ./ ...
                %     repmat( Par_info.max - Par_info.min , ABCPar.N , 1 );

                % Use a nonparametric Gaussian kernel density estimator
                % q = w(1:ABCPar.N,t-1) .* exp(-1/2*100*sum(delta.^2,2));
                % Check J.N Hwang, S.R Lay, and A. Lippman, Nonparametric 
                % Multivariate Density Estimation: 
                % A Comparative Study, IEEE Trans. Sig. Proc., 42(10), 
                % pp. 2795-2810
                % Note: kernel width (h) is assumed 1, and other constants 
                % have been removed from calculation 
                %   --> normalization of weights will remove their effect

                % Try full approach
                for ii = 1:ABCPar.N
                    % Calculate possible proposals for each ii
                    proposals = repmat(theta(idx,1:ABCPar.d,t-1), ...
                        ABCPar.N,1) + scale_factor_DREAM * ...
                        ( repmat(theta(ii,1:ABCPar.d,t-1),ABCPar.N,1) - ...
                        theta(1:ABCPar.N,1:ABCPar.d,t-1) );
                    % Calculate q_tot --> mvnpdf thus we should covariance
                    q(ii,1) = w(ii,t-1) * sum ( mvnpdf( repmat( ...
                        theta(i,1:ABCPar.d,t),ABCPar.N,1),proposals, ...
                        eps * eye(ABCPar.d) ) );
                end
                % If number of dimensions becomes really large (> 250) then
                % calculate log of kernel -- scale with maximum log-value
                % and then rederive the weights by exp(.). This scales all
                % the different log values linearly so does not affect the
                % weight values -- just a computational trick               
            otherwise
                % Define error
                disp('ABC_PMC:ERROR - CANNOT CALCULATE JUMP PROBABILITY'); 
                return
        end
        % Now calculate the weight of theta(i,t) ( = theta_prop)
        w(i,t) = pi_theta/sum(q);
        % Reset rho so we go back in while loop!
        rho = 2 * err(t);
    end % for i = 1:N loop

    % Update covariance of proposal distribution based on accepted draws
    sigma_theta(:,:,t) = scale_factor * cov(theta(1:ABCPar.N,1:ABCPar.d,t));
    % Note the factor "2" is optimal according to Kullback measure
    % To create proposals with RWM
    R = chol(sigma_theta(:,:,t) + 1e-5 * eye(ABCPar.d));
    % And normalize the importance weights so that sum of weights becomes 1
    w(1:ABCPar.N,t) = w(1:ABCPar.N,t)./sum(w(1:ABCPar.N,t));
    % Calculate the effective sample size
    ESS(t) = 1/(w(:,t)' * w(:,t)); disp(ESS(t))
           
end

% Required CPU time
output.RunTime = cputime - start_time;
% # total model evaluations?
output.func_eval = ndraw;
% Acceptance rate
output.AR = 100 * (ABCPar.N./ndraw);
% Effective sample size
output.ESS = ESS;
% # function evaluations
output.tot_func_eval = sum(ndraw);
% Acceptance rate to get last population
output.tot_AR = 100 * ( ABCPar.N / sum(ndraw) );
% Return the weights
output.w = w;

end
