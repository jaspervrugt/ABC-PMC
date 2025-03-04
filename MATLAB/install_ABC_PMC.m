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

% Add main ABC_PMC directory and underlying postprocessing directory
addpath(pwd,[pwd,'/postprocessing'],[pwd,'/miscellaneous']);
% Now go to example 1
cd example_1

% And you can now execute example_1 using (uncomment)
% example_1

%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
% POSTPROCESSING: Please go to directory \PostProcessing and run the        
% "postprocABC" script. This will compute various statistics and create a 
% number of different plots, including marginal posterior parameter 
% distributions, two-dimensional correlation plots of the posterior 
% parameter samples, confidence and prediction limits, etc.
% NOTE: This postprocessing code has not been verified in latest clean up
%<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
