# ####################################################################### #
#                                                                         #
#       A       BBBBBB     CCCCCCC    PPPPPPPPP   MMM       MMM  CCCCCCC  #
#      AA      BBBBBBBB   CCCCCCCCC   PPPPPPPPPP  MMM       MMM CCCCCCCCC #
#     AAAA     BBB   BBB  CCC         PPP     PPP MMM       MMM CCC       #
#    AAAAAA    BBB    BBB CC          PPP     PPP MMMM     MMMM CCC       #
#   AAA  AAA   BBB    BBB CCC         PPP     PPP MMMMM   MMMMM CCC       #
#   AAA  AAA   BBB   BBB  CCC      == PPPPPPPPPP  MMMMMM MMMMMM CCC       #
#   AAAAAAAA   BBBBBBBB   CCC      == PPPPPPPPP   MMMMMMMMMMMMM CCC       #
#  AAA    AAA  BBB   BBB  CCC         PPP         MMM       MMM CCC       #
#  AAA    AAA  BBB    BBB CCC         PPP         MMM       MMM CCC       #
# AAA      AAA BBB    BBB CCC         PPP         MMM       MMM CCC       #
# AAA      AAA BBB   BBB  CCCCCCCCC   PPP         MMM       MMM CCCCCCCCC #
# AAA      AAA BBBBBBBB    CCCCCCC    PPP         MMM       MMM  CCCCCCC  #
#                                                                         #
# ####################################################################### #
# Approximate Bayesian Computation avoids the use of an explicit          #
# likelihood function in favor a (number of) summary statistics that      #
# measure the distance between the model simulation and the data. This    #
# ABC approach is a vehicle for diagnostic model calibration and          #
# evaluation for the purpose of learning and model correction. The PMC    #
# algorithm is not particularly efficient and hence we have alternative   #
# implementations that adaptively selects the sequence of epsilon values. #
# I recommend using the DREAM_(ABC) algorithm developed by Sadegh and     #
# Vrugt (2014). This code is orders of magnitude more efficient than the  #
# ABC-PMC method                                                          #
#                                                                         #
# SYNOPSIS                                                                #
#  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err)           #
#  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin)    #
#  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin,... #
#      Par_info)                                                          #
#  [theta,S,sigma_theta,output] = ABC_PMC(ABCPar,Func_name,err,plugin,... #
#      Par_info,Meas_info)                                                #
# where                                                                   #
#  ABCPar       [input] Structure of algorithmic parameters               #
#   .N              Population size                                       #
#   .d              # decision variables (= # parameters)   [= from code] #
#   .T              # generations                           [= from code] #
#   .method         Method used to create proposals                       #
#     = 'RWM'       Random Walk Metropolis algorithm                      #
#     = 'DREAM'     Differential Evolution Adaptive Metropolis            #
#  Func_name    [input] Function (string) returns summary metrics         #
#  err          [input] 1xT vector with decreasing error tolerances PMC   #
#                        → # entries of this vector equals ABCPar.T       # 
#  Par_info     [input] Parameter structure: ranges, initial/prior & bnd  #
#   .min            1xd-vector of min parameter values    DEF: -inf(1,d)  #
#   .max            1xd-vector of max parameter values    DEF: inf(1,d)   #
#   .boundhandling  Treat the parameter bounds or not?                    #
#     = 'reflect'   Reflection method                                     #
#     = 'bound'     Set to bound                                          #
#     = 'fold'      Folding [Vrugt&Braak: doi:10.5194/hess-15-3701-2011]  #
#     = 'none'      No boundary handling                  DEFault         #
#  Meas_info    [input] Structure with measurement information (fitting)  #
#   .S              Scalar/vector with summary metrics                    #
#  theta        [outpt] NxdxT array: N popsize, d paramtrs, T generations #
#  S            [outpt] NxnsxT array: N popsize, ns summary met, T genrts #
#  sigma_theta  [outpt] Σ matrix of proposal distribution each generation #
#  output       [outpt] Structure of fields summarizing PMC performance   #
#   .RunTime        Required CPU time                                     #
#   .func_eval      # total model evaluations each PMC pop./generation    #
#   .AR             Acceptance rate each PMC pop./generation              #
#   .ESS            Effective sample size each PMC pop./generation        #
#   .tot_func_eval  Total number of function evaluations                  #
#   .tot_AR         Acceptance rate last PMC population/generation        #
#   .w              NxT matrix with weights of samples each generation    #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  BUILT-IN CASE STUDIES                                                  #
#   Example 1   Toy example from Sisson et al. (2007)                     #
#   Example 2   Linear regression example from Vrugt and Sadegh (2013)    #
#   Example 3   Hydrologic modeling using hydrograph functionals          #
#                                                                         #
# ####################################################################### #
#                                                                         #
# ALGORITHM HAS BEEN DESCRIBED IN                                         #
#   Turner, B.M, and T. van Zandt (2012), A tutorial on approximate       #
#       Bayesian computation, Journal of Mathematical Psychology, 56,     #
#       pp. 69-85                                                         #
#   Sadegh, M., and J.A. Vrugt (2014), Approximate Bayesian computation   #
#       using Markov chain Monte Carlo simulation: DREAM_(ABC), Water     #
#       Resources Research,                                               #
#           https://doi.org/10.1002/2014WR015386                          #
#   Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model            #
#       calibration and evaluation: Approximate Bayesian computation,     #
#       Water Resources Research, 49, pp. 4335–4345,                      #
#           https://doi.org/10.1002/wrcr.20354                            #
#   Sadegh, M., and J.A. Vrugt (2013), Bridging the gap between GLUE and  #
#       formal statistical approaches: approximate Bayesian computation,  #
#       Hydrology and Earth System Sciences, 17, pp. 4831–4850, 2013      #
#                                                                         #
# FOR MORE INFORMATION, PLEASE READ                                       #
#   Vrugt, J.A., R. de Punder, and P. Grünwald, A sandwich with water:    #
#       Bayesian/Frequentist uncertainty quantification under model       #
#       misspecification, Submitted to Water Resources Research,          #
#       May 2024, https://essopenarchive.org/users/597576/articles/...    #
#           937008-a-sandwich-with-water-bayesian-frequentist-...         #
#           uncertainty-quantification-under-model-misspecification       #
#   Vrugt, J.A., R. de Punder, and P. Grünwald, A sandwich with water:    #
#       Bayesian/Frequentist uncertainty quantification under model       #
#       misspecification, Submitted to Water Resources Research,          #
#       May 2024, https://essopenarchive.org/users/597576/articles/...    #
#           937008-a-sandwich-with-water-bayesian-frequentist-...         #
#           uncertainty-quantification-under-model-misspecification       #
#   Vrugt, J.A. (2024), Distribution-Based Model Evaluation and           #
#       Diagnostics: Elicitability, Propriety, and Scoring Rules for      #
#       Hydrograph Functionals, Water Resources Research, 60,             #
#       e2023WR036710                                                     #
#           https://doi.org/10.1029/2023WR036710                          #
#   Vrugt, J.A., D.Y. de Oliveira, G. Schoups, and C.G.H. Diks (2022),    #
#       On the use of distribution-adaptive likelihood functions:         #
#       Generalized and universal likelihood functions, scoring rules     #
#       and multi-criteria ranking, Journal of Hydrology, 615, Part B,    #
#       2022, doi:10.1016/j.jhydrol.2022.128542.                          #
#           https://www.sciencedirect.com/science/article/pii/...         #
#           S002216942201112X                                             #
#   Vrugt, J.A. (2016), Markov chain Monte Carlo simulation using the     #
#       DREAM software package: Theory, concepts, and MATLAB              #
#       implementation, Environmental Modeling and Software, 75,          #
#       pp. 273-316, doi:10.1016/j.envsoft.2015.08.013                    #
#   Laloy, E., and J.A. Vrugt (2012), High-dimensional posterior          #
#       exploration of hydrologic models using multiple-try DREAM_(ZS)    #
#       and high-performance computing, Water Resources Research, 48,     #
#       W01526, doi:10.1029/2011WR010608                                  #
#   Vrugt, J.A., C.J.F. ter Braak, H.V. Gupta, and                        #
#       B.A. Robinson (2009), Equifinality of formal (DREAM) and          #
#       informal (GLUE) Bayesian approaches in                            #
#       hydrologic modeling? Stochastic Environmental Research and Risk   #
#       Assessment, 23(7), 1011-1026, doi:10.1007/s00477-008-0274-y       #
#   Vrugt, J.A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon,                #
#       B.A. Robinson, and J.M. Hyman (2009), Accelerating Markov chain   #
#       Monte Carlo simulation by differential evolution with             #
#       self-adaptive randomized subspace sampling, International         #
#       Journal of Nonlinear Sciences and Numerical Simulation, 10(3),    #
#       271-288                                                           #
#   Vrugt, J.A., C.J.F. ter Braak, M.P. Clark, J.M. Hyman, and            #
#       B.A. Robinson (2008), Treatment of input uncertainty in           #
#       hydrologic modeling: Doing hydrology backward with Markov chain   #
#       Monte Carlo simulation, Water Resources Research, 44, W00B09,     #
#       doi:10.1029/2007WR006720                                          #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  PYTHON CODE                                                            #
#  © Written by Jasper A. Vrugt using GPT-4 OpenAI's language model       #
#    University of California Irvine                                      #
#  Version 1.0    Dec 2024                                                #
#                                                                         #
# ####################################################################### #


import numpy as np
import time, array, os, sys                              
import multiprocessing
from scipy.stats import multivariate_normal             

# Add the present directory to the Python path
module_path = os.getcwd()
if module_path not in sys.path:
    sys.path.append(module_path)

# Add related functions to Python path
parent_directory = os.path.join(module_path, 'miscellaneous')
sys.path.append(parent_directory)

from ABC_PMC_functions import *


def ABC_PMC(ABCPar, Func_name, err, Par_info, Meas_info, *args):
    """
    Approximate Bayesian Computation using Population Monte Carlo (ABC-PMC) algorithm.
    
    Parameters:
    ABCPar : dict
        Algorithmic parameters, including:
        - 'N' (Population size)
        - 'd' (Number of parameters)
        - 'T' (Number of generations)
        - 'method' (Sampling method: 'RWM' or 'DREAM')
    Func_name : function
        Function handle that returns summary statistics for the model.
    err : array
        Vector with decreasing error tolerances.
    Par_info : dict
        Parameter bounds and other info. It should contain:
        - 'min' (Minimum parameter values)
        - 'max' (Maximum parameter values)
    Meas_info : dict
        Measurement info, including the summary statistics S.
    
    Returns:
    theta : ndarray
        Parameter samples for each generation.
    S : ndarray
        Summary statistics for each generation.
    sigma_theta : ndarray
        Covariance matrix of the proposal distribution for each generation.
    output : dict
        Performance metrics of the PMC algorithm.
    """

    plugin = None                   # Initialize options and plugin as empty
    if len(args) == 1:              # Handle input arguments
        plugin = args[0]

    # Print header information
    print('  -----------------------------------------------------------------------------            ')
    print('      AAA     BBBBBBBBB    CCCCCCCCCC     PPPPPPPPP   MMM        MMM CCCCCCCCCC            ')
    print('     AAAAA    BBBBBBBBBB   CCCCCCCCCC     PPPPPPPPPP  MMMM      MMMM CCCCCCCCCC            ')
    print('    AAA AAA   BBB     BBB  CCC            PPP     PPP MMMMM    MMMMM CCC                   ')
    print('   AAA   AAA  BBB     BBB  CCC            PPP     PPP MMMMMM  MMMMMM CCC                   ')
    print('  AAA     AAA BBB    BBB   CCC       ---- PPP     PPP MMM MMMMMM MMM CCC            /^ ^\  ')
    print('  AAAAAAAAAAA BBB    BBB   CCC       ---- PPPPPPPPPP  MMM  MMMM  MMM CCC           / 0 0 \ ')
    print('  AAA     AAA BBB     BBB  CCC            PPPPPPPPP   MMM   MM   MMM CCC           V\ Y /V ')
    print('  AAA     AAA BBB     BBB  CCC            PPP         MMM        MMM CCC            / - \  ')
    print('  AAA     AAA BBBBBBBBBB   CCCCCCCCCC     PPP         MMM        MMM CCCCCCCCCC    /     | ')
    print('  AAA     AAA BBBBBBBBB    CCCCCCCCCC     PPP         MMM        MMM CCCCCCCCCC    V__) || ')
    print('  -----------------------------------------------------------------------------            ')
    print('  © Jasper A. Vrugt, University of California Irvine & GPT-4 OpenAI''s language model')
    print('    ________________________________________________________________________')
    print('    Version 1.0, Dec. 2024, Beta-release: MATLAB implementation is benchmark')
    print('\n')

    # Initialize measurement info
    if isinstance(Meas_info['S'], (int, float)) or isinstance(Meas_info['S'], list):
        Meas_info['S'] = np.array(Meas_info['S'])
    elif isinstance(Meas_info['S'], array.array):
        Meas_info['S'] = np.array(Meas_info['S'])
    if isinstance(Par_info['min'], (int, float)) or isinstance(Par_info['min'], list):
        Par_info['min'] = np.array(Par_info['min'])
    elif isinstance(Par_info['min'], array.array):
        Par_info['min'] = np.array(Par_info['min'])
    if isinstance(Par_info['max'], (int, float)) or isinstance(Par_info['max'], list):
        Par_info['max'] = np.array(Par_info['max'])
    elif isinstance(Par_info['max'], array.array):
        Par_info['max'] = np.array(Par_info['max'])
    ABCPar['N'] = int(ABCPar['N'])

    # How many parameters?
    if np.ndim(Par_info['min']) == 0:
        ABCPar['d'] = 1
        Par_info['min'] = np.array([Par_info['min']])
        Par_info['max'] = np.array([Par_info['max']])
    else: 
        ABCPar['d'] = len(Par_info['min'])

    # How many summary statistics?
    if np.ndim(Meas_info['S']) == 0:
        nS = 1
    else: 
        nS = len(Meas_info['S'])

    # Algorithmic values
    ABCPar['T'] = len(err)                  # Number of error tolerances

    # Matrix R: Each chain stores indices of other chains available for DREAM
    R = np.zeros((ABCPar['N'], ABCPar['N'] - 1), dtype=int)
    for i in range(ABCPar['N']):
        R[i, :] = np.setdiff1d(np.arange(0, ABCPar['N']), i)

    # Create the function handle
    f_handle = globals()[Func_name]
    # Initialize variables
    theta = np.full((ABCPar['N'], ABCPar['d'], ABCPar['T']), np.nan)
    w = np.full((ABCPar['N'], ABCPar['T']), np.nan)
    S = np.full((ABCPar['N'], nS, ABCPar['T']), np.nan)

    # Set initial conditions
    t = 1
    ndraw = np.zeros(ABCPar['T'], dtype=int)
    scale_factor = 2.38 / np.sqrt(ABCPar['d'])
    scale_factor_DREAM = 2.38 / np.sqrt(2 * ABCPar['d'])

    # Prior distribution for theta
    pi_theta = np.prod(1. / (Par_info['max'] - Par_info['min']))

    # Start the timer
    start_time = time.time()

    # First generation
    rho = 2 * err[0]
    for i in range(0,ABCPar['N']):
        while rho > err[0]:
            theta_prop = Par_info['min'] + (Par_info['max'] - Par_info['min']) * np.random.rand(1, ABCPar['d'])
            if plugin is None:            
                results = f_handle(theta_prop)
            else:
                results = f_handle(theta_prop,plugin)

            if isinstance(results, (tuple, list)):
                S_mod = results[0]      # Summary metrics
                Y = results[1]          # 2nd output argument of Func_name 
            else:
                S_mod = results         # Summary metrics
                Y = None                # No 2nd output argument

            rho = np.max(np.abs(S_mod - Meas_info['S'] + np.random.normal(0, err[ABCPar['T'] - 1])))
            ndraw[t - 1] += 1

        theta[i, :, t - 1] = theta_prop
        S[i, :, t - 1] = S_mod
        w[i, t - 1] = 1 / ABCPar['N']  # Equal weights at the start
        rho = 2 * err[0]
        # Print progress
        print(f"ABC-PMC INITIAL GENERATION: CANDIDATE POINT {i+1} ACCEPTED: {100 * (i+1) / ABCPar['N']:.2f}% DONE", end='\r')

    print("\n")
    # Update the covariance of the proposal distribution for RWM
    sigma_theta = np.zeros((ABCPar['d'], ABCPar['d'], ABCPar['T']))
    sigma_theta[:, :, t - 1] = scale_factor * np.cov(theta[:ABCPar['N'], :, t - 1].T)
    R_matrix = np.linalg.cholesky(sigma_theta[:, :, t - 1] + 1e-5 * np.eye(ABCPar['d']))

    # Initialize effective sample size
    ESS = np.full(ABCPar['T'], np.nan)
    # Store the effective sample size at zeroth iteration
    ESS[0] = 1 / np.dot(w[:, t - 1].T, w[:, t - 1])

    # Sequential Monte Carlo sampling (generations)
    for t in range(1, ABCPar['T']):
        np.save(f'old_results_{t}.npy', theta)  # Save intermediate results

        for i in range(ABCPar['N']):
            while rho > err[t]:
                idx = np.random.choice(ABCPar['N'], p=w[:, t - 1])
                if ABCPar['method'] == 'RWM':
                    jump = np.random.randn(1, ABCPar['d']) @ R_matrix.T
                elif ABCPar['method'] == 'DREAM':
                    rr = np.random.choice(R[idx, :], size=2)
                    jump = scale_factor_DREAM * (theta[rr[0], :, t - 1] - theta[rr[1], :, t - 1]) + np.random.normal(0, 1, size=(1, ABCPar['d']))
                else:
                    raise ValueError(f"Unknown sampling method: {ABCPar['method']}")

                theta_prop = theta[idx, :, t - 1] + jump
                theta_prop = boundary_handling(theta_prop, Par_info)

                # Evaluate the model
                if plugin is None:            
                    results = f_handle(theta_prop)
                else:
                    results = f_handle(theta_prop,plugin)

                if isinstance(results, (tuple, list)):
                    S_mod = results[0]      # Summary metrics
                    Y = results[1]          # 2nd output argument of Func_name 
                else:
                    S_mod = results         # Summary metrics
                    Y = None                # No 2nd output argument

                rho = np.max(np.abs(S_mod - Meas_info['S'] + np.random.normal(0, err[ABCPar['T'] - 1])))
                ndraw[t] += 1

            theta[i, :, t] = theta_prop
            S[i, :, t] = S_mod
            # Compute the weight
            if ABCPar['method'] == 'RWM':
                #q = w[:ABCPar['N'], t - 1] * mvnpdf(theta[:ABCPar['N'], :, t - 1], theta[i, :, t], sigma_theta[:, :, t - 1])
                # Create a multivariate normal distribution object
                rv = multivariate_normal(mean=theta[i, :, t], cov=sigma_theta[:, :, t - 1])
                # Compute jump probability
                q = w[:ABCPar['N'], t - 1] * rv.pdf(theta[:ABCPar['N'], :, t - 1])
            elif ABCPar['method'] == 'DREAM':
                # Gaussian kernel density estimation for DREAM
                pass

            w[i, t] = pi_theta / np.sum(q)
            rho = 2 * err[t]
            # Print progress
            print(f"ABC-PMC {ordinal(t)} GENERATION: CANDIDATE POINT {i+1}: {100 * (i+1) / ABCPar['N']:.2f}% DONE", end='\r')

        print("\n")
        sigma_theta[:, :, t] = scale_factor * np.cov(theta[:ABCPar['N'], :, t].T)
        R_matrix = np.linalg.cholesky(sigma_theta[:, :, t] + 1e-5 * np.eye(ABCPar['d']))

        w[:, t] /= np.sum(w[:, t])              # Normalize the weights
        ESS[t] = 1 / (np.sum(w[:, t] ** 2))     # Effective sample size

    # End timer and summarize performance metrics
    output = {
        'RunTime': time.time() - start_time,
        'func_eval': ndraw,
        'AR': 100 * (ABCPar['N'] / ndraw),
        'ESS': ESS,
        'tot_func_eval': np.sum(ndraw),
        'tot_AR': 100 * (ABCPar['N'] / np.sum(ndraw)),
        'w': w
    }

    # Postprocessing of results
    fig_number = ABC_PMC_postprocessing(theta, S, err, output, Par_info, Meas_info)

    return theta, S, sigma_theta, output


## PARALLEL IMPLEMENTATION
def ABC_PMC_parallel(ABCPar, Func_name, err, Par_info, Meas_info, *args):
    """
    Approximate Bayesian Computation using Population Monte Carlo (ABC-PMC) algorithm with parallelization.
    """
    plugin = None                   # Initialize options and plugin as empty
    if len(args) == 1:              # Handle input arguments
        plugin = args[0]

    # Initialize measurement info
    if isinstance(Meas_info['S'], (int, float)) or isinstance(Meas_info['S'], list):
        Meas_info['S'] = np.array(Meas_info['S'])
    elif isinstance(Meas_info['S'], array.array):
        Meas_info['S'] = np.array(Meas_info['S'])
    if isinstance(Par_info['min'], (int, float)) or isinstance(Par_info['min'], list):
        Par_info['min'] = np.array(Par_info['min'])
    elif isinstance(Par_info['min'], array.array):
        Par_info['min'] = np.array(Par_info['min'])
    if isinstance(Par_info['max'], (int, float)) or isinstance(Par_info['max'], list):
        Par_info['max'] = np.array(Par_info['max'])
    elif isinstance(Par_info['max'], array.array):
        Par_info['max'] = np.array(Par_info['max'])
    ABCPar['N'] = int(ABCPar['N'])

    # How many parameters?
    if np.ndim(Par_info['min']) == 0:
        ABCPar['d'] = 1
        Par_info['min'] = np.array([Par_info['min']])
        Par_info['max'] = np.array([Par_info['max']])
    else: 
        ABCPar['d'] = len(Par_info['min'])

    # How many summary statistics?
    if np.ndim(Meas_info['S']) == 0:
        nS = 1
    else: 
        nS = len(Meas_info['S'])

    # Algorithmic values
    ABCPar['T'] = len(err)                  # Number of error tolerances

    # Matrix R for DREAM method
    R = np.zeros((ABCPar['N'], ABCPar['N'] - 1), dtype=int)
    for i in range(ABCPar['N']):
        R[i, :] = np.setdiff1d(np.arange(0, ABCPar['N']), i)

    # Create the function handle
    f_handle = globals()[Func_name]

    # Initialize variables
    theta = np.full((ABCPar['N'], ABCPar['d'], ABCPar['T']), np.nan)
    w = np.full((ABCPar['N'], ABCPar['T']), np.nan)
    S = np.full((ABCPar['N'], nS, ABCPar['T']), np.nan)
    ind_draws = np.full((ABCPar['N'], ABCPar['T']), np.nan)
    ndraws = np.zeros(ABCPar['T'], dtype=int)

    # Algorithmic values
    scale_factor = 2.38 / np.sqrt(ABCPar['d'])
    scale_factor_DREAM = 2.38 / np.sqrt(2 * ABCPar['d'])
    pi_theta = np.prod(1. / (Par_info['max'] - Par_info['min']))
    rho = 2 * err[0]

    # Start the timer
    start_time = time.time()

    # First generation (parallelize across population)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_initial_generation, [(i, f_handle, Meas_info, Par_info, err, ABCPar, plugin, rho) for i in range(ABCPar['N'])])

    for i, result in enumerate(results):
        theta[i, :, 0] = result[0]
        S[i, :, 0] = result[1]
        w[i, 0] = result[2]
        ind_draws[i, 0] = result[3]

    print("\n")

    # Update the covariance of the proposal distribution for RWM
    sigma_theta = np.zeros((ABCPar['d'], ABCPar['d'], ABCPar['T']))
    sigma_theta[:, :, 0] = scale_factor * np.cov(theta[:ABCPar['N'], :, 0].T)
    R_matrix = np.linalg.cholesky(sigma_theta[:, :, 0] + 1e-5 * np.eye(ABCPar['d']))

    # Initialize effective sample size
    ESS = np.full(ABCPar['T'], np.nan)
    # Store the effective sample size at zeroth iteration
    ESS[0] = 1 / np.dot(w[:, 0].T, w[:, 0])

    # Update the number of draws
    ndraws[0] = np.sum(ind_draws[:, 0])

    # Sequential Monte Carlo sampling (parallelize over generations)
    for t in range(1, ABCPar['T']):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_generation, [(i, f_handle, Meas_info, Par_info, ABCPar, t, err, rho, plugin, theta, sigma_theta, w, R_matrix, R, pi_theta, scale_factor_DREAM) for i in range(ABCPar['N'])])

        # Update theta, S, w from the results returned by the pool
        for i, result in enumerate(results):
            theta[i, :, t] = result[0]
            S[i, :, t] = result[1]
            w[i, t] = result[2]
            ind_draws[i, t] = result[3]

        # Update the covariance of the proposal distribution for RWM
        sigma_theta = np.zeros((ABCPar['d'], ABCPar['d'], ABCPar['T']))
        sigma_theta[:, :, t] = scale_factor * np.cov(theta[:ABCPar['N'], :, t].T)
        R_matrix = np.linalg.cholesky(sigma_theta[:, :, t] + 1e-5 * np.eye(ABCPar['d']))

        # Normalize the weights and compute effective sample size
        w[:, t] /= np.sum(w[:, t])
        ESS[t] = 1 / (np.sum(w[:, t] ** 2))

        # Update the number of draws
        ndraws[t] = np.sum(ind_draws[:, t])

    # End timer and summarize performance metrics
    output = {
        'RunTime': time.time() - start_time,
        'func_eval': ndraws,
        'AR': 100 * (ABCPar['N'] / ndraws),
        'ESS': ESS,
        'tot_func_eval': np.sum(ndraws),
        'tot_AR': 100 * (ABCPar['N'] / np.sum(ndraws)),
        'w': w
    }

    # Postprocessing of results
    fig_number = ABC_PMC_postprocessing(theta, S, err, output, Par_info, Meas_info)

    return theta, S, sigma_theta, output



def process_initial_generation(i, f_handle, Meas_info, Par_info, err, ABCPar, plugin, rho):
    """
    Helper function for processing the initial generation.
    """
    n_local_draw = 0
    while rho > err[0]:
        theta_prop = Par_info['min'] + (Par_info['max'] - Par_info['min']) * np.random.rand(1, ABCPar['d'])
        if plugin is None:
            results = f_handle(theta_prop)
        else:
            results = f_handle(theta_prop, plugin)

        S_mod = results[0]  # Summary metrics
        rho = np.max(np.abs(S_mod - Meas_info['S'] + np.random.normal(0, err[ABCPar['T'] - 1])))
        n_local_draw += 1
    #theta[i, :, 0] = theta_prop
    #S[i, :, 0] = S_mod
    #w[i, 0] = 1 / ABCPar['N']  # Equal weights at the start
    print(f"Initial Generation: Candidate point {i+1} accepted")

    return theta_prop, S_mod, 1 / ABCPar['N'], n_local_draw  # Return updated values

def process_generation(i, f_handle, Meas_info, Par_info, ABCPar, t, err, rho, plugin, theta, sigma_theta, w, R_matrix, R, pi_theta, scale_factor_DREAM):
    """
    Helper function for processing each generation (parallelized).
    """
    n_local_draw = 0
    while rho > err[t]:
        idx = np.random.choice(ABCPar['N'], p=w[:, t - 1])
        if ABCPar['method'] == 'RWM':
            jump = np.random.randn(1, ABCPar['d']) @ R_matrix.T
        elif ABCPar['method'] == 'DREAM':
            rr = np.random.choice(R[idx, :], size=2)
            jump = scale_factor_DREAM * (theta[rr[0], :, t - 1] - theta[rr[1], :, t - 1]) + np.random.normal(0, 1, size=(1, ABCPar['d']))
        else:
            raise ValueError(f"Unknown sampling method: {ABCPar['method']}")

        jump = np.random.randn(1, ABCPar['d']) @ R_matrix.T
        theta_prop = theta[idx, :, t - 1] + jump
        theta_prop = boundary_handling(theta_prop, Par_info)

        # Evaluate the model
        if plugin is None:
            results = f_handle(theta_prop)
        else:
            results = f_handle(theta_prop, plugin)

        S_mod = results[0]  # Summary metrics
        rho = np.max(np.abs(S_mod - Meas_info['S'] + np.random.normal(0, err[t])))
        n_local_draw += 1

    theta[i, :, t] = theta_prop
    # Compute the weight
    if ABCPar['method'] == 'RWM':
        #q = w[:ABCPar['N'], t - 1] * mvnpdf(theta[:ABCPar['N'], :, t - 1], theta[i, :, t], sigma_theta[:, :, t - 1])
        # Create a multivariate normal distribution object
        rv = multivariate_normal(mean=theta[i, :, t], cov=sigma_theta[:, :, t - 1])
        # Compute jump probability
        q = w[:ABCPar['N'], t - 1] * rv.pdf(theta[:ABCPar['N'], :, t - 1])
    elif ABCPar['method'] == 'DREAM':
        # Gaussian kernel density estimation for DREAM
        pass

    w = pi_theta / np.sum(q)

    print(f"Generation {t}: Candidate point {i+1} accepted")

    return theta_prop, S_mod, w, n_local_draw  # Return updated values
