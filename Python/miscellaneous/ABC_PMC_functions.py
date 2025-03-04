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


import numpy as np                                      # type: ignore
import matplotlib.pyplot as plt                         # type: ignore
import os                     
from matplotlib.backends.backend_pdf import PdfPages    # type: ignore
from scipy.stats import multivariate_normal             # type: ignore
from screeninfo import get_monitors                     # type: ignore
from matplotlib.lines import Line2D                     # type: ignore
from scipy.optimize import minimize                     # type: ignore

def ABC_PMC_postprocessing(theta, S, err, output, Par_info=None, Meas_info=None):
    """
    Postprocessing of the results of ABC_PMC.

    Parameters:
    ABCPar (dict): Algorithmic parameters (e.g., population size, number of generations).
    theta (ndarray): Posterior parameter samples (N, d, T).
    sigma_theta (ndarray): Covariance matrix of the proposal distribution.
    err (ndarray): Error tolerances.
    output (dict): Output from PMC containing various metrics.
    Par_info (dict, optional): Parameter bounds and information.
    Meas_info (dict, optional): Measurement information for fitting.

    Returns:
    None (plots and prints are generated within the function).
    """

    # Print wait statement to the screen
    print('ABC_PMC PLOTTING: PLEASE WAIT ...')
    
    # Define name of program
    n_program = 'ABC_PMC'
    
    # Define name of figures file
    file_name = f'{n_program}_figures.pdf'
    
    # Determine screen size (using matplotlib to get screen dimensions)
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    x_mult = screen_width / 1920
    y_mult = screen_height / 1080
    t_mult = min(x_mult, y_mult)

    # Define fontsize for figures
    fontsize_xylabel = 16 * t_mult
    fontsize_axis = 16 * t_mult
    fontsize_legend = 14 * t_mult
    fontsize_text = 14 * t_mult
    fontsize_title = 18 * t_mult
    markersize_symbols = 5
    fontsize_titlepage = 20 * t_mult
    markersize_legend = 10 * t_mult

    maxbins = 25
    # How many summary statistics?
    if np.ndim(Meas_info['S']) == 0:
        nS = 1
    else: 
        nS = len(Meas_info['S'])

    # Extract relevant parameters
    N, d, T = theta.shape
    fig_number = 1
    str_theta = [r"$\theta_{" + str(i+1) + "}$" for i in range(d)]
    str_S = [r"$S_{" + str(i+1) + "}$" for i in range(nS)]
    
    # Print some simple statistics
    TotFuncEval = output['tot_func_eval']  # Total function evaluations
    RunTime = output['RunTime']  # Total CPU time
    print(f"Total Function Evaluations: {TotFuncEval}")
    print(f"Runtime: {RunTime}")

    # Correlation matrix of posterior samples
    CORR = np.corrcoef(theta[:, :, -1].reshape(N, -1), rowvar=False)
    
    with PdfPages(file_name) as pdf:

        ### Plot Empty Figure for PDF
        plt.figure(figsize=(12, 6))
        plt.plot([], [], 'ro')  # Empty plot
        plt.axis([0, 1, 0, 1])
        plt.gca().set_facecolor('w')
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])        
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.text(0.3 * x_mult, 0.6 * y_mult, r'${\rm Visual \; results \; of \; ABC-PMC \; toolbox}$', fontsize=fontsize_titlepage) #, ha='center', va='center')
        plt.text(0.27 * x_mult, 0.5 * y_mult, r'$\;\;\;\;\;{\rm Tables \; are \; not \; printed \; to \; PDF \; file}$', fontsize=fontsize_titlepage) #, ha='center', va='center') #, fontweight='bold')
        ax = plt.gca()  # Get current axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        pdf.savefig()    
        plt.show()

        # Relationship between error (err) and posterior standard deviation
        STD = np.std(theta, axis=0).T  # Standard deviation for each parameter across generations
        plt.figure(fig_number)
        for i in range(d):
            plt.plot(err, STD[:, i] / (Par_info['max'][i] - Par_info['min'][i]), label=str_theta[i], linewidth=2)
        plt.xlabel(r'$\epsilon$ value', fontsize=fontsize_xylabel)
        plt.ylabel('Posterior standard deviation', fontsize=fontsize_xylabel)
        plt.legend(fontsize=fontsize_legend)
        plt.title('Error vs. Posterior Standard Deviation', fontsize=fontsize_title)
        pdf.savefig()
        plt.show()

        # Relationship between generation and acceptance rate
        plt.figure(fig_number)
        for i in range(d):
            plt.plot(np.arange(1,T+1), output['AR'], linewidth=2)
        plt.xlabel('Number of generations', fontsize=fontsize_xylabel)
        plt.ylabel('Acceptance rate', fontsize=fontsize_xylabel)
        # plt.legend()
        plt.title('Acceptance rate of candidate points in %', fontsize=fontsize_title)
        pdf.savefig()
        plt.show()

        ### Marginal distributions of posterior parameter (rank 1) solutions [of ultimate population]
        row, col = 2, 4
        idx_y_label = np.arange(0, d + 1 , col)
        # Calculate the number of bins for each parameter
        Nbins = [calcnbins(theta[:, i, -1]) for i in range(d)]
        nbins = min(np.min(Nbins), maxbins)
        nbins = max(5, round(nbins / 2))  # Adjust number of bins
        counter = 0
        while counter <= d - 1:
            # Create a row x col grid of subplots
            fig, axs = plt.subplots(row, col, figsize=(15, 10))
            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Increase the space horizontally and vertically
            # Loop over each subplot
            for ax in axs.flat:
                # Compute histogram for parameter j
                M, edges = np.histogram(theta[:, counter, -1], bins=nbins, density=True)
                X = 0.5 * (edges[:-1] + edges[1:])
                ax.bar(X, M / np.max(M), width=(edges[1] - edges[0]), color='gray', edgecolor='w', alpha=0.7, zorder=1)
                # Add xlabels = parameter name in Latex
                ax.set_xlabel(str_theta[counter], fontsize=fontsize_xylabel, labelpad = 10)
                # Adjust axis limits (tight)
                yticks = np.arange(0, 1.02, 0.2)
                ax.set_yticks(yticks)  # Set x-tick positions
                ax.set_yticklabels([str(round(tick,1)) for tick in yticks])
                ax.set_ylim(0, 1.02)                # Adjust y limits with padding
                # Add y-label
                if counter in idx_y_label:
                    ax.set_ylabel('Empirical density', fontsize=fontsize_xylabel, labelpad=10)
                # Plot the MEAN value
                ax.plot(np.mean(theta[:, counter, -1]), 0, f'kx', markersize=12, markeredgewidth=3, linewidth=3,zorder=2, clip_on=False)
                # Add letter
                label_plot = get_label(counter)  # This converts 0 -> 'A', 1 -> 'B', etc.
                ax.text(0.02,0.97, f'({label_plot})', transform=ax.transAxes, fontsize=fontsize_text, horizontalalignment='left', va='top', ha='left')
                counter += 1
                if counter == d:
                    # delete empty subplots
                    for ax in axs.flat:
                        if not ax.has_data():       # If the axis has no data, remove it
                            fig.delaxes(ax)
                    break
            # Set the title of the figure
            fig.suptitle(r"$\text{ABC-PMC: Marginal posterior parameter distributions}$", fontsize=fontsize_title, y=0.98)
            # Optionally adjust spacing to avoid overlap with subplots
            fig.tight_layout() #rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.show()
            # plt.close()

        ### Correlation Matrix Plot
        if d > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(CORR, cmap='coolwarm')
            plt.colorbar(cax)
            ax.set_xticks(np.arange(d))
            ax.set_yticks(np.arange(d))
            ax.set_xticklabels(str_theta, fontsize=fontsize_xylabel)
            ax.set_yticklabels(str_theta, fontsize=fontsize_xylabel)
            plt.title('ABC-PMC: Map of correlation coefficients of posterior parameter samples', fontsize=fontsize_title*3/4)
            pdf.savefig()        
            plt.show()
        elif d == 1:
            print("ABC-PMC WARNING: Cannot plot map with bivariate scatter plots as d = 1")
        else:
            print(f"ABC-PMC WARNING: Cannot plot map with posterior correlation estimates as d = {d} [too large]")

        ### Plot matrix plot of posterior parameter samples
        X1 = theta[:, :, -1]
        std_X1 = np.std(X1,axis=0)  # Compute the standard deviation of the Pareto solutions
        dx_move = 1/12              # How much do we extend x,y intervals beyond prior range (= 3%)
        # Correlation plots of the posterior parameter samples
        # X,Y axis of scatter plots correspond to prior ranges if defined
        if d > 1 and d <= 25:
            # Create a matrix plot for the marginal distributions and bivariate scatter plots
            fig, axs = plt.subplots(d, d, figsize=(15, 15))
            # Calculate the number of bins for each parameter
            Nbins = np.array([calcnbins(theta[:, i, -1]) for i in range(d)])
            nbins = min(np.min(Nbins), maxbins)
            nbins = max(5, round(nbins / 2))  # Adjust number of bins
            for i in range(d):
                for j in range(d):
                    if i != j:
                        axs[i, j].scatter(theta[:, j, -1], theta[:, i, -1], color='gray', s=12)
                        # Add a least-squares line for off-diagonal plots
                        # You can use numpy's polyfit for fitting a line if necessary
                        if std_X1[i] > 0 and std_X1[j] > 0:         # only if std exceeds zero
                            fit = np.polyfit(theta[:, j, -1], theta[:, i, -1], 1)
                            axs[i, j].plot(theta[:, j, -1], np.polyval(fit, theta[:, j, -1]), 'r--', linewidth=1)
                            # adjust the axes
                            axs[i, j].set_xlim([min(theta[:, j, -1]), max(theta[:, j, -1])])
                            axs[i, j].set_ylim([min(theta[:, i, -1]), max(theta[:, i, -1])])
                    if i == j:
                        # make a histogram
                        axs[i, j].hist(theta[:, i, -1], bins=nbins, density=True, alpha=0.6, color='gray', edgecolor='black')
                        # adjust the axes
                        axs[i, j].set_xlim([min(theta[:, i, -1]), max(theta[:, i, -1])])
                    # Set custom x-ticks and x-tick labels
                    x_min, x_max = axs[i, j].get_xlim()
                    dx = x_max - x_min
                    xticks = np.array([x_min + 1/12*dx, x_min + 6/12*dx, x_min + 11/12*dx])
                    axs[i, j].set_xticks(xticks)  # Set x-tick positions first, then labels, otherwise warning
                    axs[i, j].set_xticklabels([str(round(tick,2)) for tick in xticks])
                    y_min, y_max = axs[i, j].get_ylim()
                    dy = y_max - y_min
                    yticks = np.array([y_min + 1/12*dy, y_min + 6/12*dy, y_min + 11/12*dy])
                    axs[i, j].set_yticks(yticks)  # Set y-tick positions first, then labels, otherwise warning
                    axs[i, j].set_yticklabels([str(round(tick,2)) for tick in yticks])
                    # Add values and labels to the axes
                    if i == d - 1:
                        axs[i, j].set_xlabel(str_theta[j], fontsize=fontsize_xylabel)
                    else:
                        axs[i, j].set_xticklabels([])

                    # Add values and labels to the axes
                    if j == 0:
                        axs[i, j].set_ylabel(str_theta[i], fontsize=fontsize_xylabel)
                    else:
                        axs[i, j].set_yticklabels([])

            # Title of the figure
            fig.suptitle(f"ABC-PMC: Marginal distribution and bivariate scatter plots of posterior samples", fontsize=fontsize_title, y=0.92)
            pdf.savefig()
            plt.show()
        elif d == 1:
            fig, ax = plt.subplots(d, d, figsize=(15, 15))
            # Calculate the number of bins for each parameter
            Nbins = np.array([calcnbins(theta[:, 0, -1])])
            nbins = min(np.min(Nbins), maxbins)
            nbins = max(5, round(nbins / 2))  # Adjust number of bins
            # make a histogram
            ax.hist(theta[:, 0, -1], bins=nbins, density=True, alpha=0.6, color='gray', edgecolor='black')
            plt.xticks(fontsize=fontsize_axis, family='sans-serif', weight='normal', style='normal')
            plt.yticks(fontsize=fontsize_axis, family='sans-serif', weight='normal', style='normal')
            # Add values and labels to the axes
            ax.set_xlabel(str_theta[0], fontsize=fontsize_xylabel)
            # Title of the figure
            fig.suptitle(f"ABC-PMC: Marginal distribution of posterior samples", fontsize=fontsize_title, y=0.92)
            pdf.savefig()
            plt.show()
        else:
            print(f"\nABC-PMC WARNING: Cannot plot bivariate scatter plots as d = {d} (= too large)\n")

        ### Plot matrix plot of posterior parameter samples
        X1 = S[:, :, -1]
        std_X1 = np.std(X1,axis=0)  # Compute the standard deviation of the Pareto solutions
        dx_move = 1/12              # How much do we extend x,y intervals beyond prior range (= 3%)
        # Correlation plots of the posterior parameter samples
        # X,Y axis of scatter plots correspond to prior ranges if defined
        if nS > 1 and nS <= 25:
            # Create a matrix plot for the marginal distributions and bivariate scatter plots
            fig, axs = plt.subplots(nS, nS, figsize=(15, 15))
            # Calculate the number of bins for each parameter
            Nbins = np.array([calcnbins(S[:, i, -1]) for i in range(nS)])
            nbins = min(np.min(Nbins), maxbins)
            nbins = max(5, round(nbins / 2))  # Adjust number of bins
            for i in range(nS):
                for j in range(nS):
                    if i != j:
                        axs[i, j].scatter(S[:, j, -1], S[:, i, -1], color='gray', s=12)
                        # Add a least-squares line for off-diagonal plots
                        # You can use numpy's polyfit for fitting a line if necessary
                        if std_X1[i] > 0 and std_X1[j] > 0:         # only if std exceeds zero
                            fit = np.polyfit(S[:, j, -1], S[:, i, -1], 1)
                            axs[i, j].plot(S[:, j, -1], np.polyval(fit, S[:, j, -1]), 'r--', linewidth=1)
                            # adjust the axes
                            axs[i, j].set_xlim([min(S[:, j, -1]), max(S[:, j, -1])])
                            axs[i, j].set_ylim([min(S[:, i, -1]), max(S[:, i, -1])])
                            # add the measured values
                            axs[i, j].plot(Meas_info['S'][j], Meas_info['S'][i], f'kx', markersize=12, markeredgewidth=3, linewidth=3,zorder=2, clip_on=False)
                    if i == j:
                        # make a histogram
                        axs[i, j].hist(S[:, i, -1], bins=nbins, density=True, alpha=0.6, color='gray', edgecolor='black')
                        # add the measured values
                        axs[i, j].plot(Meas_info['S'][i], 0, f'kx', markersize=12, markeredgewidth=3, linewidth=3,zorder=2, clip_on=False)
                        # adjust the axes
                        axs[i, j].set_xlim([min(S[:, i, -1]), max(S[:, i, -1])])

                    # Set custom x-ticks and x-tick labels
                    x_min, x_max = axs[i, j].get_xlim()
                    dx = x_max - x_min
                    xticks = np.array([x_min + 1/12*dx, x_min + 6/12*dx, x_min + 11/12*dx])
                    axs[i, j].set_xticks(xticks)  # Set x-tick positions first, then labels, otherwise warning
                    axs[i, j].set_xticklabels([str(round(tick,2)) for tick in xticks])
                    y_min, y_max = axs[i, j].get_ylim()
                    dy = y_max - y_min
                    yticks = np.array([y_min + 1/12*dy, y_min + 6/12*dy, y_min + 11/12*dy])
                    axs[i, j].set_yticks(yticks)  # Set y-tick positions first, then labels, otherwise warning
                    axs[i, j].set_yticklabels([str(round(tick,2)) for tick in yticks])
                    # Add values and labels to the axes
                    if i == nS - 1:
                        axs[i, j].set_xlabel(str_S[j], fontsize=fontsize_xylabel)
                    else:
                        axs[i, j].set_xticklabels([])

                    # Add values and labels to the axes
                    if j == 0:
                        axs[i, j].set_ylabel(str_S[i], fontsize=fontsize_xylabel)
                    else:
                        axs[i, j].set_yticklabels([])

            # Title of the figure
            fig.suptitle(f"ABC-PMC: Marginal distribution and bivariate scatter plots of simulated summary statistics", fontsize=fontsize_title, y=0.92)
            pdf.savefig()
            plt.show()
        elif nS == 1:
            fig, ax = plt.subplots(d, d, figsize=(15, 15))
            # Calculate the number of bins for each parameter
            Nbins = np.array([calcnbins(S[:, 0, -1])])
            nbins = min(np.min(Nbins), maxbins)
            nbins = max(5, round(nbins / 2))  # Adjust number of bins
            # make a histogram
            ax.hist(S[:, 0, -1], bins=nbins, density=True, alpha=0.6, color='gray', edgecolor='black')
            # add the measured values
            ax.plot(Meas_info['S'], 0, f'kx', markersize=12, markeredgewidth=3, linewidth=3,zorder=2, clip_on=False)
            plt.xticks(fontsize=fontsize_axis, family='sans-serif', weight='normal', style='normal')
            plt.yticks(fontsize=fontsize_axis, family='sans-serif', weight='normal', style='normal')
            # Add values and labels to the axes
            ax.set_xlabel(str_S[0], fontsize=fontsize_xylabel)
            # Title of the figure
            fig.suptitle(f"ABC-PMC: Marginal distribution of posterior samples", fontsize=fontsize_title, y=0.92)
            pdf.savefig()
            plt.show()
        else:
            print(f"\nABC-PMC WARNING: Cannot plot bivariate scatter plots as number of summary metrics = {nS} (= too large)\n")

    # Open the final PDFs
    os.startfile(file_name)      

    return fig_number


def boundary_handling(x, Par_info):
    """
    Function to check whether parameter values remain within prior bounds.

    Parameters:
    x (ndarray): A 2D array of parameter values.
    Par_info (dict): A dictionary with 'min', 'max', and 'boundhandling' keys.
                     'min' and 'max' are the lower and upper bounds for each parameter.
                     'boundhandling' is the method used to handle boundary violations.
                     Options are 'reflect', 'bound', 'fold'.

    Returns:
    x (ndarray): The adjusted parameter values, respecting the bounds.
    """
    # Get the shape of x
    m, d = x.shape  # m is number of samples, d is number of parameters

    # Replicate min and max for broadcasting
    min_d = np.tile(Par_info['min'], (m, 1))
    max_d = np.tile(Par_info['max'], (m, 1))

    # Find which elements of x are smaller than their respective bound
    ii_low = np.where(x < min_d)

    # Find which elements of x are larger than their respective bound
    ii_up = np.where(x > max_d)

    if Par_info['boundhandling'] == 'reflect':  # Reflection
        # Reflect in min
        x[ii_low] = 2 * min_d[ii_low] - x[ii_low]

        # Reflect in max
        x[ii_up] = 2 * max_d[ii_up] - x[ii_up]

    elif Par_info['boundhandling'] == 'bound':  # Set to bound
        # Set lower values to min
        x[ii_low] = min_d[ii_low]

        # Set upper values to max
        x[ii_up] = max_d[ii_up]

    elif Par_info['boundhandling'] == 'fold':  # Folding
        # Fold parameter space lower values
        x[ii_low] = max_d[ii_low] - (min_d[ii_low] - x[ii_low])

        # Fold parameter space upper values
        x[ii_up] = min_d[ii_up] + (x[ii_up] - max_d[ii_up])

    else:
        print("ABC_PMC:boundary_handling: do not know this boundary handling option! - treat as unbounded parameter space")

    # Double-check if all elements are within bounds
    if Par_info['boundhandling'] in ['reflect', 'fold']:
        # Lower bound
        ii_low = np.where(x < min_d)
        x[ii_low] = min_d[ii_low] + np.random.rand(*ii_low[0].shape) * (max_d[ii_low] - min_d[ii_low])

        # Upper bound
        ii_up = np.where(x > max_d)
        x[ii_up] = min_d[ii_up] + np.random.rand(*ii_up[0].shape) * (max_d[ii_up] - min_d[ii_up])

    return x


def get_label(counter):
    label = ""
    while counter >= 0:
        label = chr(65 + (counter % 26)) + label
        counter = counter // 26 - 1
    return label


def calcnbins(x, method='middle', minb=1, maxb=np.inf):
    """
    Compute the "ideal" number of bins using different methods for histogram bin calculation.
    
    Parameters:
    - x: vector of data points
    - method: string with choice of method for calculating bins. Default is 'middle'.
        Options: 'fd', 'scott', 'sturges', 'middle', 'all'
    - minb: smallest acceptable number of bins (default: 1)
    - maxb: largest acceptable number of bins (default: np.inf)
    
    Returns:
    - nbins: The calculated number of bins based on the selected method.
    """
    
    # Input checking
    if not isinstance(x, (np.ndarray, list, np.generic)):
        raise ValueError('The x argument must be numeric or logical.')

    x = np.asarray(x)
    
    # Ensure the array is real, discard imaginary part
    if np.iscomplexobj(x):
        x = np.real(x)
        print('Warning: Imaginary parts of x will be ignored.')
    
    # Ensure x is a vector (1D array)
    if x.ndim != 1:
        x = x.flatten()
        print('Warning: x will be coerced to a vector.')
    
    # Remove NaN values
    x = x[~np.isnan(x)]
    if len(x) == 0:
        raise ValueError("x must contain at least one valid number.")
    
    # Choose method if not specified
    valid_methods = ['fd', 'scott', 'sturges', 'all', 'middle']
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")
    
    # Method selection
    if method == 'fd':
        nbins = calc_fd(x)
    elif method == 'scott':
        nbins = calc_scott(x)
    elif method == 'sturges':
        nbins = calc_sturges(x)
    elif method == 'middle':
        nbins = [calc_fd(x), calc_scott(x), calc_sturges(x)]
        nbins = np.median(nbins)
    elif method == 'all':
        nbins = {
            'fd': calc_fd(x),
            'scott': calc_scott(x),
            'sturges': calc_sturges(x)
        }
    
    # Confine number of bins to the acceptable range
    nbins = confine_to_range(nbins, minb, maxb)
    
    return nbins


def calc_fd(x):
    """Freedman-Diaconis rule"""
    h = np.subtract(*np.percentile(x, [75, 25]))  # Interquartile range (IQR)
    if h == 0:
        h = 2 * np.median(np.abs(x - np.median(x)))  # Median absolute deviation (MAD)
    
    if h > 0:
        nbins = np.ceil((np.max(x) - np.min(x)) / (2 * h * len(x) ** (-1/3)))
    else:
        nbins = 1
    return nbins


def calc_scott(x):
    """Scott's method"""
    h = 3.5 * np.std(x) * len(x) ** (-1/3)
    if h > 0:
        nbins = np.ceil((np.max(x) - np.min(x)) / h)
    else:
        nbins = 1
    return nbins


def calc_sturges(x):
    """Sturges' method"""
    nbins = np.ceil(np.log2(len(x)) + 1)
    return nbins


def confine_to_range(x, lower, upper):
    """Ensure bin count is within the specified range"""
    x = np.maximum(x, lower)
    x = np.minimum(x, upper)
    return np.floor(x)


def ordinal(i):
    # Handle special cases for 11, 12, and 13
    if 10 <= i % 100 <= 20:
        suffix = 'TH'
    else:
        # For all other cases, the suffix depends on the last digit
        suffix = {1: 'ST', 2: 'ND', 3: 'RD'}.get(i % 10, 'TH')
    
    # Return the number followed by its ordinal suffix in uppercase
    return str(i) + suffix


### Case studdy 1
def mixture(theta):
    """
    Univariate mixture distribution of Sisson et al. (2007)
    
    Parameters:
    theta (float): The parameter value
    
    Returns:
    float: A sufficient summary metric of the mixture distribution
    """
    # Draw 100 samples from N(0, 1) using theta as the mean
    y = np.random.normal(theta.flatten(), 1, 100)

    # Define distance function based on random choice
    if np.random.rand() < 0.5:
        rho = np.abs(np.mean(y))
    else:
        rho = np.abs(y[0])

    return rho


### Case study 2
def linear(theta):
    """
    Linear regression: Toy example from Vrugt and Sadegh (2013).
    
    Parameters:
    theta (array-like): A 1D array with two elements representing the parameters 
                         for the linear regression model (slope and intercept).
    
    Returns:
    S (list): A list containing the mean and standard deviation of the regression data.
    y (ndarray): A 100x1 vector of values of regression data with added noise.
    """
    theta = theta.flatten()
    # Persistent storage for x
    if not hasattr(linear, 'x'):
        linear.x = np.linspace(0, 10, 100)  # 100 equally spaced values in [0, 10]

    # Linear transformation
    y = theta[0] * linear.x + theta[1]  # Linear transformation using theta
    
    # Add random noise
    y = np.random.normal(y, 0.5)  # Add random error with a standard deviation of 0.5

    # Compute summary metrics (mean and std)
    S = np.array([np.mean(y), np.std(y)])
    
    return S, y


### Case study 3
def hmodel(theta, plugin):
    """
    hmodel simulation of discharge according to Schoups et al. 2010
    Returns Root Mean Square Error (RMSE) of driven and non-driven part hydrograph.
    
    Parameters:
        par (array): Parameter values for the model.
        plugin (dict): Dictionary containing plugin information, including time vector, model options, observed data, etc.
    
    Returns:
        F (array): RMSE of driven and non-driven part hydrograph.
        Y_sim (array): Simulated discharge after burn-in period.
    """
    
    # Extract various input data from the plugin dictionary
    tout = plugin['tout']
    data = plugin['data']
    hmodel_opt = plugin['hmodel_opt']
    y0 = plugin['y0']
    theta = theta.flatten()
    # Assign parameters from theta (model parameters)
    data['Imax'] = theta[0]     # interception storage capacity (mm)
    data['Sumax'] = theta[1]    # unsaturated zone storage capacity (mm)
    data['Qsmax'] = theta[2]    # maximum percolation rate (mm/d)
    data['aE'] = theta[3]       # evaporation coefficient
    data['aF'] = theta[4]       # runoff coefficient
    data['Kf'] = theta[5]       # fast-flow response time (d)
    data['Ks'] = theta[6]       # slow-flow response time (d)
#    data['aS'] = 1e-6           # percolation coefficient (constant)

    # Run the model (placeholder function)
    Y = crr_model(tout, y0, data, hmodel_opt)

    # Compute discharge from infinite reservoir (this is the flow output)
    y = np.diff(Y[4, :])        # Taking the difference to compute flow
    y = y[data['idx']]          # Burn-in
    
    # Calculate the summary metrics
    S_mod = calc_metrics(y, data['P'][data['idx']])

    return S_mod, Y


# Run crr model
def crr_model(tout, y0, data, options):
    nvar = len(y0)  # Number of state variables
    nt = len(tout)  # Number of time steps
    
    # Call the Runge-Kutta solver
    return runge_kutta(nvar, nt, tout, y0, data, options)


def runge_kutta(nvar, nt, tout, y0, data, options):
    hin = options['InitialStep']
    hmax_ = options['MaxStep']
    hmin_ = options['MinStep']
    reltol = options['RelTol']
    abstol = options['AbsTol']
    order = options['Order']
    
    LTE = np.zeros(nvar)
    ytmp = np.zeros(nvar)
    w = np.zeros(nvar)
    ns = nt - 1

    # Initialize y
    y = np.zeros((nvar, nt))
    y[:, 0] = y0
    
    for s in range(1, ns + 1):
        t1 = tout[s-1]
        t2 = tout[s]
        
        h = hin
        h = max(hmin_, min(h, hmax_))
        h = min(h, t2 - t1)
        
        y[:, s] = y[:, s - 1]  # Set initial y
        
        t = t1
        while t < t2:
            ytmp[:] = y[:, s]  # Copy current y
            
            # RK2 integration step
            rk2(data, nvar, s, t, h, ytmp, LTE)
            
            # Check if the step is acceptable
            accept = 0
            wrms = 0
            for i in range(nvar):
                w[i] = 1.0 / (reltol * abs(ytmp[i]) + abstol)
                wrms += (w[i] * LTE[i])**2
            wrms = np.sqrt(wrms / nvar)
            if wrms <= 1:
                accept = 1
            
            if accept > 0:
                y[:, s] = ytmp  # Update y
                t += h
            
            # Adjust step size
            h = h * max(0.2, min(5.0, 0.9 * wrms**(-1.0/order)))
            h = max(hmin_, min(h, hmax_))
            h = min(h, t2 - t)
    
    return y


def rk2(data, nvar, s, t, h, u, LTE):
    udotE = np.zeros(nvar)
    uE = np.zeros(nvar)
    udot = np.zeros(nvar)

    # Euler solution
    flag = fRhs(s, t, u, udotE, data)
    uE[:] = u + h * udotE
    # Heun solution
    flag = fRhs(s, t + h, uE, udot, data)
    u[:] = u + 0.5 * h * (udotE + udot)
    
    # Estimate LTE (Local Truncation Error)
    LTE[:] = np.abs(uE - u)


def fRhs(s, t, u, udot, data):
    P = data['P']
    Ep = data['Ep']
    Imax = data['Imax']
    Sumax = data['Sumax']
    Qsmax = data['Qsmax']
    aE = data['aE']
    aF = data['aF']
    aS = data['aS']
    Kf = data['Kf']
    Ks = data['Ks']

    Si = u[0]
    Su = u[1]
    Sf = u[2]
    Ss = u[3]

    Precip = P[s-1]
    
    if Imax > 0.0:
        EvapI = Ep[s-1] * expFlux(Si / Imax, 50.0)
        P_e = P[s-1] * expFlux(Si / Imax, -50.0)
        Ep_e = max(0.0, Ep[s-1] - EvapI)
    else:
        EvapI = 0.0
        P_e = P[s-1]
        Ep_e = Ep[s-1]

    Evap = Ep_e * expFlux(Su / Sumax, aE)
    Perc = Qsmax * expFlux(Su / Sumax, aS)
    Runoff = P_e * expFlux(Su / Sumax, aF)
    FastQ = Sf / Kf
    SlowQ = Ss / Ks

    udot[0] = Precip - EvapI - P_e
    udot[1] = P_e - Evap - Perc - Runoff
    udot[2] = Runoff - FastQ
    udot[3] = Perc - SlowQ
    udot[4] = FastQ + SlowQ

    return 0


def expFlux(Sr, a):
    Sr = max(0.0, min(1.0, Sr))
    if abs(a) < 1e-6:
        Qr = Sr  # Linear approximation
    else:
        Qr = (1.0 - exponen(-a * Sr)) / (1.0 - exponen(-a))
    return Qr


def exponen(x):
    return np.exp(min(300.0, x))


# Primary function to calculate summary metrics
def calc_metrics(y, P):
    # Determine the size of the data
    ny = len(y)

    # Sort flows & calculate exceedance probabilities using Weibull plotting position
    y_s, E_p = calc_FDC(y, ny, method=1)

    # Initial parameter values for Nelder-Mead optimization
    x0 = [1, 1]

    # Fit the FDC using van Genuchten WRF
    result = minimize(lambda x: VG_Ret(x, y_s, E_p, ny), x0, method='Nelder-Mead')
    S = np.zeros(4)
    S[2:4] = result.x  # Fitted parameters (air-entry and slope)

    # Derive the annual runoff coefficient
    S[0] = np.sum(y) / np.sum(P)

    # Initialize baseflow components
    yb = np.full(ny, np.nan)
    yb[0] = 0.25 * y[0]

    # Derive annual baseflow coefficient with low-pass filter
    phi = 0.925
    for j in range(1, ny):
        yb[j] = min(phi * yb[j - 1] + (1/2) * (1 - phi) * (y[j] + y[j - 1]), y[j])

    # Annual baseflow coefficient
    S[1] = np.sum(yb) / np.sum(y)

    return S


# Secondary function 1: Calculate exceedance probabilities
def calc_FDC(Y, nY, method):
    if method == 1:
        y_s = np.sort(Y)  # Sort discharge in ascending order
        E_p = 1 - ((np.arange(1, nY + 1) - 0.5) / nY)  # Weibull exceedance probabilities
    else:
        # Use empirical CDF to calculate exceedance probabilities
        F, y_s = np.histogram(Y, bins='auto', density=True)
        E_p = 1 - F.cumsum()  # Empirical exceedance probabilities
    return y_s, E_p


# Secondary function 2: Calculate RMSE for van Genuchten WRF
def VG_Ret(pars, y_s, E_ptrue, nY):
    alpha, n = pars  # Extract parameter values
    E_p = (1 + (alpha * y_s) ** n) ** (1 / n - 1)  # E_p according to VG WRF
    RMSE = np.sqrt(np.sum((E_p - E_ptrue) ** 2) / nY)  # Calculate RMSE
    return RMSE
