# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE      333333   #
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE              33   #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE          333   #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              33   #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE      333333   #
#                                                                         #
# ####################################################################### #
#                                                                         #
# Example 3: Hydrologic modeling with hydrograph functions                #
#  Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model             #
#      calibration and evaluation: Approximate Bayesian computation,      #
#      Water Resources Research, 49, pp. 4335–4345,                       #
#          https://doi.org/10.1002/wrcr.20354                             #
#                                                                         #
# Please check example 14 of DREAM_Package                                #
# = more efficient/better implementation                                  #
# ####################################################################### #

import sys
import os

# Get the current working directory
current_directory = os.getcwd()
# Go up one directory
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
# add this to path
sys.path.append(parent_directory)
# Add another directory
misc_directory = os.path.abspath(os.path.join(parent_directory, 'miscellaneous'))
# add this to path
sys.path.append(misc_directory)

import numpy as np
from ABC_PMC import ABC_PMC
from ABC_PMC_functions import *

# ABC Parameter settings
ABCPar = {
    'N': 50, 		# Population size
    'method': 'RWM' 	# Which method to create proposals
}

# Parameter information (min, max) for each parameter
Par_info = {
    'min': [0.2, 25,   0, 1e-6, -10, 0.001, 0.1],  	# min values for parameters [Imax, Smax, Qsmax, alE, alF, Kfast, Kslow]
    'max': [10,  1000, 100, 100, 10, 10, 150]  	# max values for parameters
}

# Boundary handling method
Par_info['boundhandling'] = 'reflect'

# Model name
Func_name = 'hmodel'

# Load daily discharge data (assuming it's in a .dly file format)
daily_data = np.loadtxt('03451500.dly')

# Skipping the first two years (i.e., index 730 onward in Python)
data_idx = np.arange(730, daily_data.shape[0])
n = data_idx.shape[0]

# Measured summary metrics for observed data
Meas_info = {}
Meas_info['S'] = calc_metrics(daily_data[data_idx, 5], daily_data[data_idx, 3])

# Data variables
data = {
    'P': daily_data[:, 3],  		# Daily rainfall (mm/d)
    'Ep': daily_data[:, 4],  		# Daily evaporation (mm/d)
    'aS': 1e-6,  			# Percolation coefficient
    'idx': data_idx			# Indices to use	
}

# Define training data set
tout = np.arange(0, data['P'].shape[0]+1)

# hmodel options
hmodel_opt = {
    'InitialStep': 1,  # Initial time-step (d)
    'MaxStep': 1,      # Maximum time-step (d)
    'MinStep': 1e-6,   # Minimum time-step (d)
    'RelTol': 1e-3,    # Relative tolerance
    'AbsTol': 1e-3,    # Absolute tolerances (mm)
    'Order': 2         # 2nd order method (Heun)
}

y0 = 1e-6 * np.ones(5)  # Initial conditions

# Plugin structure (used to pass to the AMALGAM algorithm)
plugin = {
    'tout': tout,
    'data': data,
    'hmodel_opt': hmodel_opt,
    'y0': y0,
    'n': n,
}

# Error tolerance (used in the ABC algorithm)
err = [1.00, 0.75, 0.50, 0.25, 0.15, 0.10, 0.075, 0.05, 0.04, 0.03, 0.025]

if __name__ == '__main__':
	# Run the ABC Population Monte Carlo Sampler: parallel
	theta, S, sigma_theta, output = ABC_PMC_parallel(ABCPar, Func_name, err, Par_info, Meas_info, plugin)
