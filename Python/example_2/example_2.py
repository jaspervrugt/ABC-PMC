# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE      222222   #
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE          22 22    #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE         22     #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE           22      #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE      222222   #
#                                                                         #
# ####################################################################### #
#                                                                         #
# Example 1: Linear regression example from Vrugt and Sadegh (2013)       #
#  Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model             #
#      calibration and evaluation: Approximate Bayesian computation,      #
#      Water Resources Research, 49, pp. 4335–4345,                       #
#          https://doi.org/10.1002/wrcr.20354                             #
#                                                                         #
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

# Define the parameters for the ABC population sampler
ABCPar = {
    'N': 50,                          # Population size
    'method': 'RWM'                   # Proposal method (Random Walk Metropolis)
}

# Parameter information
Par_info = {
    'min': np.array([0, -10]),        # Minimum values of parameters
    'max': np.array([5, 10]),         # Maximum values of parameters
    'boundhandling': 'reflect'        # Boundary handling method
}

# Function name (model)
Func_name = 'linear'

# Generate synthetic data
y = 0.5 * np.linspace(0, 10, 100) + 5  # Linear model: 0.5 * x + 5
y = y + np.random.normal(0, 0.5, 100)  # Add random error (normal noise)

# Define observed summary metrics
Meas_info = {
    'S': np.array([np.mean(y), np.std(y)])  # Mean and standard deviation
}

# Define error tolerance
err = np.array([1.00, 0.75, 0.50, 0.25, 0.10, 0.05, 0.025])

if __name__ == '__main__':
	# Run ABC-PMC toolbox
	theta, S, sigma_theta, output = ABC_PMC(ABCPar, Func_name, err, Par_info, Meas_info)
