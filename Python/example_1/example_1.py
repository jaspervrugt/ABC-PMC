# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE        1111   #
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE           11 11   #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE       11  11   #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              11   #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE          11   #
#                                                                         #
# ####################################################################### #
#                                                                         #
# Example 1: Mixture toy example from Sisson et al. (2007)                #
#  Sisson, S.A., Y. Fan, and M.M. Tanaka (2007), Sequential Monte Carlo   #
#      without likelihoods, Proceedings of the National Academy of        #
#      Sciences of the United States of America, 104(6), pp. 1760 - 1765, #
#          https://doi.org/10.1073/pnas.0607208104                        #
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

# Set parameters for ABC Population Monte Carlo (ABC_PMC)
ABCPar = {
    'N': 50,                         # Population size
    'method': 'RWM'                  # Method to create proposals
}

# Parameter bounds and boundary handling
Par_info = {
    'min': -10,                      # Minimum value of parameter
    'max': 10,                       # Maximum value of parameter
    'boundhandling': 'reflect'       # Boundary handling method
}

# Measurement information
Meas_info = {
    'S': 0                           # Define observed summary metrics
}

Func_name = 'mixture'                # Model name

# Define the error tolerance
err = np.array([1.00, 0.75, 0.50, 0.25, 0.10, 0.05, 0.025])

if __name__ == '__main__':
	# Run ABC-PMC toolbox
	theta, S, sigma_theta, output = ABC_PMC(ABCPar, Func_name, err, Par_info, Meas_info)

