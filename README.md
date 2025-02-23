# ABC-PMC: Population Monte Carlo Sampler of Hydrograph Signatures in MATLAB and Python

## Description

Approximate Bayesian Computation avoids the use of an explicit likelihood function in favor a (number of) summary statistics that measure the distance between the model simulation and the data. This ABC approach is a vehicle for diagnostic model calibration and evaluation for the purpose of learning and model correction. This toolbox in MATLAB and Python implements the Population Monte Carlo sampler to approximate the posterior distribution of the summary metrics used. In short, the PMC sampler starts out as ABC-REJ during the first iteration, j = 1, but using a much larger initial value for ε, the acceptance threshold. During each successive next step (iteration), j = (2,...,J), the value of ε is decreased and the proposal distribution, q_{j}(**θ**^{j−1}_{k},·) = N_{d}(θ^{j−1}_{k},Σ^{j}) adapted using Σ^{j} = Cov(**θ**^{j−1}_{1},...,**θ**^{j−1}_{N}) with **θ**_{k} drawn from a multinomial distribution, F(**θ**^{j−1}_{1:N}|**w**^{j-1}_{1:N}) where **w**^{j-1}_{1:N} are the posterior weights (w^{j-1}_{l} ≥ 0; Σ_{l=1}^{N} w^{j-1}_{l} = 1). Through a sequence of successive (multi)normal proposal distributions, the prior sample is thus iteratively refined until a sample of the posterior distribution is obtained. This approach, similar in spirit to the adaptive Metropolis sampler of Haario et al. (1999, 2001), receives a much higher sampling
efficiency than ABC-REJ, particularly for cases where the prior sampling distribution p(**θ**) is a poor approximation of the actual posterior distribution.

The PMC sampler of Turner and van Zandt (2012) assumes that the sequence of ε-values is specified by the user. This does not necessarily lead to the most efficient search. The ABC-PMC sampler therefore adaptively determines the next value of ε_{j}; j > 1 from the cumulative distribution function of the ρ(·) values of the N most recent accepted samples. Details of this procedure are given in Appendix B of Sadegh and Vrugt (2014). The PMC algorithm is not particularly efficient and hence we have alternative implementations that adaptively selects the sequence of epsilon values. I recommend using the DREAM_(ABC) algorithm developed by Sadegh and Vrugt (2014). This code is orders of magnitude more efficient than the ABC-PMC method and part of eDREAM_package   

## Getting Started

### Installing: MATLAB

* Download and unzip the zip file 'MATLAB_code_ABC_PMC_V1.0.zip' in a directory 'ABC_PMC'
* Add the toolbox to your MATLAB search path by running the script 'install_ABC_PMC.m' available in the root directory
* You are ready to run the examples

### Executing program

* After intalling, you can simply direct to each example folder and execute the local 'example_X.m' file
* Please make sure you read carefully the instructions (i.e., green comments) in 'install_ABC_PMC.m'  

### Installing: Python

* Download and unzip the zip file 'Python_code_ABC_PMC_V1.0.zip' to a directory called 'ABC_PMC'

### Executing program

* Go to Command Prompt and directory of example_X in the root of ABC_PMC
* Now you can execute this example by typing "python example_X.py".
* Instructions can be found in the file 'ABC_PMC.py' 
  
## Authors

* Vrugt, Jasper A. (jasper@uci.edu) 

## Literature
Turner, B.M, and T. van Zandt (2012), A tutorial on approximate Bayesian computation, _Journal of       Mathematical Psychology_, 56, pp. 69-85
Sadegh, M., and J.A. Vrugt (2014), Approximate Bayesian computation using Markov chain Monte Carlo     simulation: DREAM_(ABC), _Water Resources Research_, https://doi.org/10.1002/2014WR015386        Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model calibration and evaluation: Approximate Bayesian computation, _Water Resources Research_, 49, pp. 4335–4345,       
   https://doi.org/10.1002/wrcr.20354
Sadegh, M., and J.A. Vrugt (2013), Bridging the gap between GLUE and formal statistical approaches:   approximate Bayesian computation, _Hydrology and Earth System Sciences_, 17, pp. 4831–4850

## Version History

* 1.0
    * Initial Release
    * Built-in Case Studies
    * Basic Postprocessing
    * Python Implementation

## Acknowledgments


# ABC-PMC
Approximation Bayesian Computation: Population Monte Carlo
