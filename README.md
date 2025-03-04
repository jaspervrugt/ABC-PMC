# ABC-PMC: Population Monte Carlo Sampler of Hydrograph Signatures in MATLAB and Python

## Description

Approximate Bayesian Computation avoids the use of an explicit likelihood function in favor a (number of) summary statistics that measure the distance between the model simulation and the data. This ABC approach is a vehicle for diagnostic model calibration and evaluation for the purpose of learning and model correction. The ABC-PMC toolbox in MATLAB and Python implements the Population Monte Carlo sampler to approximate the posterior distribution of the summary metrics using the distance function $\rho(\cdot)$ and sequence of successively smaller epsilon values, $\epsilon_{1},\ldots,epsilon_{J}$. Samples are accepted if $\rho(\cdot) \leq \epsilon$. The PMC sampler starts out as ABC-REJ during the first iteration, $j = 1$, but using a much larger initial value for $\epsilon$, the acceptance threshold. During each successive next step (iteration), $j = (2,\ldots,J)$, the value of $\epsilon$ is decreased and the proposal distribution, $q_{j}(\theta_{k}^{j−1},\cdot) = N_{d}(\theta_{k}^{j−1},\Sigma^{j})$ adapted using $\Sigma^{j} = \text{Cov}(\theta_{1}^{j−1},\ldots,\theta_{N}^{j−1})$ with $\theta_{k}$ drawn from a multinomial distribution, $F(\theta_{1:N}^{j−1} \vert w_{1:N}^{j-1})$ where $w_{1:N}^{j-1}$ are the posterior weights $(w_{l}^{j-1} \geq 0; \sum_{l=1}^{N} w_{l}^{j-1} = 1)$. Through a sequence of successive (multi)normal proposal distributions, the prior sample is thus iteratively refined until a sample of the posterior distribution is obtained. This approach, similar in spirit to the adaptive Metropolis sampler of Haario et al. (1999, 2001), receives a much higher sampling efficiency than ABC-REJ, particularly for cases where the prior sampling distribution $p(\theta)$ is a poor approximation of the posterior distribution. Details of the ABC-PMC procedure are given in Appendix B of Sadegh and Vrugt (2013). Another significant improvement in efficiency is obtained from MCMC simulation using the DREAM$_{(ABC)}$ algorithm of Sadegh and Vrugt (2014). This code is part of DREAM-Suite.

## Getting Started

### Installing: MATLAB

* Download and unzip the zip file 'MATLAB_code_ABC_PMC_V1.0.zip' in a directory 'ABC_PMC'
* Add the toolbox to your MATLAB search path by running the script 'install_ABC_PMC.m' available in the root directory
* You are ready to run the built-in examples

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
1. Turner, B.M, and T. van Zandt (2012), A tutorial on approximate Bayesian computation, _Journal of Mathematical Psychology_, 56, pp. 69-85
2. Sadegh, M., and J.A. Vrugt (2014), Approximate Bayesian computation using Markov chain Monte Carlo simulation: DREAM_(ABC), _Water Resources Research_, https://doi.org/10.1002/2014WR015386
3. Vrugt, J.A., and M. Sadegh (2013), Toward diagnostic model calibration and evaluation: Approximate Bayesian computation, _Water Resources Research_, 49, pp. 4335–4345, https://doi.org/10.1002/wrcr.20354
4. Sadegh, M., and J.A. Vrugt (2013), Bridging the gap between GLUE and formal statistical approaches: approximate Bayesian computation, _Hydrology and Earth System Sciences_, 17, pp. 4831–4850
5. Sisson, S.A., Y. Fan, and M.M. Tanaka (2007), Sequential Monte Carlo without likelihoods, _Proceedings of the National Academy of Sciences of the United States of America_, 104(6), pp. 1760-1765, https://doi.org/10.1073/pnas.0607208104

## Version History

* 1.0
    * Initial Release
    * Built-in Case Studies
    * Basic Postprocessing
    * Python Implementation

## Built-in Case Studies

1. Example 1: Toy example from Sisson et al. (2007)
2. Example 2: Linear regression example from Vrugt and Sadegh (2013)
3. Example 3: Hydrologic modeling using hydrograph functionals

## Acknowledgments
