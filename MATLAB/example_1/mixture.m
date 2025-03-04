function rho = mixture(theta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Univariate mixture distribution of Sisson et al. (2007)                 %
%                                                                         %
% SYNOPSIS: rho = mixture(theta)                                          %
% where                                                                   %
%  theta       [input] parameter value                                    %
%  rho         [outpt] sufficient summary metric of mixture distribution  %
%                                                                         %
% Check the following paper                                               %
%  Sisson, S.A., Y. Fan, and M.M. Tanaka (2007), Sequential Monte Carlo   %
%      without likelihoods, Proceedings of the National Academy of        %
%      Sciences of the United States of America, 104(6), pp. 1760 - 1765, %
%          https://doi.org/10.1073/pnas.0607208104                        %
%                                                                         %
% (c) Written by Jasper A. Vrugt, Oct. 2009                               %
% Los Alamos National Laboratory                                          %
% University of California Irvine                                         %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

y = normrnd(theta,1,100,1);     % Draw 100 samples from N(0,1)
if rand < 1/2                   % Define distance function
    rho = abs(mean(y));
else
    rho = abs(y(1));
end

end
