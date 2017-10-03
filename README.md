Distributions
===================

# Overview
This repository contains python implementations of hyperbolic and variance-gamma distributions.<P>

# Definitions
The __hyperbolic__ density function parametrized with 
<img src="http://latex.codecogs.com/gif.latex?\left&space;(\pi,&space;\zeta,&space;\delta&space;,\mu&space;\right&space;)" title="\left (\pi, \zeta, \delta ,\mu \right )" />
is defined as:<P>
<img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{2\sqrt{1&plus;\pi^{2}}K_{1}\left&space;(&space;\zeta&space;\right&space;)}e^{-\zeta&space;\left&space;[&space;\sqrt{1&plus;\pi^2}&space;\sqrt{1&plus;\left&space;(&space;\frac{x-\mu}{\delta&space;}&space;\right&space;)^2}-\pi\frac{x-\mu}{\delta&space;}&space;\right&space;]}" title="\frac{1}{2\sqrt{1+\pi^{2}}K_{1}\left ( \zeta \right )}e^{-\zeta \left [ \sqrt{1+\pi^2} \sqrt{1+\left ( \frac{x-\mu}{\delta } \right )^2}-\pi\frac{x-\mu}{\delta } \right ]}" /><P>
where 
<img src="http://latex.codecogs.com/gif.latex?K_{1}()" title="K_{1}()" /> 
is the modified Bessel function of the third kind with order 1.<P><P>
  
The __variance-gamma__ density function parametrized with
<img src="http://latex.codecogs.com/gif.latex?(c,\sigma,\theta,\nu)" title="(c,\sigma,\theta,\nu)" />
is defined as:<P>
<img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;c\left&space;(&space;c,&space;\sigma,&space;\theta,&space;\nu&space;\right&space;)e^{\theta\f{(x-c)/&space;\sigma^2}}\left&space;|&space;x-c&space;\right&space;|^{1/\nu-1/2}K_{1/\nu-1/2}\left&space;(&space;\left&space;|&space;x-c&space;\right&space;|&space;\sqrt{2\sigma^2/\nu&plus;\theta^2}/\sigma^2\right&space;)" title="c\left ( c, \sigma, \theta, \nu \right )e^{\theta\f{(x-c)/ \sigma^2}}\left | x-c \right |^{1/\nu-1/2}K_{1/\nu-1/2}\left ( \left | x-c \right | \sqrt{2\sigma^2/\nu+\theta^2}/\sigma^2\right )" /> <P>
where 
<img src="http://latex.codecogs.com/gif.latex?K_\nu()" title="K_\nu()" /> 
is the modified Bessel function of the third kind with order 
<img src="http://latex.codecogs.com/gif.latex?\nu" title="\nu" />
and <P>
<img src="http://latex.codecogs.com/gif.latex?c(c,\sigma,\theta,\nu)&space;=&space;\frac{2}{\sigma\sqrt{2\pi}\nu^{1/\nu}\Gamma(1/\nu)}\left&space;(&space;\frac{1}{\sqrt{2\sigma^2/\nu&plus;\theta^2}}&space;\right&space;)^{1/\nu-1/2}" title="c(c,\sigma,\theta,\nu) = \frac{2}{\sigma\sqrt{2\pi}\nu^{1/\nu}\Gamma(1/\nu)}\left ( \frac{1}{\sqrt{2\sigma^2/\nu+\theta^2}} \right )^{1/\nu-1/2}" />
  
# Methods
The following methods are available:
* __pdf__ - Probability density function at x
* __pdf_deriv__ - Probability density function derivative at x
* __cdf__ - Cumulative distribution function at x
* __ppf__ - Percent point function (inverse of `cdf`) at q
* __mode__ - Mode of distribution
* __stats__ - Some statistics of distribution
* __fit__ - Return MLEs for parameters from data.
            MLE stands for Maximum Likelihood Estimate. Starting estimates for
            the fit are given by `_fitstart`, which uses method of moments.
* __freeze__ - Freeze the distribution for the given arguments

# Other
The distribution can be called in two ways:
* self - standard distribution instance, the parameters needs to be specified
* self(parameters) - frozen distribution instance, the parameters are set at the beggining. The methods `fit` and `freeze` are unavailable for frozen distributions
 





