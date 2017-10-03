import math
import numpy as np
from numpy import (inf, nan, isinf)
from scipy import (special, interpolate, optimize)
from scipy.integrate import quad, trapz
from scipy.stats.mstats import rankdata
from scipy.stats import (skew, kurtosis)

# Frozen RV class
class rv_frozen(object):

    def __init__(self, dist, hpi, zeta, delta = 1.0, mu = 0.0):
        self.args = hpi, zeta, delta, mu
        
        # create a new instance
        self.dist = dist.__class__()

    def pdf(self, x):
        return self.dist.pdf(x, *self.args)

    def pdf_deriv(self, x):
        return self.dist.pdf_deriv(x, *self.args)
    
    def cdf(self, x):
        return self.dist.cdf(x, *self.args)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args)

    def mode(self):
        return self.dist.mode(*self.args)
    
    def stats(self, moments='mv'):
        return self.dist.stats(*self.args, moments)
    
class hyperb_gen(object):
    
    # The largest [in magnitude] usable floating value.
    _XMAX = np.finfo(float).max
 
    def freeze(self, hpi, zeta, delta = 1.0, mu = 0.0):
        """Freeze the distribution for the given arguments.
        Parameters
        ----------
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.
        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        return rv_frozen(self, hpi, zeta, delta, mu)
    
    def __call__(self, hpi, zeta, delta = 1.0, mu = 0.0):
        return self.freeze(hpi, zeta, delta, mu)
    
    def pdf(self, x, hpi, zeta, delta = 1.0, mu = 0.0):
        """
        Probability density function at x of the Hyperbolic distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive")
            
        kv = special.kv(1,zeta)
        
        return (2.0 * delta * (1.0 + hpi**2)**0.5 * 
             kv)**(-1) * np.exp(-zeta * ((1.0 + hpi**2)**0.5 * 
            (1 + ((x - mu)/delta)**2)**0.5 - hpi * 
            (x - mu)/delta))
    
    def pdf_deriv(self, x, hpi, zeta, delta = 1.0, mu = 0.0):
        """
        Probability density function derivative at x of the Hyperbolic distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        pdf_deriv : ndarray
            Probability density function derivative evaluated at x
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive")
            
        return (self.pdf(x, hpi, zeta, delta, mu) * (zeta * hpi/delta - 
              zeta * (1.0 + hpi**2)**0.5 * (x - mu)/((1 + ((x - 
              mu)/delta)**2)**0.5 * delta**2)))
        
    def cdf(self, x, hpi, zeta, delta = 1.0, mu = 0.0):
        """
        Cumulative distribution function at x of the Hyperbolic distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive") 
            
        # Make sure x is numpy array and sort the values
        x = np.atleast_1d(x)
        x = (x - mu)/delta
        xs = np.sort(x)
        delta, mu = 1.0, 0.0
        kv = special.kv(1,zeta)
        phi = zeta/delta * ((1 + hpi**2)**0.5 + hpi)
        gamma = zeta/delta * ((1 + hpi**2)**0.5 - hpi)
        const = 1.0/(2.0 * (1.0 + hpi**2)**0.5 * kv)
        
        # Get the breaks and add -inf and inf 
        xi = self._breaks(hpi, zeta, delta, mu)
        xi = np.concatenate([[-inf],xi,[inf]])
    
        # Evaluate cdf at break points
        int_xi = np.zeros(1)
        int_xi = np.append(int_xi, (1/phi) * const * math.exp(phi * xi[2]))
        for j in range(2,xi.size-4):
            int_xi = np.append(int_xi, quad(self.pdf, xi[j], xi[j+1], args=(hpi, zeta, delta, mu))[0]+int_xi[j-1])
        int_xi = np.append(int_xi, (1/gamma) * const * math.exp(-gamma * xi[xi.size-3])) 
        
        # Create masks that contains index of a range
        mx = np.full(x.shape, xi.size-2, dtype=np.int32)
        for j in range(xi.size-1):
            mx[(xs >= xi[j]) & (xs < xi[j+1])] = j
        
        # Integrate
        interval = np.sort(np.stack((xi[mx], xs), axis=-1))
        xint = np.full(x.shape+(101,),nan)
        for index in np.ndindex(xs.shape):
            if not isinf(interval[index][0]) and not isinf(interval[index][1]):
                xint[index] = np.linspace(interval[index][0],interval[index][1],xint.shape[-1])
        yint = self.pdf(xint, hpi, zeta, delta, mu)
        resint = trapz(yint,xint)
        
        # Accumulate
        resint[mx==0] = 0
        resint[mx==1] = (1/phi) * const * np.exp(phi * xs[mx==1])
        for j in range(2,xi.size-3):
            resint[mx==j] = int_xi[j-1] + resint[mx==j]
        resint[mx==(xi.size-3)] = 1 - (1/gamma) * const * np.exp(-gamma * xs[mx==(xi.size-3)])
        resint[mx==(xi.size-2)] = 1
        
        return resint.flatten()[rankdata(x).astype(int)-1].reshape(resint.shape)
    
    def _breaks(self, hpi, zeta, delta = 1.0, mu = 0.0):
        """
        Breaks function at x of the Hyperbolic distribution.
        
        Parameters
        ----------
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        breaks : tuple
            Coordinates of breaking points dividing distribution into 8 regions
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive") 
        deriv=0.3
        epsTiny = 10**(-10)
        epsSmall = 10**(-6)
      
        delta_orig, mu_orig = delta, mu
        delta, mu = 1.0, 0.0
        kv = special.kv(1,zeta)
        phi = zeta/delta * ((1 + hpi**2)**0.5 + hpi)
        gamma = zeta/delta * ((1 + hpi**2)**0.5 - hpi)
        const = 1.0/(2.0 * (1.0 + hpi**2)**0.5 * kv)
      
        xMode = self.mode(hpi, zeta, delta, mu)
        xTiny = 1/phi * math.log(epsTiny * phi/const)
        xSmall = 1/phi * math.log(epsSmall * phi/const)
        xLarge = -1/gamma * math.log(epsSmall * gamma/const)
        xHuge = -1/gamma * math.log(epsTiny * gamma/const)
        
        xDeriv = np.linspace(xSmall, xMode, 101)
        derivVals = self.pdf_deriv(xDeriv, hpi, zeta, delta, mu)
        maxDeriv = np.max(derivVals)
        breakSize = deriv * maxDeriv
        breakFun = lambda x: self.pdf_deriv(x, hpi, zeta, delta, mu) - breakSize
        if (maxDeriv < breakSize) or (derivVals[0] > breakSize):
            xBreakLow = xSmall
        else:
            whichMaxDeriv = derivVals.argmax(0)
            xBreakLow = optimize.brentq(breakFun, a=xSmall, b=xDeriv[whichMaxDeriv])
    
        xDeriv = np.linspace(xMode, xLarge, 101)
        derivVals = -self.pdf_deriv(xDeriv, hpi, zeta, delta, mu)
        maxDeriv = np.max(derivVals)
        breakSize = deriv * maxDeriv
        breakFun = lambda x: -self.pdf_deriv(x, hpi, zeta, delta, mu) - breakSize
        if (maxDeriv < breakSize) or (derivVals[-1] > breakSize):
            xBreakHigh = xLarge
        else:
            whichMaxDeriv = derivVals.argmax(0)
            xBreakHigh = optimize.brentq(breakFun, a=xDeriv[whichMaxDeriv], b=xLarge)
    
        rng = delta_orig * np.array([xTiny, xSmall, xBreakLow, xMode, xBreakHigh, xLarge, xHuge]) + mu_orig
        
        return tuple(rng) 
    
    def mode(self, hpi, zeta, delta = 1.0, mu = 0.0):
        """
        Mode for the Hyperbolic distribution.
        
        Parameters
        ----------
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        mode : float
            the mode of distribution
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive") 
        return mu + delta * hpi
    
    def ppf(self, q, hpi, zeta, delta = 1.0, mu = 0.0): 
        """
        Percent point function (inverse of `cdf`) at q of the Hyperbolic distribution.
        
        Parameters
        ----------
        q : array_like
            lower tail probability
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        ppf : ndarray
            quantile corresponding to the lower tail probability q.
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive") 
    
        eps = 0.00001
        
        # Make sure q is numpy array and sort the values
        q = np.atleast_1d(q)
        qs = np.sort(q)
        
        delta_orig, mu_orig = delta, mu
        delta, mu = 1.0, 0.0
        kv = special.kv(1,zeta)
        phi = zeta/delta * ((1 + hpi**2)**0.5 + hpi)
        gamma = zeta/delta * ((1 + hpi**2)**0.5 - hpi)
        const = 1.0/(2.0 * (1.0 + hpi**2)**0.5 * kv)
       
        xi = self._breaks(hpi, zeta, delta, mu)
        xi = np.concatenate([[-inf],xi,[inf]])
        
        # Create masks that contains index of a range
        mx = np.full(q.shape, xi.size-2, dtype=np.int32)
        for j in range(xi.size-1):
            a = self.cdf(xi[j], hpi, zeta, delta, mu)
            b = self.cdf(xi[j+1], hpi, zeta, delta, mu)
            mx[(qs >= a) & (qs < b)] = j
    
        interval = np.sort(np.stack((xi[mx], xi[mx+1]), axis=-1))
        
        knots = [[]]
        for j in range(1,xi.size-2):
            if np.any(mx==j):
                xVal = np.linspace(xi[j]-eps, xi[j+1]+eps, 100)
                cdfVal = self.cdf(xVal, hpi, zeta, delta, mu)
                knots.append(interpolate.splrep(xVal, cdfVal, s=0))
            else:
                knots.append([])
                
        qqs = np.full(qs.shape,nan)           
        for index in np.ndindex(qs.shape):
            if mx[index] == 0:
                qqs[index] = -inf
            elif mx[index] == 1:
                qqs[index] = (1/phi) * np.log(phi * qs[index]/const)
            elif mx[index] == (xi.size-3):
                qqs[index] = -(1/gamma) * np.log(gamma * (1 - qs[index])/const)            
            elif mx[index] == (xi.size-2):
                qqs[index] = inf       
            else:
                tknots = knots[mx[index]]
                tqs = qs[index]
                zeroFun = lambda x: interpolate.splev(x, tknots, der=0) - tqs
                a = interval[index][0]
                b = interval[index][1]
                if zeroFun(a) >= 0:
                    qqs[index] = a
                elif zeroFun(b) <= 0:
                    qqs[index] = b
                else:
                    qqs[index] = optimize.brentq(zeroFun, a=a, b=b) 
        
        return qqs.flatten()[rankdata(q).astype(int)-1].reshape(qqs.shape) * delta_orig + mu_orig
    
    def _rlambda(self, zeta, lam = 1):
       return  special.kv(lam+1, zeta) / special.kv(lam, zeta)
    
    def _slambda(self, zeta, lam = 1):
        return (special.kv(lam+2, zeta) * special.kv(lam, zeta) - special.kv(lam+1, zeta)**2) / special.kv(lam, zeta)**2
    
    def _wlambda1(self, zeta, lam = 1):
        return self._rlambda(zeta, lam)
    
    def _wlambda2(self, zeta, lam = 1):
        return -self._rlambda(zeta, lam)**2 + 2.0 * (lam + 1.0) * self._rlambda(zeta, lam)/zeta + 1.0
    
    def _wlambda3(self, zeta, lam = 1):
        return (2.0 * self._rlambda(zeta, lam)**3 - 6.0 * (lam + 1.0) * self._rlambda(zeta, 
                lam)**2/zeta + (4.0 * (lam + 1.0) * (lam + 2.0)/zeta**2 - 
                2.0) * self._rlambda(zeta, lam) + 2.0 * (lam + 2.0)/zeta)
    
    def _wlambda4(self, zeta, lam = 1):
        return (-6.0 * self._rlambda(zeta, lam)**4 + 24.0 * (lam + 1.0) * self._rlambda(zeta, 
        lam)**3/zeta + (-4.0 * (lam + 1.0) * (7.0 * lam + 
        11.0)/zeta**2 + 8.0) * self._rlambda(zeta, lam)**2 + (8.0 * (lam + 
        1.0) * (lam + 2.0) * (lam + 3.0)/zeta**3 - 4.0 * (4.0 * lam + 
        5.0)/zeta) * self._rlambda(zeta, lam) + 4.0 * (lam + 2.0) * 
        (lam + 3.0)/zeta**2 - 2.0)
    
    def stats(self, hpi, zeta, delta = 1.0, mu = 0.0, moments='mv'):
        """
        Some statistics of the VarianceGamma distribution.
        
        Parameters
        ----------
        hpi, zeta, delta, mu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = skewness,
            'k' = kurtosis.
            (default is 'mv')    
        Returns
        -------
        stats : sequence
            of requested moments.
            
        """
        if zeta <= 0 or delta <= 0:
            raise ValueError("The value of zeta and delta must be positive") 
            
        output = []
        
        if 'm' in moments:
            output.append(mu + delta * hpi * self._rlambda(zeta))
            
        if 'v' in moments:
            output.append(delta**2 * (1.0/zeta * self._rlambda(zeta) + hpi**2 * self._slambda(zeta)))
            
        if 's' in moments:
            output.append((hpi**3 * self._wlambda3(zeta) + 3.0 * hpi * self._wlambda2(zeta) / 
                zeta)/(hpi**2 * self._wlambda2(zeta) + self._wlambda1(zeta)/zeta)**(3/2))
            
        if 'k' in moments:
            output.append((hpi**4 * self._wlambda4(zeta) + 6.0 * hpi**2 * 
              self._wlambda3(zeta)/zeta + 3.0 * self._wlambda2(zeta)/zeta**2)/(hpi**2 * 
              self._wlambda2(zeta) + self._wlambda1(zeta)/zeta)**2)           
        
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)        
 
    # log-likelihood function for MLEs fitting procedure
    def _penalized_nnlf(self, logP, x):
        hpi, zeta, delta, mu = logP[0], math.exp(logP[1]), math.exp(logP[2]), logP[3]
        logpdf = np.log(self.pdf(x, hpi, zeta, delta, mu))
        finite_logpdf = np.isfinite(logpdf)
        n_bad = np.sum(~finite_logpdf, axis=0)
        if n_bad > 0:
            penalty = n_bad * math.log(self._XMAX) * 100
            return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
        return -np.sum(logpdf, axis=0)
    
    def fit(self, data):
        """
        Return MLEs for parameters from data.
        MLE stands for Maximum Likelihood Estimate. Starting estimates for
        the fit are given by ``self.fitstart(data)``, which uses method of moments.
 
        Parameters
        ----------
        data : array_like
            Data to use in calculating the MLEs.

        Returns
        -------
        mle_tuple : tuple of floats
            MLEs for any parameters
            
        Notes
        -----
        This fit is computed by maximizing a log-likelihood function, with
        penalty applied for samples outside of range of the distribution. The
        returned answer is not guaranteed to be the globally optimal MLE, it
        may only be locally optimal, or the optimization may fail altogether.
        """
        data = np.ravel(data)
        x0 = self._fitstart(data)
        optResults = optimize.minimize(self._penalized_nnlf, x0, args=(data), method='nelder-mead')
        logP = optResults.x
        P = logP[0], math.exp(logP[1]), math.exp(logP[2]), logP[3]
        return P
    
    # return starting point for fit using method of moments
    def _fitstart(self, data):
        fun1 = lambda P: self.stats(*P, moments='m') - np.mean(data)
        fun2 = lambda P: self.stats(*P, moments='v') - np.var(data)
        fun3 = lambda P: self.stats(*P, moments='s') - skew(data)
        fun4 = lambda P: self.stats(*P, moments='k') - kurtosis(data)
        def optimFun(logP):
            P = (logP[0], math.exp(logP[1]), math.exp(logP[2]), logP[3])
            return fun1(P)**2 + fun2(P)**2 + fun3(P)**2 + fun4(P)**2
     
        xi = (kurtosis(data)/3.0)**0.5
        chi = skew(data)/3.0
        xi = min(xi,0.999)
        adjust = 0.001
        if abs(chi) > xi:
            if xi < 0:
                chi = xi + adjust
            else:
                chi = xi - adjust
        hpi = chi/(xi**2 - chi**2)**0.5
        zeta = 3/xi**2 - 1
        rho = chi/xi
        delta = ((1 + zeta)**0.5 - 1) * (1 - rho**2)**0.5
        mu = np.mean(data) - delta * hpi * self._rlambda(zeta)
                          
        x0 = (hpi, math.log(zeta), math.log(delta), mu)
        optResults = optimize.minimize(optimFun, x0, method='nelder-mead')
        return tuple(optResults.x)

hyperb = hyperb_gen()