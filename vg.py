import math
import numpy as np
from numpy import (inf, nan, isnan, isinf, isclose)
from scipy import (special, interpolate, optimize)
from scipy.integrate import quad, trapz
from scipy.stats.mstats import rankdata
from scipy.stats import (skew, kurtosis)

# Frozen RV class
class rv_frozen(object):

    def __init__(self, dist, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        self.args = c, sigma, theta, nu
        
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
    
class vg_gen(object):
    
    # The largest [in magnitude] usable floating value.
    _XMAX = np.finfo(float).max
 
    def freeze(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        """Freeze the distribution for the given arguments.
        Parameters
        ----------
        c, sigma, thata, nu : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.
        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        return rv_frozen(self, c, sigma, theta, nu)
    
    def __call__(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        return self.freeze(c, sigma, theta, nu)
    
    def pdf(self, x, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        """
        Probability density function at x of the VarianceGamma distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
            
        x = np.atleast_1d(x)
        
        mask = isclose(x,c)
        d = np.full(x.shape,nan)
        mx = x[~mask]
        
        if isclose(nu,2.0):
            d[mask] = inf
            d[~mask] = (((2.0 * np.exp(theta * (mx - c)/sigma**2))/(nu**(1.0/nu) * 
                (2.0 * math.pi)**0.5 * sigma * special.gamma(1/nu))) * ((abs(mx - c)/
                (2.0 * (sigma**2)/nu + theta**2)**0.5)**(1.0/nu - 0.5)) *
                special.kv(1.0/nu - 0.5, (1/sigma**2) * abs(mx - c) *
                ((2.0 * sigma**2/nu)**0.5 + theta**2)))
        else:
            if nu < 2.0:
                if isclose(nu**(1.0/nu), 0.0):
                    d = inf
                else:
                    d[mask] = (special.gamma(1.0/nu - 0.5)/(sigma * (2.0 * math.pi)**0.5 * 
                        nu**(1.0/nu) * special.gamma(1.0/nu)) * ((2.0 * sigma**2/(2 * 
                        sigma**2/nu + theta**2))**(1.0/nu - 0.5)))
                    d[~mask] = (((2.0 * np.exp(theta * (mx - c)/sigma**2))/(nu**(1.0/nu) * 
                        (2.0 * math.pi)**0.5 * sigma * special.gamma(1.0/nu))) * 
                        ((abs(mx - c)/(2.0 * (sigma**2)/nu + 
                        theta**2)**0.5)**(1.0/nu - 0.5)) * special.kv(1.0/nu - 0.5, (1.0/sigma**2) * 
                        abs(mx - c) * ((2.0 * sigma**2/nu) + theta**2)**0.5))
            if nu > 2.0:
                d[mask] = inf
                d[~mask] = (((2.0 * np.exp(theta * (mx - c)/sigma**2))/(nu**(1.0/nu) * 
                    (2.0 * math.pi)**0.5 * sigma * special.gamma(1.0/nu))) * ((abs(mx - c)/
                    (2.0 * (sigma**2)/nu + theta**2)**0.5)**(1.0/nu - 0.5)) *
                    special.kv(1.0/nu - 0.5, (1.0/sigma**2) * abs(mx - c) *
                    ((2 * sigma**2/nu) + theta**2)**0.5))
                        
        d = np.where(isnan(d),inf,d)
        
        return d
    
    def pdf_deriv(self, x, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        """
        Probability density function  derivative at x of the VarianceGamma distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        pdf_deriv : ndarray
            Probability density function derivative evaluated at x
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")  
            
        x = np.atleast_1d(x)

        mask = isclose(x,c)
        dd = np.full(x.shape,nan)
        mx = x[~mask]
        
        if isclose(nu,2):
            dd[~mask] = (np.exp(theta * (mx - c)/sigma**2) * 2.0**(0.5) * abs(mx - c)**(-0.5 * 
            (-2.0 + nu)/nu) * ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.25 * 
            (-2.0 + nu)/nu) * (theta * special.kv(0.5 * (-2.0 + nu)/nu, 1.0/sigma**2 * 
            abs(mx - c) * ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.5)) - ((mx - c)/abs(mx - 
            c)) * special.kv(0.5 * (-2.0 + 3.0 * nu)/nu, 1.0/sigma**2 * abs(mx - c) * 
            ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.5)) * ((2.0 * sigma**2 + theta**2 * 
            nu)/nu)**(0.5)) * nu**(-1.0/nu)/math.pi**(0.5)/sigma**3.0/special.gamma(1.0/nu))
        elif nu < 2:
            dd[mask] = 0.0
            dd[~mask] = (np.exp(theta * (mx - c)/sigma**2) * 2.0**(0.5) * 
            abs(mx - c)**(-0.5 * (-2.0 + nu)/nu) * ((2.0 * 
            sigma**2 + theta**2 * nu)/nu)**(0.25 * (-2.0 + 
            nu)/nu) * (theta * special.kv(0.5 * (-2.0 + nu)/nu, 1.0/sigma**2 * 
            abs(mx - c) * ((2.0 * sigma**2 + theta**2 * 
            nu)/nu)**(0.5)) - ((mx - c)/abs(mx - c)) * special.kv(0.5 * (-2.0 + 3.0 * nu)/nu, 
            1/sigma**2 * abs(mx - c) * ((2 * sigma**2 + theta**2 * nu)/nu)**(0.5)) * 
            ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.5)) * 
            nu**(-1.0/nu)/math.pi**(0.5)/sigma**3.0/special.gamma(1.0/nu))
        else:
            dd[~mask] = (np.exp(theta * (mx - c)/sigma**2) * 2.0**(0.5) * 
            abs(mx - c)**(-0.5 * (-2 + nu)/nu) * ((2.0 * 
            sigma**2 + theta**2 * nu)/nu)**(0.25 * (-2.0 + 
            nu)/nu) * (theta * special.kv(0.5 * (-2.0 + nu)/nu, 1.0/sigma**2 * 
            abs(mx - c) * ((2.0 * sigma**2 + theta**2 * 
            nu)/nu)**(0.5)) - ((mx - c)/abs(mx - c)) * special.kv(0.5 * (-2.0 + 3.0 * nu)/nu, 
            1.0/sigma**2 * abs(mx - c) * ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.5)) * 
            ((2.0 * sigma**2 + theta**2 * nu)/nu)**(0.5)) * 
            nu**(-1.0/nu)/math.pi**(0.5)/sigma**3.0/special.gamma(1.0/nu))
    
        return dd 
        
        
    def cdf(self, x, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):     
        """
        Cumulative distribution function  derivative at x of the VarianceGamma distribution.
        
        Parameters
        ----------
        x : array_like
            quantiles
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at x
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
            
        x = np.atleast_1d(x)
        xs = np.sort(x)
        
        # Get the breaks and add -inf and inf 
        xi = self._breaks(c, sigma, theta, nu)
        xi = np.concatenate([[-inf],xi,[inf]])
    
        # Evaluate cdf at break points
        int_xi = np.zeros(1)
        for j in range(1,xi.size-3):
            int_xi = np.append(int_xi, quad(self.pdf, xi[j], xi[j+1], args=(c, sigma, theta, nu))[0]+int_xi[j-1])
        
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
        yint = self.pdf(xint, c, sigma, theta, nu)
        resint = trapz(yint,xint)
        
        # Accumulate
        resint[mx==0] = 0
        for j in range(1,xi.size-2):
            resint[mx==j] = int_xi[j-1] + resint[mx==j]
        resint[mx==(xi.size-2)] = 1
        
        return resint.flatten()[rankdata(x).astype(int)-1].reshape(resint.shape)
    
    def _breaks(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        """
        Breaks function for the VarianceGamma distribution.
        
        Parameters
        ----------
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        breaks : tuple
            Coordinates of breaking points dividing distribution into 8 regions
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
            
        deriv=0.3
        epsTiny = 10**(-10)
        epsSmall = 10**(-6)
        
        xMode = self.mode(c, sigma, theta, nu)
        xTiny = self._treshold(c, sigma, theta, nu, xMode, upper = False, tol = epsTiny)
        xSmall = self._treshold(c, sigma, theta, nu, xMode, upper = False, tol = epsSmall)
        xLarge = self._treshold(c, sigma, theta, nu, xMode, upper = True, tol = epsSmall)
        xHuge = self._treshold(c, sigma, theta, nu, xMode, upper = True, tol = epsTiny)
        
        xDeriv = np.linspace(xSmall, xMode, 101)
        derivVals = self.pdf_deriv(xDeriv, c, sigma, theta, nu)
        maxDeriv = np.max(derivVals)
        breakSize = deriv * maxDeriv
        breakFun = lambda x: self.pdf_deriv(x, c, sigma, theta, nu) - breakSize
        if (maxDeriv < breakSize) or (derivVals[0] > breakSize):
            xBreakLow = xSmall
        else:
            whichMaxDeriv = derivVals.argmax(0)
            xBreakLow = optimize.brentq(breakFun, a=xSmall, b=xDeriv[whichMaxDeriv])
    
        xDeriv = np.linspace(xMode, xLarge, 101)
        derivVals = -self.pdf_deriv(xDeriv, c, sigma, theta, nu)
        maxDeriv = np.max(derivVals)
        breakSize = deriv * maxDeriv
        breakFun = lambda x: -self.pdf_deriv(x, c, sigma, theta, nu) - breakSize
        if (maxDeriv < breakSize) or (derivVals[-1] > breakSize):
            xBreakHigh = xLarge
        else:
            whichMaxDeriv = derivVals.argmax(0)
            xBreakHigh = optimize.brentq(breakFun, a=xDeriv[whichMaxDeriv], b=xLarge)
    
        rng = (xTiny, xSmall, xBreakLow, xMode, xBreakHigh, xLarge, xHuge)
        
        return rng 
    
    def mode(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0):
        """
        Mode for the VarianceGamma distribution.
        
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
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
            
        if nu >= 2:
            m = c
        else:
            mFun = lambda x: -np.log(self.pdf(x, c, sigma, theta, nu))
            optResults = optimize.minimize(mFun, c, method='BFGS')
            m = optResults.x
    
        return m
    
    def _treshold(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0, mode = 0.0, upper = False, tol = 10**(-5)):
        """
        Treshold function for the VarianceGamma distribution.
        
        Parameters
        ----------
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
        mode : array_like
            Mode of distribution
        upper : boolean, optional
            If True then the upper treshold of distribution is returned,
            otherwise lower treshold (default=False)
        tol : array_like, optional
            The tolerance at which the treshold is calculated (default=10**(-5))
        Returns
        -------
        treshold : float
            Treshold at which the probability is equal to tol 
            (or 1-tol in case upper = True)
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
     
        tresFun = lambda x: self.pdf(x, c, sigma, theta, nu) - tol
        sd = np.where(upper,1,-1)*(sigma**2 + theta**2 * nu)**0.5
              
        b = mode + sd
        while self.pdf(b, c, sigma, theta, nu) > tol:
            b += sd
        interval = np.sort((mode,b))
        xtres = optimize.brentq(tresFun, a=interval[0], b=interval[1])
    
        return xtres
    
    def ppf(self, q, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0): 
        """
        Percent point function (inverse of `cdf`) at q of the VarianceGamma distribution.
        
        Parameters
        ----------
        q : array_like
            lower tail probability
        c, sigma, theta, nu : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information) (default=0, 1, 0, 1)
            
        Returns
        -------
        ppf : ndarray
            quantile corresponding to the lower tail probability q.
            
        """
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive") 
            
        eps = 0.00001
    
        # Make sure q is numpy array and sort the values
        q = np.atleast_1d(q)
        qs = np.sort(q)
      
        xi = self._breaks(c, sigma, theta, nu)
        xi = np.concatenate([[-inf],xi,[inf]])
        
        # Create masks that contains index of a range
        mx = np.full(q.shape, xi.size-2, dtype=np.int32)
        for j in range(xi.size-1):
            a = self.cdf(xi[j], c, sigma, theta, nu)
            b = self.cdf(xi[j+1], c, sigma, theta, nu)
            mx[(qs >= a) & (qs < b)] = j
    
        interval = np.sort(np.stack((xi[mx], xi[mx+1]), axis=-1))
        
        knots = [[]]
        for j in range(1,xi.size-2):
            if np.any(mx==j):
                xVal = np.linspace(xi[j]-eps, xi[j+1]+eps, 100)
                cdfVal = self.cdf(xVal, c, sigma, theta, nu)
                knots.append(interpolate.splrep(xVal, cdfVal, s=0))
            else:
                knots.append([])
                
        qqs = np.full(qs.shape,nan)           
        for index in np.ndindex(qs.shape):
            if mx[index] == 0:
                zeroFun = lambda x: self.cdf(x, c, sigma, theta, nu) - qs[index]
                a, b = xi[1] - (xi[2] - xi[1]), xi[1] 
                while zeroFun(a) * zeroFun(b) > 0:
                    a -= xi[2] - xi[1]
                qqs[index] = optimize.brentq(zeroFun, a=a, b=b)
            elif mx[index] == (xi.size-2):
                zeroFun = lambda x: self.cdf(x, c, sigma, theta, nu) - qs[index]
                a, b = xi[xi.size-2], xi[xi.size-2] + (xi[xi.size-2] - xi[xi.size-3]) 
                while zeroFun(a) * zeroFun(b) > 0:
                    a += xi[xi.size-2] - xi[xi.size-3]
                qqs[index] = optimize.brentq(zeroFun, a=a, b=b)       
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
        
        return qqs.flatten()[rankdata(q).astype(int)-1].reshape(qqs.shape)
    
    def stats(self, c = 0.0, sigma = 1.0, theta = 0.0, nu = 1.0, moments = 'mv'):
        """
        Some statistics of the VarianceGamma distribution.
        
        Parameters
        ----------
        c, sigma, theta, nu : array_like
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
        if sigma <= 0 or nu <= 0:
            raise ValueError("The value of sigma and nu must be positive")
            
        output = []
        
        if 'm' in moments:
            output.append(c+theta)
            
        if 'v' in moments:
            output.append(sigma**2 + theta**2 * nu)
            
        if 's' in moments:
            output.append((2.0 * theta**3 * nu**2 + 3.0 * sigma**2 * theta * nu)/
            ((theta**2 * nu + sigma**2)**1.5))
            
        if 'k' in moments:
            output.append((2.0 * theta**3 * nu**2 + 3.0 * sigma**2 * theta * nu)/
             ((theta**2 * nu + sigma**2)**1.5))            
        
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)
    
    # log-likelihood function for MLEs fitting procedure
    def _penalized_nnlf(self, logP, x):
        c, sigma, theta, nu = logP[0], math.exp(logP[1]), logP[2], math.exp(logP[3])
        sigma = max(sigma, 0.001)
        nu = max(nu, 0.001)
        logpdf = np.log(self.pdf(x, c, sigma, theta, nu))
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
        P = logP[0], math.exp(logP[1]), logP[2], math.exp(logP[3])
        return P
    
    # return starting point for fit using method of moments
    def _fitstart(self, data):
        fun1 = lambda P: self.stats(*P, moments='m') - np.mean(data)
        fun2 = lambda P: self.stats(*P, moments='v') - np.var(data)
        fun3 = lambda P: self.stats(*P, moments='s') - skew(data)
        fun4 = lambda P: self.stats(*P, moments='k') - kurtosis(data)
        def optimFun(logP):
            P = (logP[0], math.exp(logP[1]), logP[2], math.exp(logP[3]))
            return fun1(P)**2 + fun2(P)**2 + fun3(P)**2 + fun4(P)**2
                            
        sigma = np.var(data)**0.5
        nu = (kurtosis(data) - 3.0)/3.0
        theta = (skew(data) * sigma)/(3.0 * nu)
        nu = max(nu, 0.001)
    
        c = np.mean(data) - theta
        x0 = (c, math.log(sigma), theta, math.log(nu))
        optResults = optimize.minimize(optimFun, x0, method='nelder-mead')
        return tuple(optResults.x)

vg = vg_gen()