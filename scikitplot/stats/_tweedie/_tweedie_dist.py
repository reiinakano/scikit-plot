from __future__ import division

import numpy as np
from scipy.stats import rv_continuous, poisson, gamma, invgauss, norm
from scipy.special import gammaln, gammainc
from scipy import optimize

__all__ = [
  'tweedie_gen',
  'tweedie',
]


class tweedie_gen(rv_continuous):
    r"""A Tweedie continuous random variable.
    
    .. dropdown:: View aliases

        **Main aliases**
        
        `scikitplot.stats._tweedie.tweedie_gen`

        **Compat aliases**
        
        `scikitplot.stats.tweedie_gen`
        
    See Also
    --------
    tweedie : An instance of `tweedie_gen`, providing Tweedie distribution functionality.

    Notes
    -----
    Tweedie is a family of distributions belonging to the class of exponential
    dispersion models.

    .. math::
        f(x; \mu, \phi, p) = a(x, \phi, p) \exp((y \theta - \kappa(\theta))
        / \phi)

    where :math:`\theta = {\mu^{1-p}}{1-p}` when :math:`p \ne 1` and
    :math:`\theta = \log(\mu)` when :math:`p = 1`, and :math:`\kappa(\theta) =
    [\{(1 - p) \theta + 1\} ^ {(2 - p) / (1 - p)} - 1] / (2 - p)`
    for :math:`p \ne 2` and :math:`\kappa(\theta) = - \log(1 - \theta)` for
    :math:`p = 2`.

    Except in a few special cases (discussed below) :math:`a(x, \phi, p)` is
    hard to to write out.

    This class incorporates the Series method of evaluation of the Tweedie
    density for :math:`1 < p < 2` and :math:`p > 2`. There are special cases
    at :math:`p = 0, 1, 2, 3` where the method is equivalent to the Gaussian
    (Normal), Poisson, Gamma, and Inverse Gaussian (Normal).

    For cdfs, only the special cases and :math:`1 < p < 2` are implemented.
    The author has not found any documentation on series evaluation of the cdf
    for :math:`p > 2`.

    Additionally, the R package `tweedie` also incorporates a (potentially)
    faster method that involves a Fourier inversion. This method is harder
    to understand, so I've not implemented it. However, others should feel free
    to attempt to add this themselves.

    Examples
    --------
    The density can be found using the pdf method.

    >>> tweedie(p=1.5, mu=1, phi=1).pdf(1) # doctest:+ELLIPSIS
    0.357...

    The cdf can be found using the cdf method.

    >>> tweedie(p=1.5, mu=1, phi=1).cdf(1) # doctest:+ELLIPSIS
    0.603...

    The ppf can be found using the ppf method.
    
    >>> tweedie(p=1.5, mu=1, phi=1).ppf(0.603) # doctest:+ELLIPSIS
    0.998...

    References
    ----------
    Dunn, Peter K. and Smyth, Gordon K. 2001, Tweedie Family Densities: Methods
    of Evaluation

    Dunn, Peter K. and Smyth, Gordon K. 2005, Series evaluation of Tweedie
    exponential dispersion model densities
    """
    def _pdf(self, x, p, mu, phi):
        return np.exp(self._logpdf(x, p, mu, phi))

    def _logpdf(self, x, p, mu, phi):
        p = np.broadcast_to(p, x.shape)
        mu = np.broadcast_to(mu, x.shape)
        phi = np.broadcast_to(phi, x.shape)
        return estimate_tweedie_loglike_series(x, mu, phi, p)

    def _logcdf(self, x, p, mu, phi):
        p = np.broadcast_to(p, x.shape)
        mu = np.broadcast_to(mu, x.shape)
        phi = np.broadcast_to(phi, x.shape)
        return estimate_tweeide_logcdf_series(x, mu, phi, p)

    def _cdf(self, x, p, mu, phi):
        return np.exp(self._logcdf(x, p, mu, phi))

    def _rvs(self, p, mu, phi, size=None, random_state=None):
        if size is None:
            size = self._size
        if random_state is None:
            random_state = self._random_state
        p = np.array(p, ndmin=1)
        if not (p > 1).all() & (p < 2).all():
            raise ValueError('p only valid for 1 < p < 2')
        rate = est_kappa(mu, p) / phi
        scale = est_gamma(phi, p, mu)
        shape = -est_alpha(p)
        N = poisson(rate).rvs(size=size, random_state=random_state)
        mask = N > 0
        if not np.isscalar(scale) and len(scale) == len(mask):
            scale = scale[mask]
        if not np.isscalar(shape) and len(shape) == len(mask):
            shape = shape[mask]

        rvs = gamma(
                a=N[mask] * shape,
            scale=scale).rvs(size=np.sum(mask), random_state=random_state)
        rvs2 = np.zeros(N.shape, dtype=rvs.dtype)
        rvs2[mask] = rvs
        return rvs2

    def _ppf_single1to2(self, q, p, mu, phi, left, right):
        args = p, mu, phi

        factor = 10.
        while self._ppf_to_solve(left, q, *args) > 0.:
            right = left
            left /= factor
            # left is now such that cdf(left) < q

        while self._ppf_to_solve(right, q, *args) < 0.:
            left = right
            right *= factor
            # right is now such that cdf(right) > q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    def _ppf(self, q, p, mu, phi):
        p = np.broadcast_to(p, q.shape)
        mu = np.broadcast_to(mu, q.shape)
        phi = np.broadcast_to(phi, q.shape)

        single1to2v = np.vectorize(self._ppf_single1to2, otypes='d')

        ppf = np.zeros(q.shape, dtype=float)

        # Gaussian
        mask = p == 0
        if np.sum(mask) > 0:
            ppf[mask] = norm(loc=mu[mask],
                             scale=np.sqrt(phi[mask])).ppf(q[mask])

        # Poisson
        mask = p == 1
        if np.sum(mask) > 0:
            ppf[mask] = poisson(mu=mu[mask] / phi[mask]).ppf(q[mask])

        # 1 < p < 2
        mask = (1 < p) & (p < 2)
        if np.sum(mask) > 0:
            zero_mass = np.zeros_like(ppf)
            zeros = np.zeros_like(ppf)
            zero_mass[mask] = self._cdf(zeros[mask], p[mask], mu[mask],
                                        phi[mask])
            right = 10 * mu * phi ** p
            cond1 = mask
            cond2 = q > zero_mass
            if np.sum(cond1 & ~cond2) > 0:
                ppf[cond1 & ~cond2] = zeros[cond1 & ~cond2]
            if np.sum(cond1 & cond2) > 0:
                single1to2v = np.vectorize(self._ppf_single1to2, otypes='d')
                mask = cond1 & cond2
                ppf[mask] = single1to2v(q[mask], p[mask], mu[mask],
                                        phi[mask], zero_mass[mask],
                                        right[mask])

        # Gamma
        mask = p == 2
        if np.sum(mask) > 0:
            ppf[mask] = gamma(a=1/phi[mask],
                              scale=phi[mask] * mu[mask]).ppf(q[mask])

        # Inverse Gamma
        mask = p == 3
        if np.sum(mask) > 0:
            ppf[mask] = invgauss(mu=mu[mask] * phi[mask],
                                 scale=1 / phi[mask]).ppf(q[mask])
        return ppf

    def _argcheck(self, p, mu, phi):
        cond1 = (p == 0) | (p >= 1)
        cond2 = mu > 0
        cond3 = phi > 0
        return cond1 & cond2 & cond3

    # def _argcheck(self, arg):
    #     return True

almost_zero = np.nextafter(0, -1)

tweedie = tweedie_gen(name='tweedie', a=almost_zero, b=np.inf,
                      shapes='p, mu, phi')
tweedie.__doc__ = """\
An instance of `tweedie_gen`, providing Tweedie distribution functionality.
    
.. dropdown:: View aliases

    **Main aliases**
    
    `scikitplot.stats._tweedie.tweedie`

    **Compat aliases**
    
    `scikitplot.stats.tweedie`

This instance provides:

- probability density function (pdf)
- cumulative distribution function (cdf)
- random sampling

for the Tweedie distribution.

The Tweedie distribution is part of the exponential dispersion family, characterized
by a `p` parameter that determines its behavior:

- `p = 0`: Gaussian distribution
- `p = 1`: Poisson distribution
- `p = 2`: Gamma distribution
- `p = 3`: Inverse Gaussian distribution
- `1 < p < 2`: Compound Poisson-Gamma distribution

Parameters
----------
p : float
    Tweedie power parameter.
mu : float
    Mean or location parameter.
phi : float
    Dispersion parameter, controlling the variance of the distribution.
        
See Also
--------
tweedie_gen : A Tweedie continuous random variable.


Examples
--------
Compute the pdf and cdf at a given point:

.. jupyter-execute::

    >>> import numpy as np
    >>> from scikitplot.stats import tweedie
    >>> x = 2.0
    >>> pdf_val = tweedie.pdf(x, p=1.5, mu=1, phi=1)
    >>> cdf_val = tweedie.cdf(x, p=1.5, mu=1, phi=1)
    >>> pdf_val, cdf_val


Generate random variates:

.. jupyter-execute::

    >>> import numpy as np
    >>> from scikitplot.stats import tweedie
    >>> rvs = tweedie.rvs(p=1.5, mu=1, phi=1, size=16)
    >>> rvs

Plot the pdf over a range:

.. plot::

    >>> import numpy as np
    >>> from scikitplot.stats import tweedie    
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 100)
    >>> y = tweedie.pdf(x, p=1.5, mu=1, phi=1)
    >>> plt.plot(x, y, label='Tweedie pdf (p=1.5, mu=1, phi=1)')
    >>> plt.xlabel("x")
    >>> plt.ylabel("Density")
    >>> plt.legend()
    >>> plt.show()
"""

def est_alpha(p):
    return (2 - p) / (1 - p)


def est_jmax(x, p, phi):
    return x ** (2 - p) / (phi * (2 - p))


def est_kmax(x, p, phi):
    return x ** (2 - p) / (phi * (p - 2))


def est_theta(mu, p):
    theta = np.where(
        p == 1,
        np.log(mu),
        mu ** (1 - p) / (1 - p)
    )
    return theta


def est_kappa(mu, p):
    kappa = np.where(
        p == 2,
        np.log(mu),
        mu ** (2 - p) / (2 - p)
    )
    return kappa


def est_gamma(phi, p, mu):
    mu = np.array(mu, dtype=float)
    return phi * (p - 1) * mu ** (p - 1)


def estimate_tweedie_loglike_series(x, mu, phi, p):
    """Estimate the loglikihood of a given set of x, mu, phi, and p

    Parameters
    ----------
    x : array
        The observed values. Must be non-negative.
    mu : array
        The fitted values. Must be positive.
    phi : array
        The scale paramter. Must be positive.
    p : array
        The Tweedie variance power. Must equal 0 or must be greater than or
        equal to 1.

    Returns
    -------
    estiate_tweedie_loglike_series : float
    """
    x = np.array(x, ndmin=1)
    mu = np.array(mu, ndmin=1)
    phi = np.array(phi, ndmin=1)
    p = np.array(p, ndmin=1)

    ll = np.ones_like(x) * -np.inf

    # Gaussian (Normal)
    gaussian_mask = p == 0.
    if np.sum(gaussian_mask) > 0:
        ll[gaussian_mask] = norm(
                loc=mu[gaussian_mask],
                scale=np.sqrt(phi[gaussian_mask])).logpdf(x[gaussian_mask])

    # Poisson
    poisson_mask = p == 1.
    if np.sum(poisson_mask) > 0:
        poisson_pdf = poisson(
                mu=mu[poisson_mask] / phi[poisson_mask]).pmf(
                x[poisson_mask] / phi[poisson_mask]) / phi[poisson_mask]
        zero_mask = poisson_pdf != 0.
        poisson_logpdf = ll[poisson_mask]
        poisson_logpdf[zero_mask] = np.log(poisson_pdf[zero_mask])
        ll[poisson_mask] = poisson_logpdf

    # 1 < p < 2
    ll_1to_2_mask = (1 < p) & (p < 2)
    if np.sum(ll_1to_2_mask) > 0:
        # Calculating logliklihood at x == 0 is pretty straightforward
        zeros = x == 0
        mask = zeros & ll_1to_2_mask
        ll[mask] = -(mu[mask] ** (2 - p[mask]) / (phi[mask] * (2 - p[mask])))
        mask = ~zeros & ll_1to_2_mask
        ll[mask] = ll_1to2(x[mask], mu[mask], phi[mask], p[mask])

    # Gamma
    gamma_mask = p == 2
    if np.sum(gamma_mask) > 0:
        ll[gamma_mask] = gamma(a=1/phi, scale=phi * mu).logpdf(x[gamma_mask])

    # (2 < p < 3) or (p > 3)
    ll_2plus_mask = ((2 < p) & (p < 3)) | (p > 3)
    if np.sum(ll_2plus_mask) > 0:
        zeros = x == 0
        mask = zeros & ll_2plus_mask
        ll[mask] = -np.inf
        mask = ~zeros & ll_2plus_mask
        ll[mask] = ll_2orMore(x[mask], mu[mask], phi[mask], p[mask])

    # Inverse Gaussian (Normal)
    invgauss_mask = p == 3
    if np.sum(invgauss_mask) > 0:
        cond1 = invgauss_mask
        cond2 = x > 0
        mask = cond1 & cond2
        ll[mask] = invgauss(
                mu=mu[mask] * phi[mask],
                scale=1. / phi[mask]).logpdf(x[mask])
    return ll


def ll_1to2(x, mu, phi, p):
    def est_z(x, phi, p):
        alpha = est_alpha(p)
        numerator = x ** (-alpha) * (p - 1) ** alpha
        denominator = phi ** (1 - alpha) * (2 - p)
        return numerator / denominator

    if len(x) == 0:
        return 0

    theta = est_theta(mu, p)
    kappa = est_kappa(mu, p)
    alpha = est_alpha(p)
    z = est_z(x, phi, p)
    constant_logW = np.max(np.log(z)) + (1 - alpha) + alpha * np.log(-alpha)
    jmax = est_jmax(x, p, phi)

    # Start at the biggiest jmax and move to the right
    j = max(1, jmax.max())

    def _logW(alpha, j, constant_logW):
        # Is the 1 - alpha backwards in the paper? I think so.
        logW = (j * (constant_logW - (1 - alpha) * np.log(j)) -
                np.log(2 * np.pi) - 0.5 * np.log(-alpha) - np.log(j))
        return logW

    def _logWmax(alpha, j):
        logWmax = (j * (1 - alpha) - np.log(2 * np.pi) -
                   0.5 * np.log(-alpha) - np.log(j))
        return logWmax

    # e ** -37 is approxmiately the double precision on 64-bit systems.
    # So we just need to calcuate logW whenever its within 37 of logWmax.
    logWmax = _logWmax(alpha, j)
    while np.any(logWmax - _logW(alpha, j, constant_logW) < 37):
        j += 1
    j_hi = np.ceil(j)

    j = max(1, jmax.min())
    logWmax = _logWmax(alpha, j)

    while (np.any(logWmax - _logW(alpha, j, constant_logW) < 37) and
           np.all(j > 1)):
        j -= 1
    j_low = np.ceil(j)

    j = np.arange(j_low, j_hi + 1, dtype=np.float64)
    w1 = np.tile(j, (z.shape[0], 1))

    w1 *= np.log(z)[:, np.newaxis]
    w1 -= gammaln(j + 1)
    logW = w1 - gammaln(-alpha[:, np.newaxis] * j)

    logWmax = np.max(logW, axis=1)
    w = np.exp(logW - logWmax[:, np.newaxis]).sum(axis=1)

    return (logWmax + np.log(w) - np.log(x) + (((x * theta) - kappa) / phi))


def ll_2orMore(x, mu, phi, p):
    alpha = est_alpha(p)
    kappa = est_kappa(mu, p)
    theta = est_theta(mu, p)

    def est_z(x, phi, p):
        alpha = est_alpha(p)
        numerator = (p - 1) ** alpha * phi ** (alpha - 1)
        denominator = phi ** alpha * (p - 2)
        return numerator / denominator

    def _logVenv(z, p, k):
        alpha = est_alpha(p)
        logVenv = (k * (np.log(z) + (1 - alpha) - np.log(k) + alpha *
                        np.log(alpha * k)) + 0.5 * np.log(alpha))
        return logVenv

    def _logVmax(p, k):
        alpha = est_alpha(p)
        return (1 - alpha) * k + 0.5 * np.log(alpha)

    kmax = est_kmax(x, phi, p)
    logVmax = _logVmax(p, kmax)
    z = est_z(x, phi, p)

    # e ** -37 is approxmiately the double precision on 64-bit systems.
    # So we just need to calcuate logVenv whenever its within 37 of logVmax.
    k = max(1, kmax.max())
    while np.any(logVmax - _logVenv(z, p, k) < 37):
        k += 1

    k_hi = k

    k = max(1, kmax.min())
    while np.any(logVmax - _logVenv(z, p, k) < 37) and np.all(k > 1):
        k -= 1

    k_lo = k

    k = np.arange(k_lo, k_hi + 1, dtype=np.float64)
    k = np.tile(k, (z.shape[0], 1))
    v1 = gammaln(1 + alpha[:, np.newaxis] * k)
    v1 += k * (alpha[:, np.newaxis] - 1) * np.log(phi[:, np.newaxis])
    v1 += alpha[:, np.newaxis] * k * np.log(p[:, np.newaxis] - 1)
    v1 -= gammaln(1 + k)
    v1 -= k * np.log(p[:, np.newaxis] - 2)
    logV = v1 - alpha[:, np.newaxis] * k * np.log(x[:, np.newaxis])

    logVmax = np.max(logV, axis=1)

    # This part is hard to log... so don't
    v2 = (-1) ** k * np.sin(-k * np.pi * alpha[:, np.newaxis])
    v = (np.exp(logV - logVmax[:, np.newaxis]) * v2).sum(axis=1)
    V = np.exp(logVmax + np.log(v))

    return (np.log(V / (np.pi * x)) +
            ((x * theta - kappa) / phi))


def estimate_tweeide_logcdf_series(x, mu, phi, p):
    """Estimate the logcdf of a given set of x, mu, phi, and p

    Parameters
    ----------
    x : array
        The observed values. Must be non-negative.
    mu : array
        The fitted values. Must be positive.
    phi : array
        The scale paramter. Must be positive.
    p : array
        The Tweedie variance power. Must equal 0 or must be greater than or
        equal to 1.

    Returns
    -------
    estiate_tweedie_loglike_series : float
    """
    x = np.array(x, ndmin=1)
    mu = np.array(mu, ndmin=1)
    phi = np.array(phi, ndmin=1)
    p = np.array(p, ndmin=1)

    logcdf = np.zeros_like(x)

    # Gaussian (Normal)
    mask = p == 0
    if np.sum(mask) > 0:
        logcdf[mask] = norm(loc=mu[mask],
                            scale=np.sqrt(phi[mask])).logcdf(x[mask])

    # Poisson
    mask = p == 1.
    if np.sum(mask) > 0:
        logcdf[mask] = np.log(poisson(mu=mu[mask] / phi[mask]).cdf(x[mask]))

    # 1 < p < 2
    mask = (1 < p) & (p < 2)
    if np.sum(mask) > 0:
        cond1 = mask
        cond2 = x > 0
        mask = cond1 & cond2
        logcdf[mask] = logcdf_1to2(x[mask], mu[mask], phi[mask], p[mask])
        mask = cond1 & ~cond2
        logcdf[mask] = -(mu[mask] ** (2 - p[mask]) /
                         (phi[mask] * (2 - p[mask])))

    # Gamma
    mask = p == 2
    if np.sum(mask) > 0:
        logcdf[mask] = gamma(a=1/phi[mask],
                             scale=phi[mask] * mu[mask]).logcdf(x[mask])

    # Inverse Gaussian (Normal)
    mask = p == 3
    if np.sum(mask) > 0:
        logcdf[mask] = invgauss(mu=mu[mask] * phi[mask],
                                scale=1 / phi[mask]).logcdf(x[mask])

    return logcdf


def logcdf_1to2(x, mu, phi, p):
    # I couldn't find a paper on this, so gonna be a little hacky until I
    # have a better idea. The strategy is to create a (n, 1) matrix where
    # n is the number of observations and the first column represents where
    # there are 0 occurences. We'll add an additional column for 1 occurence,
    # and test for whether the difference between the added's column value
    # and the max value is greater than 37. If not, add another column
    # until that's the case. Then, sum the columns to give a vector of length
    # n which *should* be the CDF. (I think).

    # For very high rates, this funciton might not run well as it will
    # create lots of (potentially meaningless) columns.
    rate = est_kappa(mu, p) / phi
    scale = est_gamma(phi, p, mu)
    shape = -est_alpha(p)
    W = -rate.reshape(-1, 1)

    i = 0
    while True:
        i += 1
        trial = i * np.log(rate) - rate - gammaln(i + 1)
        # trial += gamma(a=i * shape, scale=scale).logcdf(x)
        trial += np.log(gammainc(i * shape, x / scale))
        W = np.hstack((W, trial.reshape(-1, 1)))

        if (np.all(W[:, :-1].max(axis=1) - W[:, -1] > 37) &
                np.all(W[:, -2] > W[:, -1])):
            break
    logcdf = np.log(np.exp(W).sum(axis=1))
    return logcdf