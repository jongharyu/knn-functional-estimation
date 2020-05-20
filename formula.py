import numpy as np
from scipy.special import gamma
from scipy.special import gammainc

from utility import find_unit_volume


class SingleDensityFunctionalFormulas:
    def __init__(self, dims=None):
        self.dims = np.array(dims, dtype=float)
        self.unit_volumes = find_unit_volume(self.dims)

    def functionals(self, alphas, beta):
        return np.stack([self.shannon_entropy] +
                        [self.alpha_entropy(alpha) for alpha in alphas] +
                        [self.generalized_alpha_entropy(alpha) for alpha in alphas] +
                        [self.exponential_alpha_beta_entropy(alpha, beta) for alpha in alphas], 0)

    @property
    def shannon_entropy(self):
        raise NotImplementedError

    def alpha_entropy(self, alpha):
        raise NotImplementedError

    def generalized_alpha_entropy(self, alpha):
        raise NotImplementedError

    def exponential_alpha_beta_entropy(self, alpha, beta):
        raise NotImplementedError


class SingleDensityFunctionalFormulasGaussian(SingleDensityFunctionalFormulas):
    def __init__(self, r=np.inf, dims=None):
        """
        Arguments
        ---------
        dims: np.array
        r: float
            truncation radius
        """
        super().__init__(dims)

        self.r = r
        self.cdr = lambda alpha: (2 ** (self.dims / 2 - 1)) * self.dims * self.unit_volumes * \
                                 gammainc(self.dims / 2, (r ** 2) / (2 * alpha)) * gamma(self.dims / 2)

    @property
    def shannon_entropy(self):
        dims = self.dims
        r = self.r

        entropy = gammainc(1 + dims / 2, (r ** 2) / 2) / gammainc(dims / 2, (r ** 2) / 2) * \
                  gamma(1 + dims / 2) / gamma(dims / 2)
        entropy += np.log(self.cdr(1))

        return entropy

    def alpha_entropy(self, alpha):
        assert alpha > 0
        dims = self.dims

        entropy = self.cdr(1 / alpha) / ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha))

        return entropy

    def generalized_alpha_entropy(self, alpha):
        assert alpha > 0
        dims = self.dims

        entropy = (self.cdr(1 / alpha) * np.log(self.cdr(1)) /
                   ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha))) + \
                  2 ** (dims / 2 - 1) * dims * self.unit_volumes * \
                  gammainc(dims / 2 + 1, alpha * (self.r ** 2) / 2) * \
                  gamma(dims / 2 + 1) / ((alpha ** (dims / 2 + 1)) * (self.cdr(1) ** alpha))

        return entropy

    def exponential_alpha_beta_entropy(self, alpha, beta):
        # Don't have closed form expressions
        return 0 * self.dims


class SingleDensityFunctionalFormulasUniform(SingleDensityFunctionalFormulas):
    def __init__(self, r=1, dims=None):
        """
        Arguments
        ---------
        dims: np.array
        r: float
            Unif([0,a]^dims)
        """
        super().__init__(dims=dims)
        self.r = r

    @property
    def shannon_entropy(self):
        return self.dims * np.log(self.r)

    def alpha_entropy(self, alpha):
        return self.r ** ((1 - alpha) * self.dims)

    def generalized_alpha_entropy(self, alpha):
        return (self.r ** ((1 - alpha) * self.dims)) * self.dims * np.log(self.r)

    def exponential_alpha_beta_entropy(self, alpha, beta):
        return (self.r ** ((1 - alpha) * self.dims)) * np.exp(-beta / (self.r ** self.dims))


class DoubleDensityFunctionalFormulas:
    def __init__(self, dims=None):
        """
        Arguments
        ---------
        dims: np.array
        """
        self.dims = np.array(dims, dtype=float)
        self.unit_volumes = find_unit_volume(self.dims)

    def functionals(self, alphas):
        return np.stack([self.kl_divergence] +
                        [self.alpha_divergence(alpha) for alpha in alphas] +
                        [self.generalized_alpha_divergence(alpha) for alpha in alphas],
                        0)

    @property
    def kl_divergence(self):
        raise NotImplementedError

    def alpha_divergence(self, alpha):
        raise NotImplementedError

    def generalized_alpha_divergence(self, alpha):
        raise NotImplementedError


class DoubleDensityFunctionalFormulasGaussian(DoubleDensityFunctionalFormulas):
    def __init__(self, r=np.inf, sigma=1, dims=None):
        """
        Arguments
        ---------
        dims: np.array
        r: float
            truncation radius
        """
        super().__init__(dims=dims)
        assert sigma >= 1.

        self.r = r
        self.sigma = sigma
        self.const = gammainc(self.dims / 2 + 1, (r ** 2) / 2) * gamma(self.dims / 2 + 1) / \
                     (gammainc(self.dims / 2, (r ** 2) / 2) * gamma(self.dims / 2))
        self.cdr = lambda alpha: (2 ** (self.dims / 2 - 1)) * self.dims * self.unit_volumes * \
                                 gammainc(self.dims / 2, (r ** 2) / (2 * alpha)) * gamma(self.dims / 2)

    @property
    def kl_divergence(self):
        dims = self.dims
        divergence = dims * np.log(self.sigma) + np.log(self.cdr(self.sigma ** 2) / self.cdr(1)) - \
                     (1 - (self.sigma ** (-2.))) * self.const

        return divergence

    def alpha_divergence(self, alpha):
        assert alpha >= 1

        dims = self.dims
        alpha_tilde = 1 - (1 - alpha) * (1 - (self.sigma ** (-2.)))

        divergence = ((self.sigma ** (-2.)) ** (dims / 2 * (1 - alpha))) / (alpha_tilde ** (dims / 2)) * \
                     self.cdr(1 / alpha_tilde) / self.cdr(1) * \
                     (self.cdr(self.sigma ** 2) / self.cdr(1)) ** (alpha - 1)

        return divergence

    def generalized_alpha_divergence(self, alpha):
        assert alpha >= 1

        dims = self.dims
        alpha_tilde = 1 - (1 - alpha) * (1 - (self.sigma ** (-2.)))

        # Formula for $p**{\alpha-1}\ln (p/q)$
        unit_volumes = self.unit_volumes
        divergence = 1 / 2 * self.cdr(1 / alpha) * dims * np.log((self.sigma ** 2)) / \
                     ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha)) - \
                     ((1 - self.sigma ** (-2.)) * (2 ** (dims / 2 - 1)) * dims * unit_volumes *
                      gammainc(dims/2 + 1, alpha * (self.r ** 2)) * gamma(dims / 2 + 1) /
                      ((alpha ** (1 + (dims / 2))) * (self.cdr(1) ** alpha))) + \
                     (self.cdr(1 / alpha) / ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha)) *
                      np.log(self.cdr(self.sigma ** 2) / self.cdr(1)))

        return divergence


class DoubleDensityFunctionalFormulasUniform(DoubleDensityFunctionalFormulas):
    def __init__(self, a=1, b=1, dims=None):
        """
        Arguments
        ---------
        dims: np.array
        a: float
            X ~ Unif([0,a]^dims)
        b: float
            Y ~ Unif([0,b]^dims)
        """
        super().__init__(dims=dims)
        # assert a <= b
        self.a = np.float(a)
        self.b = np.float(b)

    @property
    def kl_divergence(self):
        return self.dims * np.log(self.b / self.a)

    def alpha_divergence(self, alpha):
        return (self.b / self.a) ** ((alpha - 1) * self.dims)

    def generalized_alpha_divergence(self, alpha):
        return (self.a ** ((1 - alpha) * self.dims)) * self.dims * np.log(self.b / self.a)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot(np.arange(1, 11),
             SingleDensityFunctionalFormulasGaussian(r=3, dims=np.arange(1, 11)).shannon_entropy,
             'x-')

    print(DoubleDensityFunctionalFormulasGaussian(r=np.inf, sigma=2, dims=np.arange(1, 11)).functionals(alphas=[1]))
    print(DoubleDensityFunctionalFormulasGaussian(r=10, sigma=2, dims=np.arange(1, 11)).functionals(alphas=[1]))
