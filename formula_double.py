import numpy as np

from utility import find_unit_volume, incgamma


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
                        [self.logarithmic_alpha_divergence(alpha) for alpha in alphas],
                        0)

    @property
    def kl_divergence(self):
        raise NotImplementedError

    def alpha_divergence(self, alpha):
        raise NotImplementedError

    def logarithmic_alpha_divergence(self, alpha):
        raise NotImplementedError

    @property
    def asymptotic_nn_classification_error(self):
        raise NotImplementedError

    @property
    def jensen_shannon_divergence(self):
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
        self.cdr = lambda alpha: (2 ** (self.dims / 2 - 1)) * self.dims * self.unit_volumes * \
                                 incgamma(self.dims / 2, (r ** 2) / (2 * alpha))

    @property
    def kl_divergence(self):
        dims = self.dims

        r = self.r
        divergence = dims * np.log(self.sigma) + np.log(self.cdr(self.sigma ** 2) / self.cdr(1)) - \
                     (1 - (self.sigma ** (-2.))) * incgamma(self.dims / 2 + 1, (r ** 2) / 2) / \
                     (incgamma(self.dims / 2, (r ** 2) / 2))

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
        r = self.r

        # Formula for $p**{\alpha-1}\ln (p/q)$
        unit_volumes = self.unit_volumes
        divergence = 1 / 2 * self.cdr(1 / alpha) * dims * np.log((self.sigma ** 2)) / \
                     ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha)) - \
                     ((1 - self.sigma ** (-2.)) * (2 ** (dims / 2 - 1)) * dims * unit_volumes *
                      incgamma(dims/2 + 1, alpha * (r ** 2)) /
                      ((alpha ** (1 + (dims / 2))) * (self.cdr(1) ** alpha))) + \
                     (self.cdr(1 / alpha) / ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha)) *
                      np.log(self.cdr(self.sigma ** 2) / self.cdr(1)))

        return divergence

    def logarithmic_alpha_divergence(self, alpha):
        dims = self.dims
        r = self.r
        alpha_tilde = 1 - (1 - alpha) * (1 - (self.sigma ** (-2.)))

        divergence = (self.sigma ** dims * self.cdr(self.sigma ** 2) / self.cdr(1)) ** (alpha - 1) * \
                     (np.log(self.sigma ** dims * self.cdr(self.sigma ** 2) / self.cdr(1)) *
                      self.cdr(1 / alpha_tilde) / self.cdr(1) / (alpha_tilde ** (dims / 2)) -
                      (1 - (self.sigma ** (-2.))) *
                      incgamma(self.dims / 2 + 1, (alpha_tilde * r ** 2) / 2) /
                      (incgamma(self.dims / 2, (r ** 2) / 2)) / alpha_tilde ** (dims / 2 + 1))

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

    def logarithmic_alpha_divergence(self, alpha):
        return ((self.a / self.b) ** ((1 - alpha) * self.dims)) * self.dims * np.log(self.b / self.a)

    @property
    def asymptotic_nn_classification_error(self):
        p = (1 / self.a) ** self.dims
        q = (1 / self.b) ** self.dims
        return q / (p + q)

    @property
    def jensen_shannon_divergence(self):
        c = (self.a / self.b) ** self.dims
        return (np.log(2) - 1 / 2 * (c * np.log(1 + 1 / c) + np.log(1 + c))) / 2


if __name__ == '__main__':
    print(DoubleDensityFunctionalFormulasGaussian(r=np.inf, sigma=2, dims=np.arange(1, 11)).functionals(alphas=[1]))
    print(DoubleDensityFunctionalFormulasGaussian(r=10, sigma=2, dims=np.arange(1, 11)).functionals(alphas=[1]))
