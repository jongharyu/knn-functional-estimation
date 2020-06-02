import numpy as np

from utility import find_unit_volume, incgamma


class SingleDensityFunctionalFormulas:
    def __init__(self, dims=None):
        self.dims = np.array(dims, dtype=float)
        self.unit_volumes = find_unit_volume(self.dims)

    def functionals(self, alphas, beta):
        return np.stack([self.shannon_entropy] +
                        [self.alpha_entropy(alpha) for alpha in alphas] +
                        [self.logarithmic_alpha_entropy(alpha) for alpha in alphas] +
                        [self.exponential_alpha_beta_entropy(alpha, beta) for alpha in alphas], 0)

    @property
    def shannon_entropy(self):
        raise NotImplementedError

    def alpha_entropy(self, alpha):
        raise NotImplementedError

    def logarithmic_alpha_entropy(self, alpha):
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
                                 incgamma(self.dims / 2, (r ** 2) / (2 * alpha))

    @property
    def shannon_entropy(self):
        dims = self.dims
        r = self.r

        entropy = incgamma(1 + dims / 2, (r ** 2) / 2) / incgamma(dims / 2, (r ** 2) / 2)
        entropy += np.log(self.cdr(1))

        return entropy

    def alpha_entropy(self, alpha):
        assert alpha > 0
        dims = self.dims

        entropy = self.cdr(1 / alpha) / ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha))

        return entropy

    def logarithmic_alpha_entropy(self, alpha):
        assert alpha > 0
        dims = self.dims

        entropy = (self.cdr(1 / alpha) * np.log(self.cdr(1)) /
                   ((alpha ** (dims / 2)) * (self.cdr(1) ** alpha))) + \
                  2 ** (dims / 2 - 1) * dims * self.unit_volumes * \
                  incgamma(dims / 2 + 1, alpha * (self.r ** 2) / 2) / \
                  ((alpha ** (dims / 2 + 1)) * (self.cdr(1) ** alpha))

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

    def logarithmic_alpha_entropy(self, alpha):
        return (self.r ** ((1 - alpha) * self.dims)) * self.dims * np.log(self.r)

    def exponential_alpha_beta_entropy(self, alpha, beta):
        return (self.r ** ((1 - alpha) * self.dims)) * np.exp(-beta / (self.r ** self.dims))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.plot(np.arange(1, 11),
             SingleDensityFunctionalFormulasGaussian(r=3, dims=np.arange(1, 11)).shannon_entropy,
             'x-')
