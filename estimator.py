import numpy as np
from scipy.special import digamma
from scipy.special import gamma
import warnings
warnings.filterwarnings("error")

from utility import compute_normalized_volumes


class NNSingleFunctionalEstimator:
    def __init__(self, ks=None, alphas=None, beta=0.):
        """

        Parameters
        ----------
        ks: np.array
            array of k values
        alphas: np.array
            array of alpha values
        beta: float
        """
        self.ks = np.array(ks)
        self.alphas = alphas
        self.beta = beta
        self.functional_names = \
            [r'Shannon entropy'] + \
            [r'{}-entropy'.format(alpha) for alpha in alphas] + \
            [r'Generalized ${}$-entropy'.format(alpha) for alpha in alphas] + \
            [r'Exponential $({},{})$-entropy'.format(alpha, beta) for alpha in alphas]
        self.num_functionals = len(self.functional_names)

    def phi(self, u):
        r"""

        Parameters
        ----------
        u: np.array of size (m, len(self.ks))
            array of normalized volume $U_{km}$ for k in self.ks

        Returns
        -------
        phis: np.array of size (self.num_functionals, m, len(self.ks))
            stack of estimator function values for each functional

        """
        assert (u > 0).all()
        phis = np.stack([phi_shannon_entropy(u, self.ks)] +
                        [phi_alpha_entropy(u, self.ks, alpha) for alpha in self.alphas] +
                        [phi_generalized_alpha_entropy(u, self.ks, alpha) for alpha in self.alphas] +
                        [phi_exponential_alpha_beta_entropy(u, self.ks, alpha, self.beta) for alpha in self.alphas],
                        0)  # (num_functionals, m, len(ks))
        return phis

    def estimate(self, x):
        """
        Arguments
        ---------
        x: np.array
            data points (m, dim)
        """
        # Compute normalized volumes
        u = compute_normalized_volumes(x, ks=self.ks)  # (m, len(ks))

        # Compute estimates for each k in ks by taking the mean over samples
        return self.phi(u).mean(1)  # (num_functionals, len(ks))


def phi_shannon_entropy(u, ks):
    return np.log(u) - digamma(ks)


def phi_alpha_entropy(u, ks, alpha):
    return np.exp(np.log(gamma(ks)) - np.log(gamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1))) * \
           (u ** (1 - alpha))


def phi_generalized_alpha_entropy(u, ks, alpha):
    return np.exp(np.log(gamma(ks)) - np.log(gamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1))) * \
           (u ** (1 - alpha)) * (np.log(u) - digamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1))


def phi_exponential_alpha_beta_entropy(u, ks, alpha, beta):
    return np.exp(np.log(gamma(ks)) - np.log(gamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1))) * \
           ((u - beta) * (u >= beta).astype(float)) ** np.maximum(ks - alpha, 0) / (u ** (ks - 1))


class NNDoubleFunctionalEstimator:
    def __init__(self, ks=None, ls=None, alphas=None):
        """
        
        Parameters
        ----------
        ks: np.array
            array of integer k values
        ls: np.array
            array of integer l values
        alphas: np.array
            array of alpha values for the (logarithmic) alpha divergences
        """
        assert len(ks) == len(ls)
        self.ks = np.array(ks)
        self.ls = np.array(ls)
        self.alphas = alphas
        self.functional_names = ['KL divergence'] + \
                                ['{}-divergence'.format(alpha) for alpha in alphas] + \
                                ['Generalized {}-divergence'.format(alpha) for alpha in alphas]
        self.num_functionals = len(self.functional_names)

    def phi(self, u, v):
        assert (u > 0).all() and (v > 0).all()

        phis = np.stack([phi_kl_divergence(u, v, self.ks, self.ls)] +
                        [phi_alpha_divergence(u, v, self.ks, self.ls, alpha) for alpha in self.alphas] +
                        [phi_generalized_alpha_divergence(u, v, self.ks, self.ls, alpha) for alpha in self.alphas],
                        0)  # (num_functionals, m, len(ks))
        return phis

    def estimate(self, x, y):
        """
        Arguments
        ---------
        x: np.array
            data points (m, dim)
        y: np.array
            data points (n, dim)
        """
        u = compute_normalized_volumes(x, ks=self.ks)
        v = compute_normalized_volumes(x, ks=self.ls, y=y)

        # Compute estimates for each k in ks by taking the mean over samples
        estimates = self.phi(u, v).mean(1)  # (num_functionals, len(ks))

        return estimates


def phi_kl_divergence(u, v, ks, ls):
    return -np.log(u) + np.log(v) + digamma(ks) - digamma(ls)


def phi_alpha_divergence(u, v, ks, ls, alpha):
    return np.exp(np.log(gamma(ks)) - np.log(gamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1)) +
                  np.log(gamma(ls)) - np.log(gamma(np.maximum(ls, np.ceil(1 - alpha + 1e-5)) + alpha - 1))) * \
           (u / v) ** (1 - alpha)


def phi_generalized_alpha_divergence(u, v, ks, ls, alpha):
    return np.exp(np.log(gamma(ks)) - np.log(gamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1))) * \
           u ** (1 - alpha) * (digamma(np.maximum(ks, np.ceil(alpha - 1 + 1e-5)) - alpha + 1) -
                               digamma(ls) + np.log(v) - np.log(u))