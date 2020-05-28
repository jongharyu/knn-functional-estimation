import numpy as np
from scipy.special import binom, digamma, gamma
import warnings
warnings.filterwarnings("error")

from utility import compute_normalized_volumes


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


def phi_asymptotic_nn_classification_error(u, v, ks, ls):
    """

    Parameters
    ----------
    u: np.array
        (m, len(ks))
    v: np.array
        (m, len(ls))
    ks: np.array
    ls: np.array

    Returns
    -------
    phi: (1, m)
    """
    assert u.shape[1] == len(ks)
    assert v.shape[1] == len(ls)
    assert len(ks) == len(ls)

    def phi_kl(w, k, l):
        """

        Parameters
        ----------
        w: (m, )
            shorthand notation for u/v
        k: int
        l: int

        Returns
        -------
        phi: (m, )
        """
        phi = - (w >= 1).astype(float) * (1 - 1 / w) ** (k + l - 2)
        for i in range(l):
            phi += binom(k + l - 2, i) * (-1 / w) ** i
        phi *= (-w) ** (l - 1) / binom(k + l - 2, k - 1)
        phi = 1 - phi
        return phi

    w = u / v
    phi = np.stack([
        phi_kl(w[:, idx_k], ks[idx_k], ls[idx_k])
        for idx_k in range(len(ks))], 1)  # (m, len(ks))

    return phi


def phi_jensen_shannon_divergence(u, v, ks, ls):
    """

    Parameters
    ----------
    u: np.array
        (m, len(ks))
    v: np.array
        (m, len(ls))
    ks: np.array
    ls: np.array

    Returns
    -------
    phi: (1, m)
    """
    assert u.shape[1] == len(ks)
    assert v.shape[1] == len(ls)
    assert len(ks) == len(ls)

    def phi_kl(u, v, k, l):
        """

        Parameters
        ----------
        u: (m, )
        v: (m, )
        k: int
        l: int

        Returns
        -------
        phi: (m, )
        """

        def b1(w, k, l):
            # bkl when w is less than 1
            s = 0
            for j in range(l - 1):
                s += binom(k + l - 2, j) * (-w) ** (l - 1 - j) / (l - 1 - j)
            s /= binom(k + l - 2, k - 1)
            return s

        def b2(w, k, l):
            # bkl when w is greater than 1
            s = 0
            for i in range(k - 1):
                s += binom(k + l - 2, i) * (-1 / w) ** (k - 1 - i) / (k - 1 - i)
            for i in range(k + l - 1):
                if i != k - 1:
                    s -= binom(k + l - 2, i) * (-1.) ** (k - 1 - i) / (k - 1 - i)
            s /= binom(k + l - 2, k - 1)
            s -= np.log(w)
            return s

        def b(w, k, l):
            return b1(w, k, l) * (w < 1) + b2(w, k, l) * (w >= 1)

        l = max(l, 2)
        w = u / v
        phi = np.log(2) + (l - 1) / k * w * (np.log(2) + digamma(l - 1 + 1e-5) - digamma(k + 1) + np.log(w)) + \
              (b(w, k, l) + (l - 1) / k * w * b(w, k + 1, l - 1))
        phi /= 2
        return phi

    phi = np.stack([
        phi_kl(u[:, idx_k], v[:, idx_k], ks[idx_k], ls[idx_k])
        for idx_k in range(len(ks))], 1)  # (m, len(ks))

    return phi