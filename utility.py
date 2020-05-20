import numpy as np
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors


def find_unit_volume(dims):
    return np.pi ** (dims / 2) / gamma((dims / 2) + 1)


def compute_normalized_volumes(x, ks=None, y=None):
    """
    Compute normalized volumes U_k(x|y) for points x with respect to reference y

    Parameters
    ----------
    x: np.array
        evaluation sample
    y: None or np.array
        reference sample
        if y is not specified, y is identified as x.
    ks: np.array
    Returns
    -------
    u: np.array
        (len(ks), m)
    """
    if y is None:
        y = x
        k_shift = 1  # as 1-NN is trivially 0 if y==x
    else:
        assert x.shape[1] == y.shape[1]
        k_shift = 0

    m, dim = y.shape
    k_max = max(ks)

    # Find k-NN distances
    knn_distances, _ = NearestNeighbors(n_neighbors=k_max + k_shift).fit(y).kneighbors(x)  # (m, k_max+k_shift), _
    knn_distances = knn_distances[:, k_shift:]  # (m, k_max)
    knn_distances = knn_distances[:, ks - 1]  # (m, len(ks))

    # Compute normalized volume
    u = (m - k_shift) * find_unit_volume(dim) * (knn_distances ** dim)  # (len(ks), m) matrix

    return u


def draw_truncated_gaussians(m, d, sigma=1, r=np.inf):
    # m: sample size
    # d: dimension
    # r: truncation radius
    # sigma: standard deviation

    m_curr = 0
    samples = np.zeros((m, d))
    while m_curr < m:
        gaussian_samples = sigma * np.random.randn(m, d)
        new_samples = gaussian_samples[np.sum(gaussian_samples ** 2, 1) <= r ** 2, :]

        start_idx = m_curr
        m_new = min(m - m_curr, new_samples.shape[0])
        end_idx = m_curr + m_new

        samples[start_idx:end_idx, :] = new_samples[:m_new, :]
        m_curr += m_new

    return samples
