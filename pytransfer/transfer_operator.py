"""

Contains code to implement the transfer operator approach

Author: Jesse Hoogland
Year: 2020

"""
import logging
from dataclasses import dataclass
from typing import Union, List, Optional, Any

import numpy as np
from nptyping import NDArray
from scipy import linalg
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from tqdm.contrib import tenumerate

from pynamics.utils import eigsort, normalize_rows, svd_whiten
from pynamics.trajectories import TimeSeries
from .labeling import LabelingMethod, Uniform

class TransferOperator:
    """
    Class for implementing the transfer operator approach.
    TODO: Treat this as more of a wrapper around a numpy array

    """
    def __init__(
        self,
        labeling_method: LabelingMethod = LabelingMethod.UNIFORM,
        n_clusters: int = 100,
    ):
        """
        :param labeling_method: one of either "kmeans" or "uniform"
            (the latter doesn't work for high-dimensional systems, but
            it's more interpretatble for low-dimensional systems)
        :param n_clusters:

        """
        self.eigvals: Optional[NDArray[Any, float]] = None
        self.eigvecs: Optional[NDArray[(Any, Any), float]] = None

        self.forward_matrix: Optional[NDArray[(Any, Any), float]] = None

        self.labeling_method = labeling_method
        self.n_clusters = n_clusters

        self.labeler_is_fit = False
        if self.labeling_method == "kmeans":
            self.labeler = KMeans(n_clusters=self.n_clusters)

        elif self.labeling_method == "uniform":
            self.labeler = Uniform(n_clusters=self.n_clusters)
        else:
            raise ValueError(
                f"labeling_method must be one of 'kmeans', 'unifom', but is {self.labeling_method}"
            )

    @property
    def cluster_centers(self) -> NDArray[float]:
        return self.labeler.cluster_centers_

    @property
    def invariant_dist(self) -> NDArray[Any, float]:
        """
        :returns invariant_dist: (np.ndarray of shape [trans_matrix.shape[0],])
        i.e. a row vector with values $[1, 1, 1,...,1] / \sqrt{N}$

        TODO: Why isn't this normalized
        """
        assert (
            not self.forward_matrix is None
        ), "Need to fit TransferOperator before accessing `invariant_dist`"

        return self.eigvecs[:, 0] / np.sum(self.eigvecs[:, 0])

    @property
    def reverse_matrix(self) -> np.ndarray:
        assert (
            not self.forward_matrix is None
        ), "Need to fit TransferOperator before accessing `reverse_matrix`"

        divisor = self.invariant_dist

        divisor[divisor == 0] = 1. # So we don't get any infs

        return (np.diag(1. / divisor) * self.forward_matrix.T *
                np.diag(self.invariant_dist)).real # Don't know why this is being converted

    @property
    def balanced_matrix(self) -> np.ndarray:
        assert (
                not self.forward_matrix is None
        ), "Need to fit TransferOperator before accessing `balanced_matrix`"

        return (self.forward_matrix + self.reverse_matrix) / 2.0

    def get_labels(self,
                   time_series: TimeSeries,
                   verbose: bool = False) -> np.ndarray:
        """
        produces the cluster identity labels for each entry in a time
        series `time_series` using `n_clusters`.  decides what kind of
        labels to produce according to `self.labeling_method`

        :param time_series: (np.ndarray of shape [t, d])

        :returns cluster_identities: (np.ndarray of shape [t],
            dtype="int32", with values chosen from `range(0,
            n_clusters)`) representing the cluster identities of each
            time entry.

        NOTE: Working
        """

        # if we haven't yet called this method, we have yet to fit our labels.
        if not self.labeler_is_fit:
            self.labeler.fit(time_series)
            self.labeler_is_fit = True

        # Once we've fit our labels (or if we've already done so), we
        # can predict them explicitly
        return self.labeler.predict(time_series)

    def _get_forward_weights(self,
                             cluster_labels: np.ndarray,
                             n_future_timesteps: int,
                             verbose: bool = False) -> np.ndarray:
        """
        We approximate the transfer operator as a finite-rank matrix $P_{i,j}(\tau, N)$, where
        entry $(i,j)$ counts the number of transitions between cluster $i$ and $j$ in
        a time window $\tau$.
        Here, $N$ is the number of clusters, i.e. $P$ will have shape $(N, N)$.
        """

        # We have discretized time, so $\tau$ becomes a discrete number of steps, `n_future_timesteps`.
        # This has to be at least 1, else we would not be able to measure any transitions.
        assert n_future_timesteps >= 1 and isinstance(
            n_future_timesteps,
            int), "`n_future_timesteps` must be an integer greater than 1"

        # For efficiency, we use sparse matrices
        forward_weights = np.zeros((self.n_clusters, self.n_clusters))

        cluster_labels_range = range(len(cluster_labels) - n_future_timesteps - 1)

        # For a verbose readout, we log our progress along the trajectory
        if verbose:
            cluster_labels_range = tqdm(cluster_labels_range,
                                        desc="Creating the transfer matrix")

        for i in cluster_labels_range:
            #print(i, i + n_future_timesteps, cluster_labels[i], cluster_labels[i + n_future_timesteps])
            forward_weights[cluster_labels[i],
                            cluster_labels[i + n_future_timesteps]] += 1

        return forward_weights

    @property
    def forward_entropy(self):
        """
        Calculates the entropy rate of a given partitioning (of $$N$$ sites) through

        $$ h(N)=-\sum_{i,j=1}^N \pi_i P_{ij}(dt, N)\log P_{ij}(dt,N),$$
        where $$P$$ is the transfer matrix and $$\pi$$ is its first left eigenvector, the invariant distribution.

        """
        forward_matrix = self.forward_matrix
        invariant_dist = self.invariant_dist.reshape([1, -1])

        forward_matrix_ones = forward_matrix.copy()
        # To avoid infinities in np.log
        forward_matrix_ones[forward_matrix == 0] = 1

        return -np.sum(invariant_dist *
                       (forward_matrix * np.log(forward_matrix_ones)))

    @property
    def reverse_entropy(self):
        """
        Calculates the reverse entropy rate of a given partitioning (of $$N$$ sites) through

        $$ h(N)=-\sum_{i,j=1}^N \pi_i P_{ij}(dt, N)\log P_{ij}(dt,N),$$
        where $$P$$ is the transfer matrix and $$\pi$$ is its first left eigenvector, the invariant distribution.

        """
        reverse_matrix = self.reverse_matrix
        invariant_dist = self.invariant_dist.reshape([1, -1])

        reverse_matrix_ones = reverse_matrix.copy()
        # To avoid infinities in np.log
        reverse_matrix_ones[reverse_matrix == 0] = 1

        return -np.sum((invariant_dist * (reverse_matrix * np.log(reverse_matrix_ones))))

    @property
    def entropy_production(self):
        return self.forward_entropy - self.reverse_entropy

    def _decompose(self, k: Optional[int]=None, which: str="LM"):
        self.eigvals, self.eigvecs = eigsort(self.forward_matrix.T,
                                             k,
                                             which=which)

    def fit(self,
            time_series: TimeSeries,
            n_future_timesteps: int = 1,
            verbose: bool = False,
            k: int = None,
            which="LR") -> "TransferOperator":

        # 1. Convert the time_series into a series of cluster labels
        cluster_labels = self.get_labels(time_series)

        # 2. Compute the unnormalized trans_matrix from this series of cluster labels
        forward_weights = self._get_forward_weights(cluster_labels,
                                                    n_future_timesteps,
                                                    verbose=verbose)

        # 3. Normalize
        self.forward_matrix = normalize_rows(forward_weights)

        # 4. Compute eigenvalue decomposition
        self._decompose(k, which=which)

        return self

    def fit_t_imp(self, time_series: TimeSeries, eigval_idxs: List[int],
                  tau: int, transition_time: float) -> float:

        n_eigvals = max(eigval_idxs) + 1

        self.fit(time_series, n_future_timesteps=tau, k=n_eigvals)

        eigvals = self.eigvals[eigval_idxs]
        t_imps = -transition_time * tau / (np.log(np.abs(eigvals)))

        return t_imps

def test_get_delay_embedding_1():
    transfer_operator = TransferOperator(n_delays=3)
    assert (transfer_operator.get_delay_embedding(
        np.arange(10).reshape([10, 1])) == np.array([
            np.arange(0, 7),
            np.arange(1, 8),
            np.arange(2, 9),
            np.arange(3, 10)
        ]).T).all()
    assert (transfer_operator.get_delay_embedding(
        np.concatenate(
            [np.arange(10).reshape([10, 1]),
             np.arange(10).reshape([10, 1])],
            axis=1)) == np.array([
                np.arange(0, 7),
                np.arange(0, 7),
                np.arange(1, 8),
                np.arange(1, 8),
                np.arange(2, 9),
                np.arange(2, 9),
                np.arange(3, 10),
                np.arange(3, 10),
            ]).T).all()


def test_get_trans_matrix_unnormalized_1():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator._get_trans_matrix_unnormalized(
            np.array([0, 0, 1, 2, 1, 0, 2, 1]), 3, 1).toarray(),
        np.array([[1, 1, 1], [1, 0, 1], [0, 2, 0]]),
    ).all()


def test_get_trans_matrix_unnormalized_2():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator._get_trans_matrix_unnormalized(
            np.array([0, 0, 1, 2, 1, 0, 2, 1]), 3, 2).toarray(),
        np.array([[1, 3, 2], [1, 1, 2], [1, 2, 0]]),
    ).all()


def test_get_invariant_dist():
    transfer_operator = TransferOperator()

    assert np.isclose(
        transfer_operator.get_invariant_dist(
            np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]])),
        np.array([1, 1, 1]) / np.sqrt(3),
    ).all()


def test_get_forward_entropy():
    transfer_operator = TransferOperator()

    forward_entropy = transfer_operator._get_forward_entropy(
        np.array([[0.5, 0.1, 0.4], [0, 0.5, 0.5], [0.25, 0.25, 0.5]]),
        np.array([1, 1, 1]) / np.sqrt(3),
    )
    print(forward_entropy)
    assert np.isclose(
        forward_entropy,
        1.545114226461558,
    )


def test_get_first_nondecreasing_element_idx_1():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3])) == 5


def test_get_first_nondecreasing_element_idx_2():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3]), 6) == 8


def test_get_first_nondecreasing_element_idx_3():
    transfer_operator = TransferOperator()

    assert transfer_operator.get_first_nondecreasing_element_idx(
        np.array([1, .5, .25, .2, .15, .05, .06, .03, 0, .3]), 8) == 8


def test_uniform_clusters_1():
    transfer_operator = TransferOperator(labeling_method="uniform")

    assert np.isclose(transfer_operator.get_labels(np.arange(0, 11), 10),
                      np.arange(0, 11)).all()


def test_uniform_clusters_2():
    transfer_operator = TransferOperator(labeling_method="uniform")

    assert np.isclose(
        transfer_operator.get_labels(np.arange(0, 10.5, 0.5), 10),
        np.floor_divide(np.arange(0, 21), 2)).all()


# def test_time_inverse_transfer_operator():
#     trans_matrix = sp.csr_matrix(
#         np.array([[0, 0.5, 0.5], [0.2, 0.5, 0.3], [0.1, 0.05, 0.85]])
#     )
#     transfer_operator = TransferOperator(n_delays=1)
#     assert np.isclose(
#         np.array(
#             transfer_operator.get_revers_transfer_operator(
#                 trans_matrix
#             ).todense(),
#             dtype="float64",
#         ),
#         (
#             np.array(
#                 [[0, 0.2, 0.1], [0.5, 0.5, 0.05], [0.5, 0.3, 0.85]], dtype="float64"
#             )
#         ),
#     ).all()
