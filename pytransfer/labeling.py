import enum
from dataclasses import dataclass
from typing import Union, Any, Optional

from nptyping import NDArray
import numpy as np

class LabelingMethod(enum.Enum):
    UNIFORM: str = "uniform"
    KMEANS: str = "kmeans"

@dataclass
class Uniform():
    """
    The full number of clusters is actually n_clusters ** n_features
    """

    n_clusters: int = 8
    labels_: Optional[NDArray[Any, int]] = None
    n_features: int = 0
    min_: Optional[NDArray[Any, int]] = None
    max_: Optional[NDArray[Any, int]] = None
    step: Optional[NDArray[Any, int]] = None

    @property
    def n_clusters_per_dim(self):
        assert (self.n_features), "`n_features` must be specified before computing n_clusters_per_dim"
        _n_clusters_per_dim = int(self.n_clusters ** (1. / self.n_features))
        assert (self.n_clusters == int(_n_clusters_per_dim ** self.n_features)), "`n_clusters` must be a clean `n_features` power of some integer."
        return _n_clusters_per_dim

    def to_vector_labels(self, scalar_labels: NDArray[(Any), Any]) -> NDArray[(Any, Any), int]:
        """
        Performs the reverse of `to_scalar_labels`

        Example:
        with n_clusters = 12 and n_features = 2:
        - the label 123 becomes [10, 3]

        """
        n_samples = len(scalar_labels)

        vector_labels = np.zeros((n_samples, self.n_features))

        for i in range(self.n_features):
            multiple = (self.n_clusters_per_dim ** (self.n_features - (1 + i)))

            vector_labels[:, i] = scalar_labels // multiple
            scalar_labels = scalar_labels % multiple

        return vector_labels

    def to_scalar_labels(self, vector_labels: NDArray[(Any, Any), Any]) -> NDArray[Any, int]:
        """
        Now, we have to cast the labels from an array representation
        to a scalar representation.  The easiest way to do this is to
        consider that we can read the array as a single number with
        base n_clusters.

        Examples:

            - with n_clusters = 10 and n_features = 3: the label [0,
              5, 3] becomes 053 = 53.

            - with n_clusters = 12 and n_features = 2: the label [8,
              2] becomes 8 * 12 ** 1 + 2 * 12 ** 0 = 98.
        """
        n_samples, n_features = vector_labels.shape

        assert n_features == self.n_features

        labels = np.zeros(n_samples, dtype=int)

        for i in range(self.n_features):
            labels += (self.n_clusters_per_dim ** i) * vector_labels[:, -(1 + i)]

        return labels.astype(int)

    def _fit_cluster_centers(self):
        vector_labels = self.to_vector_labels(np.arange(self.n_clusters))

        self.cluster_centers_ = self.min_ + self.step * (0.5 + vector_labels)

    def fit(self, X: NDArray[(Any, Any), Any]):
        """
        :param X: of shape (n_samples, n_features). Training instances to cluster.
        """

        self.n_features = X.shape[1]

        # Initialize min_, max_, and step with the right shapes
        self.min_ = np.zeros(self.n_features)
        self.max_ = np.zeros(self.n_features)
        self.step = np.zeros(self.n_features)

        # To uniformly partition the points in a trajectory, we have
        # to determine the boundaries, i.e. min and max value for each axis.
        for i in range(self.n_features):
            self.min_[i] = np.amin(X[:, i])
            self.max_[i] = np.amax(X[:, i])

        # At this point, min_ and max_ will be the coordinates of
        # opposite corners on a hypercube that contains all points.
        # This is the most compact hypercube possible.

        # From the diagonal between the two points, we can determine a
        # step size so that each dimension will be partitioned into
        # n_clusters.

        self.step = ((self.max_ - self.min_) / self.n_clusters_per_dim)

        # We assign cluster labels for the clusters that partition
        # this hypercube.

        self._fit_cluster_centers()

        self.labels_ = self.predict(X)

        return self

    def _predict_vector(self, X: NDArray[(Any, Any), Any]) -> NDArray[Any, int]:
        """
        :param X: of shape (n_samples, n_features). Training instances to cluster.
        """

        # For every point, we identify how many steps it is from the minimum
        # We can combine these to derive a unique index for each cell.
        vector_labels = np.floor((X - self.min_) / self.step).astype(int)

        # At this point, each dimension of label will contain a number
        # from 0 up to *and including* n_clusters. Then, there would
        # actually be n_clusters + 1 clusters.

        # To resolve this, we group any entry with value n_clusters
        # into the label n_clusters-1

        return np.where(vector_labels == self.n_clusters_per_dim, self.n_clusters_per_dim - 1, vector_labels)

    def predict(self, X: NDArray[(Any, Any), Any]) -> NDArray[Any, int]:
        """
        :param X: of shape (n_samples, n_features). Training instances to cluster.
        """
        vector_labels = self._predict_vector(X)
        return self.to_scalar_labels(vector_labels)

    def fit_predict(self, X: NDArray[(Any, Any), Any]) -> NDArray[Any, int]:
        """
        :param X: of shape (n_samples, n_features). Training instances to cluster.
        """

        self.fit(X)
        return self.predict(X)


def test_uniform_predict_vector():
    uniform = Uniform(n_features=2, n_clusters=25, max_=np.array([0., 0.]), min_=np.array([0., 0.]), step=np.array([0.2, 0.2]))
    X = np.array([[0.3, 0.7], [0.5, 0.1], [1., 0.2]])
    y = np.array([[1, 3], [2, 0], [4, 1]])
    assert np.allclose(uniform._predict_vector(X), y)

def test_uniform_predict():
    uniform = Uniform (n_features=2, n_clusters=25, max_=np.array([0., 0.]), min_=np.array([0., 0.]), step=np.array([0.2, 0.2]))
    X = np.array([[0.3, 0.7], [0.5, 0.1], [1., 0.2]])
    y = np.array([8, 10, 21])
    assert np.allclose(uniform.predict(X), y)

def test_fit_predict():
    uniform = Uniform(n_clusters=25)
    X = np.array([[0., .4], [0.3, 0.7], [0.5, 0.], [1., 0.2], [0.2, 1.]])
    y = np.array([2, 8, 10, 21, 9])
    assert np.allclose(uniform.fit_predict(X), y)

def test_fit_predict_1d():
    uniform = Uniform(n_clusters=5)
    X = np.array([[0.], [0.3], [0.5], [1.,]])
    y = np.array([0, 1, 2, 4])
    assert np.allclose(uniform.fit_predict(X), y)

def test_to_vector_labels():
    uniform = Uniform(n_features=2,n_clusters=25, max_=np.array([0., 0.]), min_=np.array([0., 0.]), step=np.array([0.2, 0.2]))

    y = np.array([[1, 3], [2, 0], [4, 1]])
    assert np.allclose(y, uniform.to_vector_labels(uniform.to_scalar_labels(y)))

def test_fit_cluster_labels():
    uniform = Uniform(n_clusters=9)
    X = np.array([[0., .4], [0.3, 0.7], [0.5, 0.], [1.2, 0.2], [0.2, 1.2]])
    uniform.fit(X)
    centers = np.array([[0.2, 0.2], [0.2, 0.6], [0.2, 1.], [0.6, 0.2], [0.6, 0.6], [0.6, 1.], [1., 0.2], [1., 0.6], [1., 1.]])
    assert np.allclose(uniform.cluster_centers_, centers)
