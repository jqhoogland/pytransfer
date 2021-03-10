# # TODO: All of this

# @dataclass
# class PhaseSpaceReconstructor(TransferOperator):
#     n_delays: int = 0
#     embedding_dim: int = 1

#     def get_max_entropy_partitioning(self,
#                                      time_series,
#                                      clusters_range=np.arange(100, 2500, 100),
#                                      n_future_timesteps=1):
#         """
#         Determines the maximum entropy of ``time_series`` over partitions of varying
#         numbers of cells, specified in ``clusters_range``

#         :param time_series: (np.ndarray of shape [t, X]), note that the spatial dimension can be any number.
#         :param clusters_range: (list-like) a selection of clusters sizes to choose from.
#         defaults to ``np.arange(100,2500,100)``

#         :returns max_entropy: the maximum entropy over all partitions considered.
#         :returns trans_matrix: the transfer matrix produced from the corresponding partitioning,
#         :returns invariant_dist: the invariant distribution produced from the corresponding partitioning,

#         """
#         logging.debug(
#             "------------------------------------------------------------")
#         logging.debug(
#             "get_max_entropy_partitioning:INPUTS:time_series:SHAPE:{}".format(
#                 time_series.shape))
#         logging.debug(
#             "get_max_entropy_partitioning:INPUTS:clusters_range:SHAPE:{}".
#             format(clusters_range.shape))
#         logging.debug(
#             "get_max_entropy_partitioning:INPUTS:n_future_timesteps:VALUE:{}".
#             format(n_future_timesteps))

#         entropies = np.zeros(clusters_range.size)
#         trans_matrices = []
#         invariant_dists = []

#         # For a range of candidate ``N`` numbers of clusters, we calcualte the entropy
#         for i in range(
#                 clusters_range.size
#         ):    #tqdm(range(clusters_range.size), desc="Determining maximum entropy over partitionings"):
#             n_clusters = clusters_range[i]
#             trans_matrices.append(
#                 self.get_trans_matrix(time_series, n_clusters,
#                                       n_future_timesteps))
#             invariant_dists.append(self.get_invariant_dist(trans_matrices[i]))

#             entropies[i] = self.get_partitioning_entropy(
#                 trans_matrices[i], invariant_dists[i])

#             logging.log(
#                 15, "Entropy for {} partitions determined to be {}".format(
#                     n_clusters, entropies[i]))

#         max_idx = np.argmax(entropies)

#         # We return the maximum entropy (and number of clusters) of this set.
#         return (entropies[max_idx], trans_matrices[max_idx],
#                 invariant_dists[max_idx]
#                 )    # TODO: Use the second returned argument somewhere

#     @staticmethod
#     def get_first_nondecreasing_element_idx(arr, min_idx=1):
#         # Choose the first delay for which the entropy is no longer decreasing.
#         # If there is no such element, returns the index of the last element
#         delta_arr = (arr - np.roll(arr, 1))[1:]

#         k = min_idx
#         while k < delta_arr.size:
#             if delta_arr[k] >= 0:
#                 break

#             k += 1

#         return (
#             k
#         )    # because we could only measure deltas starting at the first position in ``arr``

#     def get_optimal_n_delays(
#         self,
#         time_series,
#         delay_range=np.arange(1, 50),
#         clusters_range=np.arange(100, 2500, 100),
#         n_future_timesteps=1,
#         tol=1e-1,
#     ):
#         """
#         Determines the optimal number of delays to use in creating a delay embedding.
#         The optimal element is the first number of delays for which the maximum entropy of the ``time_series`` is no longer decreasing.
#         The maximum entropy of the ``time_series`` is the maximum entropy for all possible partitionings with sizes chosen from ``clusters_range``


#         :param time_series: (np.ndarray of shape [t, d])
#         :param delay_range: (list-like) a selection of embedding dimensions to choose from.
#         defaults to ``np.arange(1,50)``
#         :param clusters_range: see ``self.get_max_entropy_partitioning``

#         :returns optimal_n_delays: the optimal choice of delays from the preset range ``delay_range``
#         """
#         logging.debug(
#             "------------------------------------------------------------")
#         logging.debug(
#             "------------------------------------------------------------")
#         logging.info("Optimizing the number of delays ``n_delays``.")

#         logging.debug(
#             "get_optimal_n_delays:INPUTS:time_series:SHAPE:{}".format(
#                 time_series.shape))
#         logging.debug(
#             "get_optimal_n_delays:INPUTS:delay_range:SHAPE:{}".format(
#                 delay_range.shape))
#         logging.debug(
#             "get_optimal_n_delays:INPUTS:clusters_range:SHAPE:{}".format(
#                 clusters_range.shape))
#         logging.debug(
#             "get_optimal_n_delays:INPUTS:n_future_timesteps:VALUE:{}".format(
#                 n_future_timesteps))

#         n_delays = delay_range[-1]
#         trans_matrix, invariant_dist = None, None

#         entropies = np.zeros(delay_range.size)

#         for i in tqdm(range(delay_range.size),
#                       desc="Determining optimal number of delays:"):

#             n_delays = delay_range[i]

#             delay_embedded_series = self._get_delay_embedding(
#                 time_series, n_delays)

#             entropies[
#                 i], _trans_matrix, _invariant_dist = self.get_max_entropy_partitioning(
#                     delay_embedded_series,
#                     clusters_range=clusters_range,
#                     n_future_timesteps=n_future_timesteps,
#                 )

#             logging.debug(
#                 "------------------------------------------------------------")
#             logging.info(
#                 "Entropy for sequence of {} delays determined to be {}".format(
#                     n_delays, entropies[i]))
#             logging.debug(
#                 "------------------------------------------------------------")

#             if i > 0 and entropies[i] > entropies[i - 1] - tol:
#                 # we have encountered an increasing entropy and can break out of the for loop
#                 n_delays = delay_range[i - 1]
#                 break

#             trans_matrix = _trans_matrix
#             invariant_dist = _invariant_dist

#         logging.info(
#             "Optimal number of delays ``n_delay`` determined to be {}".format(
#                 n_delays))
#         return n_delays, trans_matrix, invariant_dist

#     def get_optimal_embedding_dim(
#         self,
#         delay_embedded_series,
#         embedding_dim_range=None,
#         clusters_range=np.arange(100, 1000, 100),
#         n_future_timesteps=1,
#         tol=1e-3,
#     ):
#         """
#         Assumes the optimal ``self.n_delays`` has already been determined.

#         Determines the optimal number of components to keep during the dimensional reduction step.
#         The optimal element is the first embedding dimension for which the maximum entropy of the ``time_series`` is no longer decreasing.
#         The maximum entropy of the ``time_series`` is the maximum entropy for all possible partitionings with sizes chosen from ``clusters_range`` with a given embedding dimensionality

#         :param time_series: (np.ndarray of shape [self.n_delays * d, t])
#         :param embedding_dim_range: (``None`` or list-like), a selection of embedding dimensions to choose from.
#         Defaults to None, in which case all possible numbers of dimensions are considered (between 1 and ``d``)
#         :param clusters_range: see ``self.get_max_entropy_partitioning``

#         :returns optimal_embedding_dim: (int) the optimal choice of embedding dimension from the preset range ``embedding_dim_range``
#         """
#         logging.debug(
#             "------------------------------------------------------------")
#         logging.debug(
#             "------------------------------------------------------------")

#         logging.info("Optimizing the embedding dimension ``embedding_dim``.")

#         if embedding_dim_range is None:
#             embedding_dim_range = np.arange(1, delay_embedded_series.shape[1])

#         logging.debug(
#             "get_optimal_embedding_dim:INPUTS:delay_embedded_series:SHAPE:{}".
#             format(delay_embedded_series.shape))
#         logging.debug(
#             "get_optimal_embedding_dim:INPUTS:embedding_dim_range:SHAPE:{}".
#             format(embedding_dim_range.shape))
#         logging.debug(
#             "get_optimal_embedding_dim:INPUTS:clusters_range:SHAPE:{}".format(
#                 clusters_range.shape))
#         logging.debug(
#             "get_optimal_embedding_dim:INPUTS:n_future_timesteps:VALUE:{}".
#             format(n_future_timesteps))

#         embedding_dim = embedding_dim_range[-1]
#         entropies = np.zeros(embedding_dim_range.size)

#         # Whiten the delay embedded time series
#         delay_embedded_series_whitened = svd_whiten(delay_embedded_series)

#         for i in tqdm(range(embedding_dim_range.size),
#                       desc="Determining optimal embedding dimension:"):
#             embedding_dim = embedding_dim_range[i]

#             if embedding_dim > delay_embedded_series.shape[1]:
#                 # We do not want to project to a higher dimensional space
#                 if (i > 0):
#                     embedding_dim = embedding_dim_range[i - 1]
#                 else:
#                     raise ValueError(
#                         "Insuitable embedding_dim_range provided. Must include values (in increasing order) less than {}. Value is {}"
#                         .format(delay_embedded_series.shape[1],
#                                 embedding_dim_range))
#                 break

#             # For each embedding_dim, determine the maximum entropy over all cluster sizes in ``clusters_range``
#             # for the delay series resulting from keeping only ``embedding_dim`` many components of the delay series.
#             entropies[i] = self.get_max_entropy_partitioning(
#                 delay_embedded_series_whitened[:, :embedding_dim],
#                 clusters_range=clusters_range,
#                 n_future_timesteps=n_future_timesteps,
#             )[0]

#             if i > 0 and entropies[i] > entropies[i - 1] - tol:
#                 # we have encountered an increasing entropy and can break out of the for loop
#                 embedding_dim = embedding_dim_range[i - 1]
#                 break

#             logging.info(
#                 "Entropy for sequence of embedding dimension {} determined to be {}"
#                 .format(embedding_dim, entropies[i]))

#         logging.info(
#             "Optimal embedding dimension ``embedding_dim`` determined to be {}"
#             .format(embedding_dim))

#         return embedding_dim

#     def fit_trans_matrix(
#         self,
#         time_series,
#         delay_range=np.arange(1, 50),
#         delay_clusters_range=np.arange(100, 1000, 100),
#         delay_n_future_timesteps=1,
#     ):
#         if self.n_delays is None:
#             self.n_delays, self.trans_matrix, self.invariant_dist = self.get_optimal_n_delays(
#                 time_series,
#                 delay_range=delay_range,
#                 clusters_range=delay_clusters_range,
#                 n_future_timesteps=delay_n_future_timesteps)
#         else:
#             _, self.trans_matrix, self.invariant_dist = self.get_max_entropy_partitioning(
#                 time_series,
#                 clusters_range=delay_clusters_range,
#                 n_future_timesteps=delay_n_future_timesteps)

#         self.fit_reversible_trans_matrix()
#         return self.reversible_trans_matrix

#     @staticmethod
#     def _get_delay_embedding(time_series: np.ndarray,
#                              n_delays: int) -> np.ndarray:
#         """
#         A helper method which produces a delay-embedded time series from ``time_series`` with ``n_delays``

#         :param time_series: (shape [t, d])
#         :param n_delays: number of delays to embed

#         :returns delay_embedded_series: (shape [t - n_delays, d * n_delays])

#         """

#         logging.debug(
#             "------------------------------------------------------------")
#         logging.debug(
#             "_get_delay_embedding:INPUTS:time_series:SHAPE:{}".format(
#                 time_series.shape))
#         logging.debug(
#             "_get_delay_embedding:INPUTS:n_delays:VALUE:{}".format(n_delays))

#         delay_embedding = time_series

#         if n_delays == 0:
#             return time_series

#         for i in range(1, n_delays + 1):
#             # Concatenatenate a [t, d]-shape array along its secondary axis
#             # with a delayed version of itself.
#             # (i.e. the same matrix rolled backwards along its principal axis)
#             # Then, we cut off the last row, since we have no information about the
#             # delays after our trajectory samples.

#             delay_embedding = np.concatenate(
#                 [delay_embedding, np.roll(time_series, -i, 0)], axis=1)

#         delay_embedding = delay_embedding[:delay_embedding.shape[0] -
#                                           n_delays, :]

#         logging.debug(
#             "------------------------------------------------------------")
#         logging.debug(
#             "_get_delay_embedding:OUTPUT:delay_embedding:SHAPE:{}".format(
#                 delay_embedding.shape))

#         return delay_embedding

#     def get_delay_embedding(self, time_series: np.ndarray) -> np.ndarray:
#         """
#         A wrapper for ``self._get_delay_embedding`` which uses ``self.n_delays`` (which must already be optimized/fitted) as the number of delays.
#         """
#         if self.n_delays is None:
#             raise AttributeError(
#                 "You must first fit or assign the attribute ``n_delays``.")

#         return self._get_delay_embedding(time_series, self.n_delays)

#     def _fit(
#         self,
#         time_series,
#         delay_range=np.arange(1, 50),
#         embedding_dim_range=None,
#         delay_clusters_range=np.arange(100, 1000, 100),
#         dim_clusters_range=np.arange(100, 1000, 100),
#         delay_n_future_timesteps=1,
#         dim_n_future_timesteps=1,
#         downsample=None,
#     ):
#         """
#         Helper function to be used with either ``self.fit`` or ``self.fit_transform``
#         This function always returns the transformed ``time_series``

#         :param time_series: (np.ndarray of shape [t, d])
#         :param embedding_dim_range: see ``self.get_optimal_embedding_dim``

#         :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
#         """

#         if not (downsample is None):
#             time_series = self.downsample(time_series, downsample)

#         # Step 1: Determine the optimal delay length if not already fit

#         self.fit_trans_matrix(
#             time_series,
#             delay_range=delay_range,
#             delay_clusters_range=delay_clusters_range,
#             delay_n_future_timesteps=delay_n_future_timesteps)

#         # Step 2: Create the appropriate delay embedding
#         delay_embedding = self.get_delay_embedding(time_series)

#         # Step 2: Determine the optimal embedding dimension if not already determined.
#         if self.embedding_dim is None:
#             self.embedding_dim = self.get_optimal_embedding_dim(
#                 delay_embedding,
#                 embedding_dim_range=
#                 embedding_dim_range,    # if this contains dimensions higher than the n_delays, this is automatically adjusted
#                 clusters_range=dim_clusters_range,
#                 n_future_timesteps=dim_n_future_timesteps,
#             )

#         # Step 3: Perform a PCA and transform (and whiten) the delay embedding,
#         # keeping only the self.embedding_dim many components
#         pca = PCA(self.embedding_dim, whiten=True)
#         self.pca = pca.fit(delay_embedding)
#         reconstructed_series = self.pca.transform(delay_embedding)

#         # Step 4: Return the dimensionally-reduced reconstructed embedding
#         return reconstructed_series

#     def fit(
#         self,
#         time_series,
#         delay_range=np.arange(1, 50),
#         embedding_dim_range=None,
#         delay_clusters_range=np.arange(100, 1000, 100),
#         dim_clusters_range=np.arange(100, 1000, 100),
#         delay_n_future_timesteps=1,
#         dim_n_future_timesteps=1,
#         downsample=None,
#     ):
#         """
#         Function which fits this TransferOperator object to the ``time_series``
#         and returns a reference to itself.

#         :param time_series: (np.ndarray of shape [t, d])
#         """

#         self._fit(time_series,
#                   delay_range=delay_range,
#                   embedding_dim_range=embedding_dim_range,
#                   delay_clusters_range=delay_clusters_range,
#                   dim_clusters_range=dim_clusters_range,
#                   delay_n_future_timesteps=delay_n_future_timesteps,
#                   dim_n_future_timesteps=dim_n_future_timesteps,
#                   downsample=downsample)
#         return self

#     def transform(self, time_series):
#         """
#         Assumes that this object has already been fitted using ``fit``.

#         :param time_series: (np.ndarray of shape [t, d])

#         First, embeds the ``time_series`` in a delay-embeded space of ``self.n_delays`` delays,
#         i.e. to a np.ndarray of shape [self.n_delays * d, t]

#         Then, performs dimensional reduction by performing SVD and keeping only the ``self.embedding_dim`` many components.

#         :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
#         """

#         delay_embedding = get_delay_embedding(time_series, n_delays)

#         if self.pca is None:
#             raise AttributeError(
#                 "You must first fit the transfer operator before using transform."
#             )

#         return self.pca.transform(delay_embedding)

#     def fit_transform(
#         self,
#         time_series,
#         delay_range=np.arange(1, 50),
#         embedding_dim_range=None,
#         delay_clusters_range=np.arange(100, 1000, 100),
#         dim_clusters_range=np.arange(100, 1000, 100),
#         delay_n_future_timesteps=1,
#         dim_n_future_timesteps=1,
#         downsample=None,
#     ):
#         """
#         Function which fits this TransferOperator object to the ``time_series``
#         and returns the transformed ``time_series``

#         :param time_series: (np.ndarray of shape [t, d])

#         :returns reconstructed_series: (np.ndarray of shape [t, self.embedding_dim])
#         """

#         return self._fit(time_series,
#                          delay_range=delay_range,
#                          embedding_dim_range=embedding_dim_range,
#                          delay_clusters_range=delay_clusters_range,
#                          dim_clusters_range=dim_clusters_range,
#                          delay_n_future_timesteps=delay_n_future_timesteps,
#                          dim_n_future_timesteps=dim_n_future_timesteps,
#                          downsample=downsample)


# # pytest
