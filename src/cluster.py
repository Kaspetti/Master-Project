from numpy.typing import NDArray
from kneed import KneeLocator
from sklearn.cluster import KMeans


def cluster_centroids(centroids: NDArray, min_k: int, max_k: int):
    inertias: list[float]= [KMeans(n_clusters=k, random_state=1, n_init="auto").fit(centroids).inertia_ for k in range(min_k, max_k)]   # type: ignore
    kneedle = KneeLocator(range(min_k, max_k), inertias, S=1.0, curve="convex", direction="decreasing")

    kmeans = KMeans(n_clusters=kneedle.elbow, random_state=0, n_init="auto").fit(centroids) # type: ignore

    return kmeans.labels_
