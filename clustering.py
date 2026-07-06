import sklearn
import sklearn.cluster as cluster
import sklearn.manifold as manifold
import matplotlib.pyplot as plt
import numpy as np

dummy = [
    [0.0, 0.1, 0.2, 0.1, 1.5, 1.6, 1.5, 1.7, 3.0, 3.0],
    [0.1, 0.0, 0.1, 0.2, 1.6, 1.5, 1.7, 1.6, 3.0, 3.0],
    [0.2, 0.1, 0.0, 0.1, 1.5, 1.7, 1.6, 1.5, 3.0, 3.0],
    [0.1, 0.2, 0.1, 0.0, 1.7, 1.6, 1.5, 1.6, 3.0, 3.0],
    [1.5, 1.6, 1.5, 1.7, 0.0, 0.1, 0.2, 0.1, 3.0, 3.0],
    [1.6, 1.5, 1.7, 1.6, 0.1, 0.0, 0.1, 0.2, 3.0, 3.0],
    [1.5, 1.7, 1.6, 1.5, 0.2, 0.1, 0.0, 0.1, 3.0, 3.0],
    [1.7, 1.6, 1.5, 1.6, 0.1, 0.2, 0.1, 0.0, 3.0, 3.0],
    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 3.0],
    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0]
]

def cluster_result(final):
    try:
        k = int(input("tell us how many clusters you want to form, where the number of clusters has to be at least 2, and at most the number of models you are testing: "))
    except ValueError:
        k = int(input("That was not an int, try again: "))
    
    cluster_tool = cluster.AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    grid_transform = manifold.MDS(n_components=2, dissimilarity="precomputed")
    grid_data = grid_transform.fit_transform(final)
    labels = cluster_tool.fit_predict(final)
    plot = plt.scatter(grid_data[:, 0], grid_data[:, 1], c=labels)
    return plot

if __name__ == "__main__":
    result = cluster_result(dummy)
    plt.show()
