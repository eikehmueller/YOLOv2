"""Module for generating optimal anchor boxes for YOLO2 classifier"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def distanceEuclidean(p, q):
    """Compute (vectorised) Euclidean distance

    Returns an array of length n_data with the Euclidean distances of p_j to q

    :arg p: array with vectors of size d (=> shape (n_data,d))
    :arg q: vector of size d (=> shape (d,))
    """
    return np.linalg.norm(p[:, :] - q[:], axis=-1)


def distanceIoU(p, q):
    """Compute (vectorized) IoU distrance.

    p_{j,0}=:w_j, p_{j,1}=:h_j are the width and height of the j-th bounding box, and
    q_0=:w, q_1=:h are the width and height of the bounding box we compare to

    For each j this computes the IoU as:

    A_{intersect} = min(w_j,w)*min(h_j,h)
    A_{union} = w_j*h_j + w*h - A_{intersect}

    IoU = A_{intersect} / A_{union}

    :arg p: array with vectors of size d (=> shape (n_data,d))
    :arg q: vector of size d (=> shape (d,))
    """
    Aq = q[0] * q[1]
    Aintersect = np.minimum(p[:, 0], q[0]) * np.minimum(p[:, 1], q[1])
    Aunion = p[:, 0] * p[:, 1] + Aq - Aintersect
    return 1.0 - Aintersect / Aunion


class KMeans(object):
    """Class for k-mean clustering with custom distance function

    The provided vectorised distance function has to be take two arguments p and q,
    where p is of shape (n,dim) and q is a vector of length dim. The function should then
    return a vector of length n, such that the j-th entry is the distance of p_j to q.

    Centroids are initialised with the random partitioning method. If after the data
    assignment step no data is assigned to a particular centroid, a new centroid will
    be choosen randomly from all available points.
    """

    def __init__(self, f_dist, n_centroid=4):
        """Construct new instance

        :arg f_fdist: function for computing distance
        :arg n_centroid: number of centroids
        """
        self.f_dist = f_dist
        self.n_centroid = n_centroid

    def cluster(self, data, maxiter=100):
        """Cluster data given in array of shape (n_data, dim)

        Returns:
         * list of assignments of all points to their nearest centroid
         * a list of centroids of the form as an array of shape (n_centroid, dim)

        :arg data: array with data
        :arg maxiter: maximal number of iterations
        """
        n_data, dim = data.shape
        # Centroids in d-dimensional space
        centroids = np.zeros((self.n_centroid, dim))
        centroid_old = np.zeros((self.n_centroid, dim))
        # Assignments to centroids, initialise with random partitioning
        assignments = np.random.randint(low=0, high=self.n_centroid, size=n_data)
        # Distances to centroids
        distances = np.zeros((self.n_centroid, n_data))
        tolerance = 1.0e-12
        converged = False
        for k_iter in range(maxiter):
            # Step 1: compute centroids
            for j in range(self.n_centroid):
                if len(data[assignments == j]) == 0:
                    centroids[j, :] = data[np.random.randint(low=0, high=n_data), :]
                else:
                    centroids[j, :] = np.average(data[assignments == j], axis=0)
            # Step 2: adjust assignments
            for j in range(self.n_centroid):
                distances[j, :] = self.f_dist(data, centroids[j, :])
            assignments = np.argmin(distances, axis=0)
            d_move = np.linalg.norm((centroids - centroid_old).flatten())
            if d_move < tolerance:
                converged = True
                break
            centroid_old[:, :] = centroids[:, :]
        if converged:
            print(f"Lloyds algorithm converged in {k_iter+1} iterations")
        else:
            print(
                f"WARNING: Lloyds algorithm failed to converge in {maxiter} iterations"
            )
        return assignments, centroids


class BestAnchorBoxFinder(object):
    """Find the best anchor boxes for a given set of bounding boxes

    Given n_bbox bounding boxes, as a list of dictionaries of the form
    [{"width":xx,"height"},...] and a list of n_anchor anchor boxes in
    the same format, this returns a list of k=min{n_anchor,n_bbox} anchor box indices
    and k bounding box indices such that the overlap between bounding boxes and
    anchor boxes is optimised.

    To achieve this, the Hungarian algorithm is run with the cost matrix

    A_{j,k} = 1-IoU(anchor_box(j), bounding_box(k))
    """

    def __init__(self, anchors):
        """Construct new instance

        :arg anchors: list of n_anchor anchor boxes in the format[{"width":xx,"height"},...]
        """
        self.anchors = anchors

    def match(self, bboxes):
        """Find the optimal assignment between given bounding boxes and anchors

        Returns list of anchor box indices and bounding box indices.

        :arg bbox: list of n_bbox bounding boxes in the format[{"width":xx,"height"},...]
        """
        # construct cost matrix
        cost_matrix = np.zeros((len(self.anchors), len(bboxes)))
        for j, anchor in enumerate(self.anchors):
            for k, bbox in enumerate(bboxes):
                width_a, height_a = anchor["width"], anchor["height"]
                width, height = bbox["width"], bbox["height"]
                Aintersect = min(width, width_a) * min(height, height_a)
                Aunion = width_a * height_a + width * height - Aintersect
                cost_matrix[j, k] = 1 - Aintersect / Aunion
        # use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind


def generate_2d_data(n_centroid, n_data, width=0.1):
    """Generate synthetic two dimensional data for testing k-means clustering.

    Samples n_data 2d points by drawing from n_centroid normal distributions with a
    specified width. The same number of samples is drawn from each normal distribution and the
    centres of the distributions are uniformly distributions in [0,1] x [0,1].
    Returns two arrays:
     * an array of shape (n_data,2) containing the data
     * an array of length n_data, containing assignments to the different normal distributions

    :arg n_centroid: number of normal distributions to sample from
    :arg n-data: total number of data points to generate.
    """
    centroid = np.random.uniform(low=0, high=1, size=(n_centroid, 2))
    data = np.zeros((n_data, 2))
    assignments = np.zeros(n_data, dtype=int)
    for j in range(n_centroid):
        j_min = j * n_data // n_centroid
        j_max = min((j + 1) * n_data // n_centroid, n_data)
        data[j_min:j_max, :] = np.random.normal(
            loc=centroid[j, :], scale=[width, width], size=(j_max - j_min, 2)
        )
        assignments[j_min:j_max] = j
    return data, assignments


def plot_kmeans_data(data, assignments, centroids, label_x="x", label_y="y"):
    """Plot k-means data together with assignments

    Visualises the data, coloured by centroid assignments and the centroids
    as black crosses.

    :arg data:array of shape (n_data, 2) containing the data points
    :arg assignments: array of length n_data, assignment of data to the centroids
    :arg centroids: array of shape (n_centroid, 2), containing the centrods
    :arg label_x: label for horizontal axis
    :arg label_y: label for vertical axis
    """
    plt.figure(figsize=(10, 10))
    for j in range(max(assignments) + 1):
        masked_data = data[assignments == j]
        X = masked_data[:, 0]
        Y = masked_data[:, 1]
        plt.scatter(X, Y, marker=".")
    for centroid in centroids:
        plt.plot(
            centroid[0],
            centroid[1],
            linewidth=0,
            marker="x",
            markersize=16,
            markeredgewidth=2,
            color="black",
        )
    ax = plt.gca()
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    plt.show()
