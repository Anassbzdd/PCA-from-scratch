import numpy as np

def pca_from_scratch(X, n_components = 2):
    # Center the data
    X_mean = X - np.mean(X,axis = 0 )

    # Covariance matrix
    cov_matrix = np.cov( X_mean , rowvar = False )

    # Eigen values / vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_matrix)

    # Sort Eigen 
    sorted_id = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_id]
    eigen_vectors = eigen_vectors[:,sorted_id]

    # Select top
    eigen_vectors = eigen_vectors[:, :n_components]

    #Project 
    X_reduced = np.dot(X_mean , eigen_vectors)

    return X_reduced , eigen_values , eigen_vectors

X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

X_reduced , eigen_val , eigen_vec = pca_from_scratch(X, 1)

print("Reduced data :\n", X_reduced )
