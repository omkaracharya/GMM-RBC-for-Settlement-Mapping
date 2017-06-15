# This is for Python 2, not sure if it works for Python 3
# I highly recommend installing/using Anaconda, saves a lot of grief

# Lets us do matrix/vector operations, MATLAB-style
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


# For computations involving multivariate normal (Gaussian) distributions
# Makes sense since this is a Gaussian Mixture Model, eh?
# from sklearn.metrics import accuracy_score

# This will be used to initialize the component responsibilites
# (more on this later on)


def log_likelihood(attribute_matrix, weights, means, cov_matrices):
    log_likelihood_value = 0
    num_observations = attribute_matrix.shape[0]

    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians = []
    for c in range(0, len(weights)):
        gaussians.append(multivariate_normal(mean=means[c, :], cov=cov_matrices[c]))

    # Iterate through the attribute_matrix by row (i.e. by observation)
    for i in range(0, num_observations):
        current_observation = attribute_matrix[i, :]
        pre_log_sum = 0

        # Iterate through each of the clusters
        for j in range(0, len(weights)):
            pre_log_sum += weights[j] * gaussians[j].pdf(current_observation)
        log_likelihood_value += np.log(pre_log_sum)

    return log_likelihood_value


def assign_responsibilities(attribute_matrix, weights, means, cov_matrices):
    num_observations = attribute_matrix.shape[0]
    num_clusters = len(weights)

    # This is the component responsibility matrix that will be returned.
    # For now, fill it up with all zeros.
    component_resp_matrix = np.zeros((num_observations, num_clusters))

    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians = []
    for c in range(0, len(weights)):
        gaussians.append(multivariate_normal(mean=means[c, :], cov=cov_matrices[c]))

    # Iterate through the attribute_matrix by row (i.e. by observation)
    for i in range(0, num_observations):
        current_observation = attribute_matrix[i, :]

        # Will have to divide by this later on to ensure that the responsibilities sum to 1
        normalization_factor = 0.

        # Iterate through each of the clusters
        for j in range(0, len(weights)):
            component_resp_matrix[i, j] = weights[j] * gaussians[j].pdf(current_observation)
            normalization_factor += component_resp_matrix[i, j]
        component_resp_matrix[i, :] /= normalization_factor

    return component_resp_matrix, gaussians


def get_soft_counts(component_resp_matrix):
    return np.sum(component_resp_matrix, axis=0)


def compute_comp_weights(component_resp_matrix):
    soft_counts = get_soft_counts(component_resp_matrix)
    num_obs = component_resp_matrix.shape[0]
    return soft_counts / float(num_obs)


def compute_comp_means(component_resp_matrix, attribute_matrix):
    soft_counts = get_soft_counts(component_resp_matrix)
    num_components = len(soft_counts)
    num_obs = attribute_matrix.shape[0]
    num_attributes = attribute_matrix.shape[1]

    # This is what will be returned
    comp_means = np.zeros((num_components, num_attributes))

    # Loop through each of the components
    for comp in range(0, num_components):

        # Loop through each of the data points
        for i in range(0, num_obs):
            # Get running sum for points' coordinates, where each point is weighted by responsibility
            # that current component has for it
            comp_means[comp, :] += component_resp_matrix[i, comp] * attribute_matrix[i, :]

        # Divide off by the component's soft count to get the mean coordinates
        comp_means[comp, :] /= float(soft_counts[comp])

    return comp_means


# Do what the formula says!

# Note that nothing is returned here; we update the list of
# component covariance matrices in-place
def compute_cov_matrices(component_resp_matrix, attribute_matrix, comp_means, comp_cov_matrices):
    soft_counts = get_soft_counts(component_resp_matrix)
    num_components = len(soft_counts)
    num_obs = attribute_matrix.shape[0]
    num_attributes = attribute_matrix.shape[1]

    # Iterate through each component's covariance matrix
    for c in range(0, num_components):

        # This will be the new, updated component covariance matrix
        new_cov_matrix = np.zeros((num_attributes, num_attributes))

        # Iterate through each of the observations in the attribute matrix
        for i in range(0, num_obs):
            # diff_vector is: (observation i vector) - (cluster mean vector)
            diff_vector = attribute_matrix[i, :] - comp_means[c, :]
            outer_product = np.outer(diff_vector, diff_vector)
            new_cov_matrix += (component_resp_matrix[i, c] * outer_product)

        # Divide off by the soft count
        new_cov_matrix /= soft_counts[c]

        # Replace it in list of component covariance matrices
        comp_cov_matrices[c] = new_cov_matrix


def do_expectation_maximization(attribute_matrix, num_components, epsilon=1e-4, max_iterations=4):
    """
        # Inputs:  attribute matrix, number of components, epsilon for convergence (if not specified, default is 1e-4),
        # maximum number of iterations in case there are issues with convergence

        # Return:  the final component responsibilities
    """

    # If attribute_matrix is df, convert to numpy matrix
    attribute_matrix = pd.DataFrame.as_matrix(attribute_matrix)

    # Some useful values
    num_obs = attribute_matrix.shape[0]
    num_attributes = attribute_matrix.shape[1]

    # First, initialize the component responsibilities using k-means algorithm
    kmeans = KMeans(n_clusters=num_components, random_state=0).fit(attribute_matrix)
    kmeans_cluster_assignments = kmeans.labels_
    comp_resp_matrix = np.zeros((num_obs, num_components))

    for i in range(0, num_obs):
        comp_resp_matrix[i, kmeans_cluster_assignments[i]] = 1.

    # OK, so we have the initial component responsibilities...so can perform the first M-step
    # We also could have initialized the Gaussian Mixture Model parameters and performed the E-step first BUT
    # the other method seems more straightfoward.
    comp_weights = compute_comp_weights(comp_resp_matrix)
    comp_means = compute_comp_means(comp_resp_matrix, attribute_matrix)
    cov_matrices = []

    for c in range(0, num_components):
        cov_matrices.append(np.zeros((num_attributes, num_attributes)))
    compute_cov_matrices(comp_resp_matrix, attribute_matrix, comp_means, cov_matrices)

    # Get the initial log-likelihood
    log_likelihood_previous = log_likelihood(attribute_matrix, comp_weights, comp_means, cov_matrices)

    # Now enter main loop bouncing back and force between the E-step and M-step until convergence
    for iter in range(0, max_iterations):

        print "EM Iteration:", iter

        # Do the E-step
        comp_resp_matrix, gaussians = assign_responsibilities(attribute_matrix, comp_weights, comp_means, cov_matrices)

        # Do the M-step
        comp_weights = compute_comp_weights(comp_resp_matrix)
        comp_means = compute_comp_means(comp_resp_matrix, attribute_matrix)
        compute_cov_matrices(comp_resp_matrix, attribute_matrix, comp_means, cov_matrices)

        # Compare to the previous log-likelihood. If the increase is smaller than epsilon,
        # then we have converged and terminate the expectation maximization
        log_likelihood_next = log_likelihood(attribute_matrix, comp_weights, comp_means, cov_matrices)

        if abs(log_likelihood_next - log_likelihood_previous) < epsilon:
            break
        else:
            log_likelihood_previous = log_likelihood_next

    # Return the final component responsibility matrix
    return comp_resp_matrix, comp_means, comp_weights, cov_matrices


def load_image(path):
    img = Image.open(path)
    pix = img.load()
    row = img.size[0]
    col = img.size[1]

    counter = 0
    data = np.zeros([row * col, 3])

    for c in range(col):
        for r in range(row):
            if path.split('.')[1] == 'png':
                data[counter] = list(pix[r, c])[:-1]
            elif path.split('.')[1] == 'jpg':
                data[counter] = list(pix[r, c])
            counter += 1

    return pd.DataFrame(data=data), row, col


def train_model(stored_covariance, stored_means, stored_Z, train_image_path):
    train_data, row, col = load_image(train_image_path)

    comp_resp_matrix, comp_means, comp_weights, cov_matrices = do_expectation_maximization(train_data, 6)
    # results = pd.DataFrame(comp_resp_matrix, columns=['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6'])

    Z = get_soft_counts(comp_resp_matrix)
    pd_Z = pd.DataFrame(Z)
    pd_Z.to_pickle(stored_Z)

    pd_means = pd.DataFrame(comp_means)
    pd_means.to_pickle(stored_means)

    pd_covs = pd.Panel(cov_matrices)
    pd_covs.to_pickle(stored_covariance)


def get_components(test_image_path):
    models_home = 'models/'
    stored_covariance = models_home + 'covs.pkl'
    stored_means = models_home + 'means.pkl'
    stored_Z = models_home + 'Z.pkl'

    Z = pd.read_pickle(stored_Z)
    means = pd.read_pickle(stored_means)
    cov_matrices = pd.read_pickle(stored_covariance)

    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians = []
    for c in range(0, len(Z)):
        gaussians.append(multivariate_normal(mean=means[c:c + 1].values[0], cov=cov_matrices.values[c]))

    test_data, row, col = load_image(test_image_path)

    components = []

    for pixel in test_data.values:
        arg_max = 0
        max_c = -1
        for c in range(len(Z)):
            ans = Z.values[c] * gaussians[c].pdf(np.array(pixel))
            if ans > arg_max:
                max_c = c
                arg_max = ans
        components.append(max_c)

    colors = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 255, 255), (0, 255, 0), (0, 0, 0)]
    # output = [tuple(np.asarray(means[c:c+1].values[0]).astype(int)) for c in components]
    output = [colors[c] for c in components]
    img2 = Image.new('RGB', [row, col])
    img2.putdata(output)
    test_image_name, test_image_extension = test_image_path.split('.')
    test_image_name = test_image_name.split('/')[1]
    img2.save('output/' + test_image_name + "_output." + test_image_extension)

    return components


def main():
    models_home = 'models/'
    images_home = 'images/'
    stored_covariance = models_home + 'covs.pkl'
    stored_means = models_home + 'means.pkl'
    stored_Z = models_home + 'Z.pkl'
    if not os.path.exists(stored_covariance) or not os.path.exists(stored_means) or not os.path.exists(
            stored_Z):
        print "Model training started.."
        train_image_path = images_home + 'city1.png'
        train_model(stored_covariance, stored_means, stored_Z, train_image_path)
        print "Model training completed.."
    print "Model loaded.."

    # test_image_path = images_home + 'city2.jpg'
    # test_model(stored_covariance, stored_means, stored_Z, test_image_path)
    # get_components(test_image_path=images_home+'city2.jpg')


if __name__ == '__main__':
    main()
