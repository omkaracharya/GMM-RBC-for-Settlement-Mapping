# I highly recommend installing/using Anaconda, saves a lot of grief
# Python 3 version

# Lets us do matrix/vector operations, MATLAB-style
import numpy as np

# For computations involving multivariate normal (Gaussian) distributions
# Makes sense since this is a Gaussian Mixture Model, eh?
from scipy.stats import multivariate_normal

# This will be used to initialize the component responsibilites
# (more on this later on)
from sklearn.cluster import KMeans

# This is for dataframes, like the ones in R
import pandas as pd


def log_likelihood(attribute_matrix, weights, means, cov_matrices):
    log_likelihood_value = 0
    
    num_observations = attribute_matrix.shape[0]
    
    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians= []
    for c in range(0, len(weights)):
        gaussians.append(multivariate_normal(mean=means[c,:], cov=cov_matrices[c]))
    
    # Iterate through the attribute_matrix by row (i.e. by observation)
    for i in range(0, num_observations):
        
        current_observation = attribute_matrix[i,:]
        pre_log_sum = 0
        
        # Iterate through each of the clusters
        for j in range(0, len(weights)):
            pre_log_sum += weights[j]*gaussians[j].pdf(current_observation)
        log_likelihood_value += np.log(pre_log_sum)
        
    return log_likelihood_value



def assign_responsibilities(attribute_matrix, weights, means, cov_matrices):
    
    num_observations = attribute_matrix.shape[0]
    num_clusters = len(weights)
    
    # This is the component responsibility matrix that will be returned.
    # For now, fill it up with all zeros.
    component_resp_matrix = np.zeros((num_observations, num_clusters))
    
    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians= []
    for c in range(0, len(weights)):
        gaussians.append(multivariate_normal(mean=means[c,:], cov=cov_matrices[c]))
        
    # Iterate through the attribute_matrix by row (i.e. by observation)
    for i in range(0, num_observations):
        
        current_observation = attribute_matrix[i,:]
        
        # Will have to divide by this later on to ensure that the responsibilities sum to 1
        normalization_factor = 0. 
        
        # Iterate through each of the clusters
        for j in range(0, len(weights)):
            component_resp_matrix[i,j] = weights[j]*gaussians[j].pdf(current_observation)
            normalization_factor += component_resp_matrix[i,j]
            
        component_resp_matrix[i,:] /= normalization_factor 
        
    return component_resp_matrix


def get_soft_counts(component_resp_matrix):
    return np.sum(component_resp_matrix, axis=0)


def compute_comp_weights(component_resp_matrix):
    soft_counts = get_soft_counts(component_resp_matrix)
    num_obs = component_resp_matrix.shape[0]
    return soft_counts/float(num_obs)


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
            comp_means[comp,:] += component_resp_matrix[i,comp]*attribute_matrix[i,:]
            
        # Divide off by the component's soft count to get the mean coordinates
        comp_means[comp,:] /= float(soft_counts[comp])
    
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
            diff_vector = attribute_matrix[i,:] - comp_means[c,:]
            outer_product = np.outer(diff_vector, diff_vector)
            new_cov_matrix += (component_resp_matrix[i, c]*outer_product)
            
        # Divide off by the soft count
        new_cov_matrix /= soft_counts[c]
        
        # Replace it in list of component covariance matrices
        comp_cov_matrices[c] = new_cov_matrix
                         
# Debugging code, commented out for now, checks the M step
'''
data_tmp = np.array([[1.,2.],[-1.,-2.]])
covariances = [np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])]

resp = assign_responsibilities(attribute_matrix=data_tmp, weights=np.array([0.3, 0.7]),
                                means=np.array([[0.,0.], [1.,1.]]),
                                cov_matrices=covariances)
counts = get_soft_counts(resp)
means = compute_comp_means(resp, data_tmp)
compute_cov_matrices(resp, data_tmp, means, covariances)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and \
    np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print('Checkpoint passed!')
else:
    print('Check your code again.')
'''

# Inputs:  attribute matrix, number of components, epsilon for convergence (if not specified, default is 1e-4), 
# maximum number of iterations in case there are issues with convergence

# Return:  the final component responsibilities

def do_expectation_maximization(attribute_matrix, num_components, epsilon = 1e-4, max_iterations=9999):
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
        
        # Do the E-step
        comp_resp_matrix = assign_responsibilities(attribute_matrix, comp_weights, comp_means, cov_matrices)
        
        # Do the M-step
        comp_weights = compute_comp_weights(comp_resp_matrix)
        comp_means = compute_comp_means(comp_resp_matrix, attribute_matrix)
        compute_cov_matrices(comp_resp_matrix, attribute_matrix, comp_means, cov_matrices)
        
        # Compare to the previous log-likelihood. If the increase is smaller than epsilon,
        # then we have converged and terminate the expectation maximization
        log_likelihood_next = log_likelihood(attribute_matrix, comp_weights, comp_means, cov_matrices)
        if abs(log_likelihood_next-log_likelihood_previous) < epsilon:
            break
        else:
            log_likelihood_previous = log_likelihood_next
        
    # Return the final component responsibility matrix
    return comp_resp_matrix


# Working with the iris data, make sure it's in the right directory
# and that the column names are correct
# Highly recommended: try working with a messier, more complex dataset
# For now, this is commented out
'''
iris_data = pd.read_csv("iris.csv")
attribute_matrix = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
comp_resp_matrix = do_expectation_maximization(attribute_matrix, 3)
results = pd.DataFrame(comp_resp_matrix, columns=['resp(Versicolor)', 'resp(Setosa)', 'resp(Virginica)'])
results['Actual'] = iris_data['class']
pd.set_option('display.max_rows', 150)
print(results)
'''