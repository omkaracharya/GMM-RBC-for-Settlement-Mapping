# This is for Python 2, not sure if it works for Python 3
# I highly recommend installing/using Anaconda, saves a lot of grief

# Lets us do matrix/vector operations, MATLAB-style
import sys
import time
import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

models_home = 'models/'
images_home = 'images/'
inputs_home = 'inputs/'
n_components = 6
header = [['Vegetation', 'Concrete', 'Ground', 'Buildings', 'Misc', 'Water', 'Class']]
PYTHON_VERSION = sys.version_info[0]


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
    for iteration in range(0, max_iterations):

        print("EM Iteration:", iteration)

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
    row = img.size[1]
    col = img.size[0]

    counter = 0
    data = np.zeros([row * col, 3])

    for r in range(row):
        for c in range(col):
            if path.split('.')[1] == 'png' or path.split('.')[1] == 'tif':
                data[counter] = list(pix[c, r])[:-1]
            elif path.split('.')[1] == 'jpg':
                data[counter] = list(pix[c, r])
            counter += 1

    return pd.DataFrame(data=data), row, col


def train_model(stored_covariance, stored_means, stored_Z, train_image_path):
    train_data, row, col = load_image(train_image_path)

    comp_resp_matrix, comp_means, comp_weights, cov_matrices = do_expectation_maximization(train_data, n_components)
    # results = pd.DataFrame(comp_resp_matrix, columns=['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6'])

    Z = get_soft_counts(comp_resp_matrix)
    pd_Z = pd.DataFrame(Z)
    pd_Z.to_pickle(stored_Z)

    pd_means = pd.DataFrame(comp_means)
    pd_means.to_pickle(stored_means)

    pd_covs = pd.Panel(cov_matrices)
    pd_covs.to_pickle(stored_covariance)


def get_model():
    stored_covariance = models_home + 'covs.pkl'
    stored_means = models_home + 'means.pkl'
    stored_Z = models_home + 'Z.pkl'

    if not os.path.exists(stored_covariance) or not os.path.exists(stored_means) or not os.path.exists(stored_Z):
        print("Gaussian Mixture Model training started..")
        train_image_path = images_home + 'city.png'
        train_model(stored_covariance, stored_means, stored_Z, train_image_path)
        print("Gaussian Mixture Model training completed..")
    print("Gaussian Mixture Model loaded..")

    cov_matrices = pd.read_pickle(stored_covariance)
    means = pd.read_pickle(stored_means)
    Z = pd.read_pickle(stored_Z)

    return Z, means, cov_matrices


# Function to generate image using output from GMM
def generate_gmm_image(test_image_path, components, col, row):
    # 0: Vegetation - Green
    # 1: Concrete/Road - Red
    # 2: Ground - Yellow
    # 3: Buildings - White
    # 4: Misc - Black
    # 5: Water - Blue

    colors = [(0, 255, 0), (255, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 0), (0, 0, 255)]

    # output = [tuple(np.asarray(means[c:c+1].values[0]).astype(int)) for c in components]
    output = [colors[c] for c in components]
    img2 = Image.new('RGB', [col, row])
    img2.putdata(output)
    test_image_name, test_image_extension = test_image_path.split('.')
    test_image_name = test_image_name.split('/')[-1]
    output_file_name = 'output/' + test_image_name + "_output." + test_image_extension
    img2.save(output_file_name)
    print("Genereted output image location: " + output_file_name + " (" + str(col) + "x" + str(row) + ")")


# Function to generate image using output from RBC
def generate_rbc_image(test_image_path, category, col, row):
    # 0: Commercial - Red
    # 1: Residential Type 1 - White
    # 2: Residential Type 2 - Black
    # 3: Vegetation - Green
    # 4: Water - Blue
    # 5: Misc - Yellow


    colors = pd.DataFrame(list(range(6)))
    colors.index = ['Commercial', 'Residential Type1', 'Residential Type2', 'Vegetation', 'Water', 'Misc']
    category_index = np.asarray(colors.loc[category:])[0]

    components = np.full((row * col), category_index, dtype=int)
    colors = [(255, 0, 0), (255, 255, 255), (0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    output = [colors[c] for c in components]
    img2 = Image.new('RGB', [col, row])
    img2.putdata(output)
    test_image_name, test_image_extension = test_image_path.split('.')
    test_image_name = test_image_name.split('/')[-1]
    output_file_name = 'output/' + test_image_name + "_output." + test_image_extension
    img2.save(output_file_name)
    print("Genereted output image location: " + output_file_name + " (" + str(col) + "x" + str(row) + ")")


def get_components(Z, means, cov_matrices, test_image_path, test_data, row, col):
    # Get the multivariate Gaussian distribution for each of the clusters
    gaussians = []
    for c in range(0, len(Z)):
        gaussians.append(multivariate_normal(mean=means[c:c + 1].values[0], cov=cov_matrices.values[c]))

    # test_data, row, col = load_image(test_image_path)

    components = []

    for pixel in test_data:
        arg_max = 0
        max_c = -1
        for c in range(len(Z)):
            ans = Z.values[c] * gaussians[c].pdf(np.array(pixel))
            if ans > arg_max:
                max_c = c
                arg_max = ans
        components.append(max_c)

    # print(components)
    generate_gmm_image(test_image_path, components, col, row)

    return components


# Function to get number of pixels in each cluster
def get_feature_frequencies(feature_classification):
    feature_frequencies = [0] * n_components
    for i in range(len(feature_frequencies)):
        feature_frequencies[i] = feature_classification.count(i)
    return feature_frequencies


# Function to get feature weight using output of function get_feature_frequencies()
def get_feature_weights(feature_frequencies):
    feature_weights = [0] * len(feature_frequencies)
    sum_frequencies = sum(feature_frequencies)
    for i in range(len(feature_weights)):
        feature_weights[i] = float(feature_frequencies[i]) / sum_frequencies
    return feature_weights


# Function applying 1-Holte Rule on a clustered image
# Variation to be specified as float (e.g 0.1 for 10%)
def rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels, variation):
    feature_frequencies = get_feature_frequencies(feature_classification)
    feature_weights = get_feature_weights(feature_frequencies)
    category_matrix = np.zeros(shape=labelled_patch_category_weights.shape)

    for i in range(labelled_patch_category_weights.shape[0]):
        for j in range(len(feature_weights)):
            category_matrix[i, j] = 1.0 / (1 + abs(labelled_patch_category_weights[i][j] - feature_weights[j]))

    category_max_row_index = np.sum(category_matrix, 1).argmax()
    return labels[category_max_row_index], category_max_row_index


# Function applying majority vote on a list of subimages to classify an image as Urban or Rural
def settlement_type(image_settlement_mapping_list):
    urban_count = 0
    rural_count = 0

    for i in range(len(image_settlement_mapping_list)):
        if image_settlement_mapping_list[[i]]["Urban"] == 1:
            urban_count += 1
        else:
            rural_count += 1

        if urban_count == rural_count:
            return "Tie"
        if urban_count < rural_count:
            return "Rural"
        else:
            return "Urban"


# Function to convert a 2d matrix to 3d matrix with num_dim_3d dimensions along 3rd axis
def matrix2dTo3d(matrix_2d, num_rows_3d, num_cols_3d, num_dim_3d):
    matrix_3d = np.zeros(shape=(num_rows_3d, num_cols_3d, num_dim_3d))

    for i in range(matrix_2d.shape[0]):
        if i == 0:
            j = 0
            k = 0
        else:
            j = int(np.ceil(float(i) / num_cols_3d)) - 1
            k = i % num_cols_3d
        matrix_3d[j, k, :] = matrix_2d[i, :]

    return matrix_3d


# Function to convert a 3d matrix to 2d matrix
# The data along 3rd axis is layed out as columns in the 2d matrix
def matrix3dTo2d(matrix_3d):
    matrix_2d = np.zeros(shape=(matrix_3d.shape[0] * matrix_3d.shape[1], matrix_3d.shape[2]))
    k = 0

    for i in range(matrix_3d.shape[0]):
        for j in range(matrix_3d.shape[1]):
            if k < matrix_3d.shape[0] * matrix_3d.shape[1]:
                matrix_2d[k, :] = matrix_3d[i, j, :]
                k += 1

    # matrix_2d = matrix_2d[matrix_2d[,6]!=0,] # Remove 0 rows
    return matrix_2d


# Function to extract a sub matrix from a 3d matrix
def extractSub3dMatrix(matrix_3d, x_min, y_min, x_max, y_max, num_dim_3d):
    num_row = y_max - y_min
    num_col = x_max - x_min

    sub_matrix_3d = np.zeros(shape=(num_row, num_col, num_dim_3d))

    k = 0
    l = 0
    for ii in range(0, matrix_3d.shape[0], num_row):
        for jj in range(0, matrix_3d.shape[1], num_col):
            for i in range(ii, ii + num_row, 1):
                for j in range(jj, jj + num_col, 1):
                    if i <= matrix_3d.shape[0] and j <= matrix_3d.shape[1]:
                        sub_matrix_3d[k, l, :] = matrix_3d[i, j, :]
                        l = l + 1
                k = k + 1
                l = 0

    return sub_matrix_3d


# Function to split a 3D matrix into (row_size x col_size) sub matrices
def split3dMatrix_by_size(matrix_3d, row_size, col_size, num_dim_3d):
    m_list = []
    num_row = int(np.ceil(float(matrix_3d.shape[0]) / row_size))
    num_col = int(np.ceil(float(matrix_3d.shape[1]) / col_size))
    num_matrices = num_row * num_col

    for i in range(num_matrices):
        m_list.append(np.zeros(shape=(row_size, col_size, num_dim_3d)))

    k = 0
    l = 0
    matrix_index = 0

    for jj in range(0, matrix_3d.shape[1], col_size):
        for ii in range(0, matrix_3d.shape[0], row_size):
            for i in range(ii, ii + row_size, 1):
                for j in range(jj, jj + col_size, 1):
                    if i <= matrix_3d.shape[0] - 1 and j <= matrix_3d.shape[1] - 1:
                        # print(matrix_3d.shape[0],matrix_3d.shape[1],num_row, num_col, ii, jj, i, j, k, l, matrix_index)
                        m_list[matrix_index][k, l,] = matrix_3d[i, j,]
                        l += 1
                k += 1
                l = 0
            k = 0
            matrix_index += 1

    return m_list


# Function to check if RBC is trained, and if not to train RBC
def check_if_RBC_trained(input_file, Z, means, cov_matrices):
    if not os.path.exists(input_file):
        # RBC training
        labelled_patches = header[:]
        unique = {}
        for image in os.listdir(images_home + 'training/'):
            input_image_path = images_home + 'training/' + image
            # Convert Input image to points in 2d
            points_input_image, nrows, ncols = load_image(input_image_path)
            print("Training RBC on", input_image_path, "with", nrows, "rows and", ncols, "columns")
            feature_classification = get_components(Z, means, cov_matrices, input_image_path,
                                                    np.asarray(points_input_image), nrows, ncols)
            frequencies = get_feature_frequencies(feature_classification)
            weights = get_feature_weights(frequencies)
            if image.split('_')[0] in unique:
                for i in range(len(weights)):
                    unique[image.split('_')[0]][0][i] += weights[i]
                unique[image.split('_')[0]][1] += 1
                # print unique

            else:
                unique[image.split('_')[0]] = [weights[:], 1]
                # weights.append(image.split('_')[0])
                # labelled_patches.append(weights)
        for k, v in unique.items():
            tempweights = v[0][:]
            for i in range(len(tempweights)):
                tempweights[i] /= v[1]
            tempweights.append(k)
            labelled_patches.append(tempweights)
        # Store the feature weights into CSV file
        # print labelled_patches
        if PYTHON_VERSION == 3:
            with open(input_file, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(labelled_patches)
        elif PYTHON_VERSION == 2:
            with open(input_file, 'wb') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(labelled_patches)
        print("\n\n")


# Function to get trained RBC labels and corresponding weights
def get_trained_RBC_info(input_file, labels, labelled_patch_category_weights):
    # Read weights of pre-trained RBC
    if PYTHON_VERSION == 3:
        with open(input_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                labels.append(row[-1])
                labelled_patch_category_weights.append(row[:-1])
    elif PYTHON_VERSION == 2:
        with open(input_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            csvreader.next()
            for row in csvreader:
                labels.append(row[-1])
                labelled_patch_category_weights.append(row[:-1])


# Function to run GMM+RBC on a test directory
def run_test(test_images_path, nbands, Z, means, cov_matrices, labels, labelled_patch_category_weights):
    labelled_patches = header[:]
    for image in os.listdir(test_images_path):

        input_image_path = test_images_path + image

        # Convert Input image to points in 2d
        points_input_image, nrows, ncols = load_image(input_image_path)

        # Convert 2d points to 3d (easier to visualize and extract)
        points_input_image_3d = matrix2dTo3d(np.asarray(points_input_image), nrows, ncols, nbands)

        # Extract the sub image to classify from the 3d Matrix containing image information
        points_sub_image_3d = extractSub3dMatrix(points_input_image_3d, 0, 0, ncols, nrows, nbands)

        # Convert the extracted 3d sub image to 2d as required by the in-built GMM library
        points_sub_image_2d = matrix3dTo2d(points_sub_image_3d)

        # Cluster the input image using GMM
        print("Testing GMM+RBC on", input_image_path, "with", nrows, "rows and", ncols, "columns")
        feature_classification = get_components(Z, means, cov_matrices, input_image_path, points_sub_image_2d,
                                                nrows, ncols)

        frequencies = get_feature_frequencies(feature_classification)

        # Apply Rule Based Classifier on clustered image
        variation = 0.05
        category = rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels,
                                                  variation)
        print("Category with ", variation, "variation :", category)

        # Generate list of mappings by changing allowed variation
        n_iterations = 5
        image_category_list = []
        for i in range(n_iterations):
            image_category_list.append(
                rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels,
                                               (i + 1) * 0.05))

        # Use majority vote to classify the image as Urban or Rural
        image_category_by_majority = max(image_category_list, key=image_category_list.count)
        print("Categories for ", n_iterations, " iterations ", ":", image_category_list)
        print("Image category by majority : ", image_category_by_majority)
        weights = get_feature_weights(frequencies)
        weights.append(str(image_category_by_majority))
        labelled_patches.append(weights)

        if image_category_by_majority == "Residential Type1" or image_category_by_majority == "Residential Type2" or image_category_by_majority == "Commercial":
            print("Settlement Type : Urban")
        else:
            print("Settlement Type : Rural")
            # majority_vote = majorityVote_settlement_type(image_settlement_mapping_list)

    output_file = inputs_home + 'test_weights.csv'

    if PYTHON_VERSION == 3:
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(labelled_patches)
        print("\n\n")
    elif PYTHON_VERSION == 2:
        with open(output_file, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(labelled_patches)
        print("\n\n")


# Function to get feature classification by using Gaussian Mixture Model from pixel data input
def get_feature_classification_pixel_input(points_input_image, nrows, ncols, Z, means, cov_matrices, input_image_path):
    # Cluster the input image using GMM
    print("Testing GMM+RBC on", input_image_path, "with", nrows, "rows and", ncols, "columns")
    feature_classification = get_components(Z, means, cov_matrices, input_image_path, points_input_image, nrows, ncols)

    return feature_classification


# Function to merge patches into complete image
def merge_image(patch_filename_pattern, image_row_size, image_col_size, patch_row_size, patch_col_size, identifier,
                delete_patch_file):
    file_name, file_extension = patch_filename_pattern.split('.')

    merged_image = Image.new('RGB', (image_col_size, image_row_size))

    image_index = 0

    # Fill output image grid with patches
    for i in range(0, image_col_size, patch_col_size):
        for j in range(0, image_row_size, patch_row_size):
            patch_file_name = 'output/' + file_name + "_" + identifier + "_" + str(
                image_index) + "_output" + '.' + file_extension
            patch = Image.open(patch_file_name)
            image_index = image_index + 1

            # insert the patch at location i,j:
            merged_image.paste(patch, (i, j))

            if (delete_patch_file):
                os.remove(patch_file_name)

    merged_image_filename = 'output/' + file_name + "_" + identifier + "_output." + file_extension
    merged_image.save(merged_image_filename)
    print("Image " + merged_image_filename + " of size " + str(image_row_size) + "x" + str(
        image_col_size) + " generated from patches of size " + str(
        patch_row_size) + "x" + str(patch_col_size))

    merged_image.show()

def weka_rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels, variation):
    feature_frequencies = get_feature_frequencies(feature_classification)
    feature_weights = get_feature_weights(feature_frequencies)
    category = []
    category_index = []

    if (feature_weights[5] >= 0.7352):
        category = "Water"
        category_index = 4
    elif feature_weights[5] >= 0.0712 and feature_weights[2] >= 0.2296:
        category = "Residential Type1"
        category_index = 1
    elif feature_weights[3] >= 0.1112 and feature_weights[4] <= 0.1188:
        category = "Commercial"
        category_index = 0
    elif feature_weights[3] <= 0.0008:
        category = "Vegetation"
        category_index = 3
    else:
        category = "Residential Type2"
        category_index = 2

    return category, category_index

# Function to classify
def classify_image(input_image_path, nbands, patch_row_size, patch_col_size, labelled_patch_category_weights, labels, Z,
                   means, cov_matrices):
    # Convert Input image to points in 2d
    points_input_image, nrows, ncols = load_image(input_image_path)
    print("Input image dimensions:", nrows, ncols)

    # Get input image name and extension
    input_image_name, input_image_extension = input_image_path.split('.')

    # Calculate number of patches using the patch dimensions
    npatches = int(np.ceil(float(nrows) / patch_row_size)) * int(np.ceil(float(ncols) / patch_col_size))
    print("Grid Layout", int(np.ceil(float(nrows) / patch_row_size)), "x", int(np.ceil(float(ncols) / patch_col_size)),
          "with",
          npatches, "patches\n")

    # Convert 2d points to 3d (easier to visualize and extract)
    points_input_image_3d = matrix2dTo3d(np.asarray(points_input_image), nrows, ncols, nbands)

    # Split input image into patches
    points_sub_image_3d_list = split3dMatrix_by_size(points_input_image_3d, patch_row_size, patch_col_size, nbands)

    # Get the index of patches to apply GMM+RBC
    # image_index_array = np.random.permutation(npatches)
    image_index_array = list(range(npatches))

    image_feature_classification = []
    # Apply GMM on patches
    for i in range(len(image_index_array)):
        # print(image_index_array[i])
        points_sub_image_3d = np.asarray(points_sub_image_3d_list[image_index_array[i]])

        # Convert the extracted 3d sub image to 2d as required by the in-built GMM library
        points_sub_image_2d = matrix3dTo2d(points_sub_image_3d)

        sub_image_gmm_file_name = input_image_name + "_gmm_" + str(image_index_array[i]) + '.' + input_image_extension
        feature_classification = get_feature_classification_pixel_input(points_sub_image_2d, patch_row_size,
                                                                        patch_col_size, Z, means, cov_matrices,
                                                                        sub_image_gmm_file_name)

        variation = 0.05
        #category, category_index = rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels, variation)
        category, category_index = weka_rule_based_classifier_category(feature_classification, labelled_patch_category_weights, labels, variation)
        print("Category with ", variation, "variation :", category)

        # Color patch by category
        sub_image_rbc_file_name = input_image_name + "_rbc_" + str(image_index_array[i]) + '.' + input_image_extension
        generate_rbc_image(sub_image_rbc_file_name, category, patch_col_size, patch_row_size)
        print("\n")

    merge_image(input_image_name.split('/')[-1] + '.' + input_image_extension, nrows, ncols, patch_row_size,
                patch_col_size, "gmm", 1)
    merge_image(input_image_name.split('/')[-1] + '.' + input_image_extension, nrows, ncols, patch_row_size,
                patch_col_size, "rbc", 1)


def main():
    # Returns the GMM model
    # If GMM model is not trained, this function calls train_model() and trains the model
    Z, means, cov_matrices = get_model()

    test_images_path = images_home + 'test/'
    nbands = 3

    # Input file that contains Feature Weights for Categories from labelled patches to be use in RBC
    input_file = inputs_home + 'labelled_patch_category_weights.csv'

    # Check whether RBC is trained or not
    check_if_RBC_trained(input_file, Z, means, cov_matrices)

    # Get labels and weights of pre-trained RBC
    # labels = ["Commercial", "Residential Type1", "Residential Type2", "Vegetation", "Water"]
    labels = []
    labelled_patch_category_weights = []
    get_trained_RBC_info(input_file, labels, labelled_patch_category_weights)
    labelled_patch_category_weights = np.asarray(labelled_patch_category_weights, dtype='float')

    # run_test(test_images_path, nbands, Z, means, cov_matrices, labels, labelled_patch_category_weights)

    #input_image_path = images_home + 'city1_tiny.png'
    #patch_row_size = 58
    #patch_col_size = 46
    #input_image_path = images_home + 'test_image_crop_vtiny2.tif'
    input_image_path = images_home + 'm_4308960_ne_16_1_20100702.tif'

    patch_row_size = 50
    patch_col_size = 50
    start_time = time.time()
    classify_image(input_image_path, nbands, patch_row_size, patch_col_size, labelled_patch_category_weights, labels, Z,
                   means, cov_matrices)
    elapsed_time = time.time() - start_time
    print ("Time taken:", elapsed_time)


if __name__ == '__main__':
    main()
