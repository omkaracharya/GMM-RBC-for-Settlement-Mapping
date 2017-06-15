#########
# Code for applying GMM and RBC in R
#########


# Clear workspace
rm(list=ls())

# Load required libraries
library(rgdal)
library(raster)
library(mclust)
library(mixtools)
library(gmm)
library(RColorBrewer)
library(imager)

# Function to classify an image using built-in GMM function Mclust() with R, G, and B bands as input
# Also plots the image classification using upto six colors
imageToCluster <- function(points_input_image, k){
  df_input_image_rgb_bands <- data.frame(points_input_image[,3],points_input_image[,4],points_input_image[,5])
  colnames(df_input_image_rgb_bands) <- c("R", "G", "B")
  mclust_input_image <- Mclust(df_input_image_rgb_bands, k)
  
  plot(points_input_image[,1], points_input_image[,2], xlab = '', ylab = '', col = c('grey','green','orange','yellow','blue','white')[mclust_input_image$classification], pch = '.', cex = 3, sub="GMM 3 bands k = 6")
  axis(side=1, at = pretty(range(points_input_image[,1])))
  axis(side=2, at = pretty(range(points_input_image[,2])))
  
  return(mclust_input_image)
}

# Function to convert a 2d matrix to 3d matrix with num_dim_3d dimensions along 3rd axis
matrix2dTo3d <- function(matrix_2d, num_rows_3d, num_cols_3d, num_dim_3d){
  matrix_3d <- array(rep(0, num_rows_3d * num_cols_3d * num_dim_3d), dim=c(num_rows_3d, num_cols_3d, num_dim_3d))
  for (i in 1:nrow(matrix_2d)){
    j <- ceiling(i / num_cols_3d)
    k <- i %% num_cols_3d
    if(k == 0){
      k <- num_cols_3d
    }
    matrix_3d[j,k,] <- matrix_2d[i,] 
  }
  return(matrix_3d)
}

# Function to convert a 3d matrix to 2d matrix
# The data along 3rd axis is layed out as columns in the 2d matrix
matrix3dTo2d <- function(matrix_3d){
  matrix_2d <- array(rep(0, dim(matrix_3d)[1] * dim(matrix_3d)[2] * dim(matrix_3d)[3]), dim=c(dim(matrix_3d)[1] * dim(matrix_3d)[2], dim(matrix_3d)[3]))
  k <- 1
  for (i in 1:nrow(matrix_3d)){
    for (j in 1: ncol(matrix_3d)){
      matrix_2d[k,] <- matrix_3d[i,j,] 
      #cat(i,j,k,"\n")
      k <- k + 1
    }
  }
  matrix_2d <- matrix_2d[matrix_2d[,6]!=0,] # Remove 0 rows
  return(matrix_2d)
}

# Function to extract a sub matrix from a 3d matrix 
extractSub3dMatrix <- function(matrix_3d, x_min, y_min, x_max, y_max, num_dim_3d){
  num_row <- y_max - y_min
  num_col <- x_max - x_min
  
  sub_matrix_3d <- (array(rep(0, num_row*num_col*num_dim_3d), dim=c(num_row, num_col, num_dim_3d)))
  
  k <- 1
  l <- 1
  for(ii in seq(from = 1, to = nrow(matrix_3d), by = num_row)){
    for(jj in seq(from = 1, to = ncol(matrix_3d), by = num_col)){
      for(i in seq(from = ii, to = ii + num_row - 1, by = 1)){
        for(j in seq(from = jj, to = jj + num_col - 1, by = 1)){
          if (i <= nrow(matrix_3d) && j <= ncol(matrix_3d)){
            sub_matrix_3d[k,l,] <- matrix_3d[i,j,]
            l <- l + 1 
          }
        }
        k <- k + 1
        l <- 1
      }
    }
  }
  return(sub_matrix_3d)
}

#Grey - 1 - Concrete Ground
#Green - 2 - Vegetation
#Orange - 3 - Ground
#Yellow - 4 - Misc
#Blue - 5 - Water
#White - 6 - Buildings

# Function to get number of pixels in each cluster
getFeatureFrequencies <- function(image_cluster, num_clusters){
  feature_frequncies <- array(rep(0, num_clusters))
  for (i in 1:length(feature_frequncies)){
    feature_frequncies[i] <- length(grep(i, image_cluster$classification))
  }
  names(feature_frequncies) <- c("Concrete Ground", "Vegetation", "Ground", "Misc", "Water", "Buildings")
  return(feature_frequncies)
}

# Function to get feature weight using output of function getFeatureFrequencies()
getFeatureWeights <- function(feature_frequncies, num_clusters){
  feature_weights <- array(rep(0, num_clusters))
  sum_frequencies <- sum(feature_frequncies)
  for (i in 1:length(feature_weights)){
    feature_weights[i] <- feature_frequncies[i] / sum_frequencies
  }
  names(feature_weights) <- c("Concrete Ground", "Vegetation", "Ground", "Misc", "Water", "Buildings")
  return(feature_weights)
}


# Function applying 1-Holte Rule on a clustered image
# Variation to be specified as float (e.g 0.1 for 10%)
ruleBasedClassifier_Settlement <- function(image_cluster, variation){
  feature_frequncies <- getFeatureFrequencies(image_cluster)
  feature_weights <- getFeatureWeights(feature_frequncies)
  
  category_matrix <- matrix(rep(0,30), ncol=6)
  
  for (i in 1:nrow(labelled_patch_category_weights)){
    for (j in 1:length(feature_weights)){
      if(feature_weights[j] <= (labelled_patch_category_weights[i,j] + (labelled_patch_category_weights[i,j] * variation)) && feature_weights[j] > (labelled_patch_category_weights[i,j] - (labelled_patch_category_weights[i,j] * variation))) {
        category_matrix[i,j] = 1
      }
    }
  }
  
  category_max_row_index <- which.max(rowSums(category_matrix))
 
  settlement_info <- c("Category"=rownames(labelled_patch_category_weights)[category_max_row_index],"Urban"=labelled_patch_category_weights[category_max_row_index, "Urban"])
  return(settlement_info)
}

# Function applying majority vote on a list of subimages to classify an image as Urban or Rural
majorityVote_SettlementType <- function(image_settlment_mapping_list){
  urban_count <- 0
  rural_count <- 0
  for (i in 1:length(image_settlment_mapping_list)){
    if (image_settlment_mapping_list[[i]]["Urban"] == 1){
      urban_count <- urban_count + 1
    }
    else{
      rural_count <- rural_count + 1
    }
  }
  
  if (urban_count == rural_count){
    return("Tie"
    )
  }
  if (urban_count < rural_count){
    return("Rural")
  }
  else{
    return("Urban")
  }
}


# Entered the info about labelled sub images here
labelled_patch_category_weights <- t(replicate(5, diff(c(0, sort(runif(5)), 1))))
colnames(labelled_patch_category_weights) <- c("Concrete Ground", "Vegetation", "Ground", "Misc", "Water", "Buildings")
rownames(labelled_patch_category_weights) <- c("Residential Type1", "Residential Type2", "Commercial", "Water", "Vegetation")
labelled_patch_category_weights <- cbind(labelled_patch_category_weights, Urban=c(TRUE, TRUE, TRUE, FALSE, FALSE))

# Input image using Raster brick
input_image_file <- '/Users/Sriware/Documents/Courses/NCSU/CSC522/Project/RCode/Tutorials/city1_tiny.png'
input_image <- brick(input_image_file)

# Convert Input image raster to points in 2d
points_input_image <- rasterToPoints(input_image)

# Convert 2d points to 3d (easier to visualize and extract)
points_input_image_3d <- matrix2dTo3d(points_input_image, input_image@nrows, input_image@ncols, ncol(points_input_image))

# Extract the sub image to classify from the 3d Matrix containing image information
points_sub_image_3d <- extractSub3dMatrix(points_input_image_3d, 0, 0, ncol(points_input_image_3d), nrow(points_input_image_3d), ncol(points_input_image))

# Convert the extracted 3d sub image to 2d as required by the in-built GMM library
points_sub_image_2d <- matrix3dTo2d(points_sub_image_3d)

# Cluster the sub image
par(mfrow=c(1,1))
sub_image_cluster <- imageToCluster(points_sub_image_2d, 6)
par(mfrow=c(1,1))

# Apply Rule Based Classifier on clustered sub image 
sub_image_settlment_mapping <- ruleBasedClassifier_Settlement(sub_image_cluster, 0.1)


# Generate list of mappings by changing allowed variation
sub_image_settlment_mapping_list <- list()
for (i in 1:5){
  sub_image_settlment_mapping_list[[i]] <- ruleBasedClassifier_Settlement(sub_image_cluster, i * 0.05)
}

# Use majority vote to classify the image as Urban or Rural
majority_vote <- majorityVote_SettlementType(sub_image_settlment_mapping_list)

