rm(list=ls())

library(rgdal)
library(raster)
library(mclust)
library(mixtools)
library(gmm)
library(RColorBrewer)
library(imager)

imageToCluster <- function(points_input_image, k){
  df_input_image_rgb_bands <- data.frame(points_input_image[,3],points_input_image[,4],points_input_image[,5])
  colnames(df_input_image_rgb_bands) <- c("R", "G", "B")
  mclust_input_image <- Mclust(df_input_image_rgb_bands, 6)
  
  #par(mfrow=c(1,1))
  #par(mar=c(5, 4, 2, 0) + 0.1)
  #plot(loaded_input_image)
  plot(points_input_image[,1], points_input_image[,2], xlab = '', ylab = '', col = c('grey','green','orange','yellow','blue','white')[mclust_input_image$classification], pch = '.', cex = 3, sub="GMM 3 bands k = 6")
  axis(side=1, at = pretty(range(points_input_image[,1])))
  axis(side=2, at = pretty(range(points_input_image[,2])))
  #par(mfrow=c(1,1))
  
  return(mclust_input_image)
}

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

split3dMatrix <- function(matrix_3d, split_number, num_dim_3d){
  m_list <- list()
  num_matrices <- split_number * split_number
  num_row <- ceiling(nrow(matrix_3d)/split_number)
  num_col <- ceiling(ncol(matrix_3d)/split_number)
  
  for (i in 1:num_matrices){
    m_list[[i]] <- (array(rep(0, num_row*num_col*num_dim_3d), dim=c(num_row, num_col, num_dim_3d)))
  }
  
  k <- 1
  l <- 1
  matrix_index <- 1
  for(ii in seq(from = 1, to = nrow(matrix_3d), by = num_row)){
    for(jj in seq(from = 1, to = ncol(matrix_3d), by = num_col)){
      for(i in seq(from = ii, to = ii + num_row - 1, by = 1)){
        for(j in seq(from = jj, to = jj + num_col - 1, by = 1)){
          if (i <= nrow(matrix_3d) && j <= ncol(matrix_3d)){
            m_list[[matrix_index]][k,l,] <- matrix_3d[i,j,]
            l <- l + 1 
          }
        }
        k <- k + 1
        l <- 1
      }
      k <- 1
      matrix_index <- matrix_index + 1
    }
  }
  return(m_list)
}

#Grey - 1 - Concrete Ground
#Green - 2 - Vegetation
#Orange - 3 - Ground
#Yellow - 4 - Misc
#Blue - 5 - Water
#White - 6 - Buildings

getCategoryFrequencies <- function(image_cluster){
  category_frequncies <- array(rep(0, 6))
  for (i in 1:length(category_frequncies)){
    category_frequncies[i] <- length(grep(i, image_cluster$classification))
  }
  names(category_frequncies) <- c("Concrete Ground", "Vegetation", "Ground", "Misc", "Water", "Buildings")
  return(category_frequncies)
}

getCategoryWeightPercentage <- function(category_frequncies){
  category_weight_percentage <- array(rep(0, 6))
  sum_frequencies <- sum(category_frequncies)
  for (i in 1:length(category_weight_percentage)){
    category_weight_percentage[i] <- category_frequncies[i] / sum_frequencies * 100
  }
  names(category_weight_percentage) <- c("Concrete Ground", "Vegetation", "Ground", "Misc", "Water", "Buildings")
  return(category_weight_percentage)
}

ruleBasedClassifier_Settlement <- function(image_cluster){
  settlement_category <- array(rep(FALSE, 6))
  names(settlement_category) <- c("Urban", "Commercial", "Residential Type1", "Residential Type2", "Water", "Vegetation")
  
  category_frequncies <- getCategoryFrequencies(image_cluster)
  category_weights <- getCategoryWeightPercentage(category_frequncies)
  
  if((category_weights["Concrete Ground"] + category_weights["Buildings"]) > 30.0 && category_weights["Ground"] < 5.0){
    settlement_category["Urban"] <- TRUE
  }
  if(category_weights["Buildings"] > 5.0 && category_weights["Water"] < 5.0 && category_weights["Ground"] < 5.0)
  {
    settlement_category["Commercial"] <- TRUE
  }
  if(category_weights["Buildings"] > 5.0 && category_weights["Concrete Ground"] < 10.0 && category_weights["Vegetation"] > 5.0)
  {
    settlement_category["Residential Type1"] <- TRUE
  }
  if(category_weights["Buildings"] > 5.0 && category_weights["Concrete Ground"] > 10.0 && category_weights["Vegetation"] < 5.0)
  {
    settlement_category["Residential Type2"] <- TRUE
  }
  if(category_weights["Water"] > 30.0 && category_weights["Buildings"] < 5.0 && category_weights["Concrete Ground"] < 5.0)
  {
    settlement_category["Water"] <- TRUE
  }
  if(category_weights["Vegetation"] > 30.0 && category_weights["Buildings"] < 5.0 && category_weights["Concrete Ground"] < 5.0)
  {
    settlement_category["Vegetation"] <- TRUE
  }
  
  return(settlement_category)
}

majorityVote_SettlementType <- function(image_settlment_mapping){
  urban_count <- 0
  rural_count <- 0
  for (i in 1:length(image_settlment_mapping)){
    if (image_settlment_mapping[[i]]["Urban"] == TRUE){
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


#input_image_file <- '/Users/Sriware/Documents/Courses/NCSU/CSC522/Project/RCode/Code/input_crop.png'
#input_image_file <- '/Users/Sriware/Documents/Courses/NCSU/CSC522/Project/RCode/Code/m_4308960_ne_16_1_20100702.tif'
input_image_file <- '/Users/Sriware/Documents/Courses/NCSU/CSC522/Project/RCode/Tutorials/city1_tiny.png'
#input_image_file <- '/Users/Sriware/Documents/Courses/NCSU/CSC522/Project/RCode/Tutorials/city1_crop.png'
input_image <- brick(input_image_file)
points_input_image <- rasterToPoints(input_image)
points_input_image_3d <- matrix2dTo3d(points_input_image, input_image@nrows, input_image@ncols, ncol(points_input_image))
points_sub_images_list_3d <- split3dMatrix(points_input_image_3d, split_number = 3, ncol(points_input_image))

points_sub_images_list_2d <- list()
for (i in 1:length(points_sub_images_list_3d)){
  points_sub_images_list_2d[[i]] <- matrix3dTo2d(points_sub_images_list_3d[[i]])
}

par(mfrow=c(3,3))
sub_image_clusters <- list()
for (i in 1:length(points_sub_images_list_2d)){
  sub_image_clusters[[i]] <- imageToCluster(points_sub_images_list_2d[[i]])
  cat("Finished Processing",i,"out of",length(points_sub_images_list_2d),"components\n")
}
par(mfrow=c(1,1))

sub_image_settlment_mapping <- list()
for (i in 1:length(sub_image_clusters)){
  sub_image_settlment_mapping[[i]] <- ruleBasedClassifier_Settlement(sub_image_clusters[[i]])
}

majority_vote <- majorityVote_SettlementType(sub_image_settlment_mapping)

