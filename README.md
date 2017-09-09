# GMM + RBC for Settlement Mapping

A combination of unsupervised learning (soft clustering through a Gaussian Mixture Model) and supervised learning (classification via a Rule-Based Classifier using the GMM results as attributes) is used to automate the process of classifying patches of satellite images as one of five area types: commercial, residential 1, residential 2, water, and vegetation. Both the GMM and RBC were built from scratch. The final results of our Rule-Based Classifier are of comparable quality to the Rule-Based Classifier used in the Weka package.

We have written a [paper](/Documentation/Paper.pdf) on this project and also have created a [poster](/poster.png).
  
![Poster](/Documentation/Poster.PNG)
