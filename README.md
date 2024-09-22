# K-NEAREST NEIGHBOR FROM SCRATCH
This repository contains an implementation of the K-Nearest Neighbors (KNN) classifier algorithm built entirely from scratch, without relying on any external libraries for core functionalities like distance computation or summation. Our aim is to demonstrate the power and simplicity of KNN by focusing on efficient, custom-built functions and leveraging vectorized operations for optimal performance.

## Features
- Custom Euclidean Distance Calculation: We implemented our own Euclidean distance function to measure the similarity between data points.
- Vectorized Operations: Instead of using traditional loops, we employed vectorized operations to calculate distances, improving efficiency and minimizing execution time.
- Binary Classification Task: The classifier was designed for a binary classification problem, achieving an accuracy of approximately 98%.
- No External Libraries: Our implementation avoids dependencies on libraries like scikit-learn for the core algorithm, offering a transparent, ground-up understanding of how KNN works.
- 
## Highlights
- Custom-built Functions: Key functions, such as Euclidean distance and custom summations, were manually implemented to control every aspect of the classification process.
- Efficient Performance: By avoiding loops in favor of vectorized operations, we optimized the model's speed without sacrificing accuracy.
- Accuracy: Our KNN classifier reached a high accuracy of ~98%, demonstrating the effectiveness of the approach in solving binary classification problems.
  
## Dataset
The dataset used in this project is provided in the repository. It is structured for binary classification, where each data point belongs to one of two classes. The dataset is split *manually* into training and testing sets to evaluate the model’s performance.

## How it Works

- Euclidean Distance Calculation: We calculate the distance between the test point and all training points using our custom Euclidean distance function.
- Finding Nearest Neighbors: Based on the distance, the k nearest neighbors are identified.
- Majority Voting: The class labels of the nearest neighbors are used to make the final prediction for the test point.
- Accuracy Calculation: After predictions are made for the test set, accuracy is calculated by comparing predictions with the true labels.
  
## Results
 - Our KNN implementation achieved an accuracy of approximately 98% on the binary classification task, demonstrating its robustness and simplicity.

## Contributions

- Vectorized Euclidean Distance: By avoiding loops and using vectorized operations for distance calculation, we significantly improved the performance of the algorithm.
- No Libraries for Core Algorithm: This project is a pure implementation of KNN, providing a deeper understanding of how the algorithm works under the hood.
  
## Future Improvements

- Extend support to multi-class classification tasks.
- Implement cross-validation for hyperparameter tuning.
- Add a graphical user interface (GUI) for easier interaction and visualization.
