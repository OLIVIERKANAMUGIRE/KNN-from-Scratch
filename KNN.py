# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:53:32 2024

@author: K.Olivier
"""
#%%  Custom functions that will be used

def custom_sum(arr, axis=None):
    if axis is None:
        total = 0
        for elem in arr:
            total += elem
        return total
    else:
        if axis == 0:  # Sum along columns
            return [custom_sum(arr[i]) for i in range(len(arr[0]))]
        elif axis == 1:  # Sum along rows
            return [custom_sum(arr[i]) for i in range(len(arr))]

def custom_argsort(arr):
    return sorted(range(len(arr)), key=lambda i: arr[i])

def euclidean_metric(traindata, testdata):
    num_train_samples = len(traindata)
    num_test_samples = len(testdata)
    num_features = len(traindata[0])
    
    distances = []
    for i in range(num_test_samples):
        test_sample = testdata[i]
        test_distances = []
        for j in range(num_train_samples):
            train_sample = traindata[j]
            squared_diff = [(a - b) ** 2 for a, b in zip(train_sample, test_sample)]
            test_distances.append(custom_sum(squared_diff))
        distances.append(test_distances)
    
    return distances

def majority_vote(labels):
    label_count = {}
    
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    
    # Find the label with the maximum count
    max_count = -1
    most_common_label = None
    for label, count in label_count.items():
        if count > max_count:
            max_count = count
            most_common_label = label
            
    return most_common_label

#%% KNN ALGORITHM IMPLEMENTATION AND ITS INSTANTIATION

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, traindata, trainclass):
        self.traindata = traindata
        self.trainclass = trainclass
    
    def predict(self, testdata):
        distances = euclidean_metric(self.traindata, testdata)
        
        k_nearest_labels = []
        for d in distances:
            k_indices = custom_argsort(d)[:self.k]
            k_labels = [self.trainclass[i] for i in k_indices]
            k_nearest_labels.append(majority_vote(k_labels))
        
        return k_nearest_labels


#%% Instantiate KNN classifier
def knn(trainclass, traindata, testdata, k):
    knn = KNN(k=k)
    knn.fit(traindata, trainclass)  
    return knn.predict(testdata)  

#%% LOAD YOUR DATA AND SPLITTING FUNCTION

def load_and_split_data(dataset):
    data = dataset.iloc[:, :-1].values  # All columns except the last one as features
    labels = dataset.iloc[:, -1].values  # Last column as class labels
    
    # Split data: first 466 samples for training, the rest for testing
    train_data, test_data = data[:466], data[466:]
    train_class, test_class = labels[:466], labels[466:]
    
    return train_data, train_class, test_data, test_class

#%% 
import pandas as pd
dataset = pd.read_csv(r"D:\Desktop\CVPR-LUT\Pattern Recognition and machine Learning\Task2_Olivier\t031.csv")
train_data, train_class, test_data, test_class = load_and_split_data(dataset)

predictions = knn(train_class, train_data, test_data, k=3)
for k in range(1,9):
    knn = KNN(k)  
    knn.fit(train_data, train_class)  
    predictions = knn.predict(test_data)  
    acc = custom_sum(predictions == test_class)/len(test_class)
    print('For k = ',k, 'The acccuracy is: ',acc)



