import numpy as np
import pandas as pd
import math

# textbook dataset
dataset = np.array([
['yes', 'no', 'no', 'yes', 'some', '$$$', 'no', 'yes', 'french', '0-10', 'yes'],
['yes', 'no', 'no', 'yes', 'full', '$', 'no', 'no', 'thai', '30-60', 'no'],
['no', 'yes', 'no', 'no', 'some', '$', 'no', 'no', 'burger', '0-10', 'yes'],
['yes', 'no', 'yes', 'yes', 'full', '$', 'yes', 'no', 'thai', '10-30', 'yes'],
['yes', 'no', 'yes', 'no', 'full', '$$$', 'no', 'yes', 'french', '>60', 'no'],
['no', 'yes', 'no', 'yes', 'some', '$$', 'yes', 'yes', 'italian', '0-10', 'yes'],
['no', 'yes', 'no', 'no', 'none', '$', 'yes', 'no', 'burger', '0-10', 'no'],
['no', 'no', 'no', 'yes', 'some', '$$', 'yes', 'yes', 'thai', '0-10', 'yes'],
['no', 'yes', 'yes', 'no', 'full', '$', 'yes', 'no', 'burger', '>60', 'no'],
['yes', 'yes', 'yes', 'yes', 'full', '$$$', 'no', 'yes', 'italian', '10-30', 'no'],
['no', 'no', 'no', 'no', 'none', '$', 'no', 'no', 'thai', '0-10', 'no'], 
['yes', 'yes', 'yes', 'yes', 'full', '$', 'no', 'no', 'burger', '30-60', 'yes'],
])

# organizing data by names of column
col_names = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'Output']


# Create a DataFrame using the dataset and column names
df = pd.DataFrame(dataset, columns=col_names)


# Print the DataFrame
print(df)

# ENTROPY CALCULATION
# Function to calculate entropy 
def calculate_entropy(labels):
    label_counts = {}
    total_count = len(labels)

    # Count the occurrences of each label
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    entropy = 0.0
    # Calculate the entropy
    for count in label_counts.values():
        # probability of each unique label in the dataset
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    return entropy


# Function to calculate information gain
# feature -> column in dataset for which we will calculate information gain. target->target variable that we are trying to predict the split based on
def calculate_information_gain(dataset, feature, target):
    # Calculate the entropy of the target variable before splitting
    initial_entropy = calculate_entropy(dataset[target])

    feature_values = dataset[feature].unique()
    weighted_entropy = 0

    # Calculate the weighted entropy for each feature value
    for value in feature_values:
        subset = dataset[dataset[feature] == value]
        subset_entropy = calculate_entropy(subset[target])
        weight = len(subset) / len(dataset)
        weighted_entropy += weight * subset_entropy

    # Calculate the information gain
    information_gain = initial_entropy - weighted_entropy
    return information_gain


# SPLITTING CRITERIA
# Function for best split
def find_best_split(dataset, features, target):
    best_feature = None
    # placeholder= minus infinity to ensure which ever first value for information gain will be bigger than the initialized one
    best_information_gain = -math.inf  

    # Iterate through each feature and calculate information gain
    for feature in features:
        information_gain = calculate_information_gain(dataset, feature, target)
        if information_gain > best_information_gain:
            best_feature = feature
            best_information_gain = information_gain

    return best_feature



# TESTING
# Specify the features and target variable
features = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
target = 'Output'

# Find the best feature for splitting
best_feature = find_best_split(df, features, target)
print("Best feature for splitting:", best_feature)



# Note: entropy is higher than 1 for this example. 

price_labels = df['Price'].tolist()
print(price_labels)

entropy = calculate_entropy(price_labels)
print("Entropy:", entropy)


# Note: in this example entropy = 1.0 

alt_labels = df['Alt'].tolist()
print(alt_labels)

entropy = calculate_entropy(alt_labels)
print("Entropy:", entropy)

# Note: higher entropy -> root of decision tree

pat_labels = df['Pat'].tolist()
print(pat_labels)

entropy = calculate_entropy(pat_labels)
print("Entropy:", entropy)

# CLASSIFICATION
