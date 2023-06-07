import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn import preprocessing
import graphviz


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


# DECISION TREE CONSTRUCTION
""" train tree """
def trainTree(dataset):

    # Determine output and input arrays
    input_matrix = dataset[:, 0:(dataset.shape[1]-1)]
    output = dataset[:, (dataset.shape[1]-1)]
    
    # Convert strings to integer values for the 12x10 dataset
    le = preprocessing.LabelEncoder()
    col_names = input_matrix.shape[1]
    for i in range(len(input_matrix)):
        for j in range(col_names): 
            input_matrix[:, j] = le.fit_transform(input_matrix[:, j]) #input_matrix is a 12x10 matrix, goes through every column
    output = le.fit_transform(output) #output is a row vector

    # Create classifier object
    dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=9) #splitting by entropy
    dtc.fit(input_matrix, output) #training classifer object to build the decision tree
    
    # Return so other functions can make use of dtc and le
    return dtc, le



# Print the DataFrame
print(df, "\n")

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



# TESTING FOR ENTROPY AND BEST FEATURE FOR SPLITTING
# Specify the features and target variable
features = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
target = 'Output'

# Find the best feature for splitting
best_feature = find_best_split(df, features, target)
print("Best feature for splitting:", best_feature, "\n")


# Note: entropy is higher than 1 for this example. 

price_labels = df['Price'].tolist()
print("Price: ", price_labels, "\n")

entropy = calculate_entropy(price_labels)
print("Entropy:", entropy, "\n")


# Note: in this example entropy = 1.0 

alt_labels = df['Alt'].tolist()
print("Alternative: ", alt_labels, "\n")

entropy = calculate_entropy(alt_labels)
print("Entropy:", entropy, "\n")

# Note: higher entropy -> root of decision tree

pat_labels = df['Pat'].tolist()
print("Patrons:", pat_labels, "\n")

entropy = calculate_entropy(pat_labels)
print("Entropy:", entropy, "\n")

""" function to print the decision tree """
def visTree(dtc, le):
    
    if dtc == None:
        print("The decision tree does not exist.")
    else: 
        dot_data = tree.export_graphviz(dtc, out_file=None,
                                        feature_names=['Alt', 'Bar', 'Fri','Hun','Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
                                        class_names=le.classes_,
                                        filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render("mini-project-decision-tree")
        print("The file 'mini-project-decision-tree.pdf' has been made with the current decision tree.\n")
        

""" function to print the dataset as a table """
def visData(dataset):
    
    if dataset.shape[0] == 0 and dataset.shape[1] == 0:
        print("The dataset does not yet exist.")
    else:
        df = pd.DataFrame(dataset, columns=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate', 'Will Wait'])
        blankIndex=[''] * len(df)
        df.index=blankIndex
        print("The current dataset : \n", dataset, "\n")


""" function to predict output based on prompted inputs """
def pred(dtc, le, alt, bar, friday, hungry, pat, price, rain, res, ty, est):
    # Call predict from Data Tree Classifier
    temp_array = np.array([alt, bar, friday, hungry, pat, price, rain, res, ty, est])
    l = preprocessing.LabelEncoder()
    user_input_array = l.fit_transform(temp_array)
    output_pred = dtc.predict([user_input_array]) 
    print("Predicted output : ", le.inverse_transform(output_pred), "\n")


    


# THIS PART CRASHES INSIDE THE MENU OPTIONS

""" user interface """
print("\n-- Welcome to Decision Tree Program! --")

# Train dataset

# Prompt the user to enter the filename
filename = input("Please enter the filename (including the extension): ")

# Read the CSV file using the provided filename
df = pd.read_csv(filename)

# Convert DataFrame to a numpy array
db = df.to_numpy()


print("\nThe default dataset : \n", df)
print("The agent is currently being trained by the default dataset...")
dtc, le = trainTree(db)  # Update dtc and le with the trained decision tree
print("The agent has been trained.\n")

# List menu options 
while(True):
    print("-- Menu --\n1. Update dataset\n2. Visualize current dataset\n3. Visualize current decision tree\n4. Predict a decision\n5. Exit\n")
    option = str(input("option (number) : "))
    
    if option == "1": #update dataset
        #prompt user for dataset
        print("Please input the following information...")
        alt = str(input("Alternative : ")).lower()
        bar = str(input("Bar : ")).lower()
        friday = str(input("Friday : ")).lower()
        hungry = str(input("Hungry : ")).lower()
        pat = str(input("Patrons : ")).lower()
        price = str(input("Price : ")).lower()
        rain = str(input("Rain : ")).lower()
        res = str(input("Reservation : ")).lower()
        ty = str(input("Type : ")).lower()
        est = str(input("Estimate : ")).lower()
        ans = str(input("Answer : ")).lower()     
        #update dataset with new values
        db = np.vstack((db, np.array([alt, bar, friday, hungry, pat, price, rain, res, ty, est, ans])))
        print("The dataset has been updated successfully. \n")
    
    elif option == "2": #visualize current dataset
        visData(df)
    
    elif option == "3": #print decision tree
        dtc, le = trainTree(db)
        visTree(dtc, le)
   
    elif option == "4": #predict
        #prompt user for dataset
        print("Please input the following information...")
        alt = str(input("Alternative : ")).lower()
        bar = str(input("Bar : ")).lower()
        friday = str(input("Friday : ")).lower()
        hungry = str(input("Hungry : ")).lower()
        pat = str(input("Patrons : ")).lower()
        price = str(input("Price : ")).lower()
        rain = str(input("Rain : ")).lower()
        res = str(input("Reservation : ")).lower()
        ty = str(input("Type : ")).lower()
        est = str(input("Estimate : ")).lower()
        dtc, le = trainTree(db)
        pred(dtc, le, alt, bar, friday, hungry, pat, price, rain, res, ty, est)
    
    else: #exit
        break

print("\n-- Program successfully terminated! --")
    
    
  









