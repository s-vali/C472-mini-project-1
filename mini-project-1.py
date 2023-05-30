# -*- coding: utf-8 -*-

""" imports """
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz

""" train on data """

# Dataset training with
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
#print(dataset, "\n")

#df2 = pd.DataFrame(dataset, columns=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate', 'Will Wait'])
#blankIndex=[''] * len(df2)
#df2.index=blankIndex
#print("The original dataset to train on: \n", df2)

input = dataset[:, 0:10]
output = dataset[:, 10]
print("\noutput: ", output, "\ninput: \n", input)

# Convert strings to integer values for the 12x10 dataset
le = preprocessing.LabelEncoder()
print("this is length:", len(input))
for i in range(len(input)):
    for j in range(10):
        input[:, j] = le.fit_transform(input[:, j]) #input is a 12x10 matrix, goes through every column
output = le.fit_transform(output) #output is a row vector
print("input after lavel encoder: \n", input)
print("output after label encoder:", output)

# Create classifier object
dtc = tree.DecisionTreeClassifier(criterion="entropy") #splitting by entropy
dtc.fit(input, output) #training classifer object to build the decision tree

# Plot the decision tree
dot_data = tree.export_graphviz(dtc, out_file=None,
feature_names=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate'],
class_names=le.classes_,
filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision-tree")

""" inputs by user to predict answer """
  