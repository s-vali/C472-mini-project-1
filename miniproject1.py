# -*- coding: utf-8 -*-

""" imports """
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz

""" train tree """
def trainTree(dataset):
    #determine output and input arrays
    y = dataset[:, 0:10]
    output = dataset[:, 10]
    #print("\noutput: ", output, "\ninput: \n", y)
    
    # Convert strings to integer values for the 12x10 dataset
    le = preprocessing.LabelEncoder()
    print("this is length:", len(y))
    for i in range(len(y)):
        for j in range(10): #HAVE TO FIND A WAY TO FIND THE NUMBER OF COLUMNS
            y[:, j] = le.fit_transform(y[:, j]) #y is a 12x10 matrix, goes through every column
    output = le.fit_transform(output) #output is a row vector
    print("y after label encoder: \n", y)
    print("output after label encoder:", output)

    # Create classifier object
    dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=9) #splitting by entropy
    dtc.fit(y, output) #training classifer object to build the decision tree
    return dtc, le

""" function to print the dataset as a table """
def visData(dataset):
    if False: #dataset.any() == False:
        print("The dataset does not yet exist.")
    else:
        df2 = pd.DataFrame(dataset, columns=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate', 'Will Wait'])
        blankIndex=[''] * len(df2)
        df2.index=blankIndex
        print("The original dataset to train on: \n", df2)

""" function to print the decision tree """
def visTree(dtc, le):
    # Plot the decision tree
    if dtc == None:
        print("The decision tree does not exist.")
    else: 
        dot_data = tree.export_graphviz(dtc, out_file=None,
                                        feature_names=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate'],
                                        class_names=le.classes_,
                                        filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render("decision-tree-1")
        print("The file 'decision-tree.pdf' has been made.")
    
""" function to predict output based on prompted inputs """
def predict(dtc, le, alt, bar, friday, hungry, pat, price, rain, res, ty, est):
    y_pred = dtc.predict([[alt, bar, friday, hungry, pat, price, rain, res, ty, est]]) 
    print("Predicted output : ", le.inverse_transform(y_pred))

""" user interface """
print("\n-- Welcome to Decision Tree Program! --")

# train dataset
print("The agent is currently being trained by the default dataset...\n")
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
print("the default dataset : \n", dataset)
 
while(True):
    print("\n-- Menu --\n1. Update dataset\n2. Visualize current dataset\n3. Visualize current decision tree\n4. Predict a decision\n5. Exit")
    option = int(input("option (number) : "))
    
    if option == 1: #update dataset
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
        
        #update dataset with new values
        dataset = np.append(dataset, [alt, bar, friday, hungry, pat, price, rain, res, ty, est], axis=0)
        print(dataset)
    elif option == 2: #visualize current dataset
        visData(dataset)
    elif option == 3: #print decision tree
        print("option 3")
        t, z = trainTree(dataset)
        visTree(t, z)
    elif option == 4: #predict
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
        t, z = trainTree(dataset)
        predict(t, z, alt, bar, friday, hungry, pat, price, rain, res, ty, est)
    else: #exit
        break

print("\n-- Program successfully terminated! --")
    
    
  