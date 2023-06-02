
""" imports """
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz


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


""" function to print the decision tree """
def visTree(dtc, le):
    
    if dtc == None:
        print("The decision tree does not exist.")
    else: 
        dot_data = tree.export_graphviz(dtc, out_file=None,
                                        feature_names=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate'],
                                        class_names=le.classes_,
                                        filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render("mini-project-decision-tree")
        print("The file 'mini-project-decision-tree.pdf' has been made with the current decision tree.")
        

""" function to print the dataset as a table """
def visData(dataset):
    
    if dataset.shape[0] == 0 and dataset.shape[1] == 0:
        print("The dataset does not yet exist.")
    else:
        df = pd.DataFrame(dataset, columns=['Alternative', 'Bar', 'Friday','Hungry','Patrons', 'Price', 'Rain', 'Reservation', 'Type', 'Estimate', 'Will Wait'])
        blankIndex=[''] * len(df)
        df.index=blankIndex
        print("The current dataset : ", df, "\n")


""" function to predict output based on prompted inputs """
def pred(dtc, le, alt, bar, friday, hungry, pat, price, rain, res, ty, est):
    # Call predict from Data Tree Classifier
    temp_array = np.array([alt, bar, friday, hungry, pat, price, rain, res, ty, est])
    l = preprocessing.LabelEncoder()
    user_input_array = l.fit_transform(temp_array)
    output_pred = dtc.predict([user_input_array]) 
    print("Predicted output : ", le.inverse_transform(output_pred), "\n")


""" user interface """
print("\n-- Welcome to Decision Tree Program! --")

# Train dataset
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
print("\nThe default dataset : \n", dataset)
print("The agent is currently being trained by the default dataset...")
trainTree(dataset)
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
        dataset = np.vstack((dataset, np.array([alt, bar, friday, hungry, pat, price, rain, res, ty, est, ans])))
        print("The dataset has been updated successfully. \n")
    
    elif option == "2": #visualize current dataset
        visData(dataset)
    
    elif option == "3": #print decision tree
        dtc, le = trainTree(dataset)
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
        dtc, le = trainTree(dataset)
        pred(dtc, le, alt, bar, friday, hungry, pat, price, rain, res, ty, est)
    
    else: #exit
        break

print("\n-- Program successfully terminated! --")
    
    
  
