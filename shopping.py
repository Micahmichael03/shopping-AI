import csv 
import sys
# import pandas as pd
# import numpy as numpy 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # # import data using pandas
    # data = pd.read_csv('shopping.csv', header=0)

    # # create dictionary that maps months names to month number
    # d = {'jan':0, 'feb':1, 'mar':2, 'apr':3, 'may':4, 'jun':5, 'jul':6, 'aug':7, 
    #     'sep':8, 'oct':9, 'nov':10, 'dec':11}
    # # change month abbreviations to numbers values
    # data.Month = data.Month.map(d)

    # #change visitor value to 1 for returning visitors and 0 for non-returning visitors
    # #NOTE: mapping dictionary not used because apart from returning visitors, all visitors are non-returning
    # data.VisitorType = data.VisitorType.map(lambda x : 1 if x == 'Returning_Visitor' else 0)
    # #change boolean values in weekend to 1 or 0
    #alternative to .map()would be not .replace(True, 1) or .replace(False, 0)
    # data.Weekend = data.Weekend.map(lambda x : 1 if x == True else 0)
    
    
    # data.Revenue = data.Revenue.map(lambda x : 1 if x == True else 0)

    # # define integer and flaot colume for check up of dtype below
    # ints = ['Administrative', 'Informational', 'ProductRelated', 'Month', 'OperatingSystems', 
    #         'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
    # floats = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 
    #           'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

    # #check type of integer colume and convert if not of integer-type
    # for value in ints:
    #     if data[value].dtype != 'int64':
    #         data = data.astype({value: 'int64'})
    #     else:
    #         continue

    # #check type of float colume and convert if not of float-type
    # for value in floats:
    #     if data[value].dtype != 'float64':
    #         data = data.astype({value: 'float64'})
    #     else:
    #         continue

    # #evidence-value: create list of lists from dataframe
    # evidence = data.iloc[:,:-1:].values.tolist()
    # #labels-value: create list of labels from dataframe
    # labels = data.iloc[:,-1].values.tolist()

    # #check if evidence and labels are of same length
    # if len(evidence) != len (labels):
    #     print("ERROR!: Evidence and labels lists not of same length. check code!")
    # else:
    #     print(f'There are {len(evidence)} entries in this dataset.\n')

    # #return evidence and labels as tuple of lists and labels
    # return (evidence, labels)
    
    evidence = []
    labels = []
    months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                int(row['VisitorType'] == 'Returning_Visitor'),
                int(row["Weekend"] == 'TRUE')
            ])
            labels.append(int(row['Revenue'] == 'TRUE'))
            
    return evidence, labels
    
def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #create classifier implementing the K-nearest neighbors
    model = KNeighborsClassifier(n_neighbors = 1)

    #train model
    model.fit(evidence, labels)

    #return training classifier
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # get number of actual positives from labels
    positives = labels.count(1)

    # get number of actual negatives from labels
    negatives = labels.count(0)

    #initiate sensitivity and specificity variables
    sens = 0
    spec = 0

    #iterate over actual labels and predicted labels
    true_positive = float(0)
    true_negative = float(0)
    total_positives = labels.count(1)
    total_negatives = labels.count(0)

    for label, prediction in zip(labels, predictions):
        if prediction == label and label == 1:
            true_positive += 1
        elif prediction == label and label == 0:
            true_negative += 1
        else:
            continue
    sensitivity = true_positive / total_positives
    specificity = true_negative / total_negatives

    return sensitivity, specificity

if __name__ == "__main__":
    main()
