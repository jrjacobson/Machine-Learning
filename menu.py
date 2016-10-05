import hardcoded
import KNNClassifier
import sys
from sklearn import datasets
from sklearn.utils import shuffle
import random
import pandas as pd
import numpy as np
import csv


def loadData(data, target):
    # shuffle the data using a random number
    data, target = shuffle(data, target, random_state=int(random.random() * 100))
    trainingData = data[:int(len(data)*.7)]
    trainingTarget = target[:int(len(data)*.7)]
    testData = data[int(len(data)*.7):]
    testTarget = target[int(len(data)*.7):]
    return trainingData, trainingTarget, testData, testTarget


def preProsVotes(filename):
    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    f.close()
    shouldRemoveRow = False
    numRemoved = 0
    for row in range(len(dataset)):
        adjusted_row = row - numRemoved
        for col in range(17):
            if dataset[adjusted_row][col] == 'democrat' or dataset[adjusted_row][col] == 'y':
                dataset[adjusted_row][col] = 0
            elif dataset[adjusted_row][col] == 'republican' or dataset[adjusted_row][col] == 'n':
                dataset[adjusted_row][col] = 1
            else:
                shouldRemoveRow = True
        if shouldRemoveRow:
            del dataset[adjusted_row]
            numRemoved += 1
            shouldRemoveRow = False
    return dataset


# Check to see what percentage was correct
def test(target, prediction):
    print('Calculating the proficiency of the prediction made...')
    right = 0
    for x in range(len(target)):
        if target[x] == prediction[x]:
            right += 1
    percent = right / float(len(target)) * 100
    return percent


def main(argv):
    print("\nPonder 01: Running hardcoded classifier")
    #hardcoded.main(argv)
    #print("\nPonder 02: Running KNN classifier")
    #KNNClassifier.main(argv)

# load data
iris = datasets.load_iris()
car_pd = pd.read_csv('car.csv',
                     header=None, usecols=[0, 1, 2, 3, 4, 5, 6])
cancer = datasets.load_breast_cancer()
filename = "votes.csv"
votes = preProsVotes(filename)

if __name__ == '__main__':
    main(sys.argv)
