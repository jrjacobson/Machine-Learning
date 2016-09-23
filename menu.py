import hardcoded
import KNNClassifier
import sys
from sklearn import datasets
import pandas as pd

# load data
iris = datasets.load_iris()
car = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
                  header=None,
                  usecols=[0, 1, 2, 3, 5, 6, 7])

cancer = datasets.load_breast_cancer()


def main(argv):
    print("Ponder 01: Running hardcoded classifier")
    hardcoded.main(argv)
    print("Ponder 02: Running KNN classifier")
    KNNClassifier.main(argv)


if __name__ == '__main__':
    main(sys.argv)
