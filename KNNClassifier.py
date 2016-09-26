import menu
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from collections import Counter
import numpy as np


rememberedData = []
rememberedTarget = []
KNN = 3


def runData(data, target):
    n_neighbors = KNN
    trainingData, trainingTarget, testData, testTarget = menu.loadData(data, target)

    # scale the data
    std_scale = preprocessing.StandardScaler().fit(trainingData)
    trainingData = std_scale.transform(trainingData)
    testData = std_scale.transform(testData)

    train(trainingData, trainingTarget)
    predictions = predict(testData)
    percent1 = menu.test(testTarget, predictions)
    print("The prediction accuracy of my algorithm was %i%%" % percent1)

    # sklearn implementation
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(trainingData, trainingTarget)
    predictions = classifier.predict(testData)
    percent = menu.test(testTarget, predictions)
    print("The prediction accuracy of the sklearn algorithm was %i%%" % percent)

    return percent1, percent


# takes car data and changes strings to numbers
def carStringToNum(data):
    car_matrix = data.as_matrix()
    cardata = []
    cartarget = []
    for row in range(len(car_matrix)):
        # string to int
        features = []
        # selecting a number to use for weight of each string
        for col in range(7):
            if car_matrix[row][col] == 'vhigh' or (car_matrix[row][col] == 'low' and col != 0 and col != 1) or \
                            car_matrix[row][col] == 'small' or car_matrix[row][col] == 'unacc':
                features.append(1.0)
            elif (car_matrix[row][col] == 'high' and col != 5) or (car_matrix[row][col] == '2' and col != 3) or \
                            car_matrix[row][col] == '5more':
                features.append(2.0)
            elif car_matrix[row][col] == '3' or car_matrix[row][col] == 'med' or car_matrix[row][col] == 'acc':
                features.append(4.0)
            elif car_matrix[row][col] == '4' or car_matrix[row][col] == 'good':
                features.append(5.0)
            else:
                features.append(9.0)

        cardata.append(features[:6])
        num = features[6]
        cartarget.append(num)
    dat = np.array(cardata)
    targ = np.array(cartarget)
    return dat, targ


def train(data, target):
    for item in range(len(data)):
        rememberedData.append(data[item])
        rememberedTarget.append(target[item])
    return


def predict(data):
    print('Making predictions...')
    predictions = []
    for item in range(len(data)):
        predictions.append(findKNN(data[item]))
    return predictions


# find kNN
def findKNN(data):
    dist = []
    for remItem in range(len(rememberedData)):
        dist.append([(data[0] - rememberedData[remItem][0]) ** 2 +
                     (data[1] - rememberedData[remItem][1]) ** 2 +
                     (data[2] - rememberedData[remItem][2]) ** 2 +
                     (data[3] - rememberedData[remItem][3]) ** 2,
                     rememberedTarget[remItem]])
    dist.sort()
    KNNlist = dist[:KNN]
    calcificationNum = []
    for item in range(len(KNNlist)):
        calcificationNum.append(KNNlist[item][1])
    c = Counter(calcificationNum)
    val, count = c.most_common()[0]
    return val


def main(argv):
    print("\nTesting Iris...")
    runData(menu.iris.data, menu.iris.target)
    print("\nTesting Car...")
    car_data, car_target = carStringToNum(menu.car_pd)
    runData(car_data, car_target)
    print("\nTesting Breast Cancer...")
    runData(menu.cancer.data, menu.cancer.target)


if __name__ == "__main__":
    main(sys.argv)
