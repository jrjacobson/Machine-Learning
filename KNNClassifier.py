import hardcoded
import menu
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from collections import Counter


rememberedData = []
rememberedTarget = []
KNN = 3


def runData(dataType):
    n_neighbors = KNN
    trainingData, trainingTarget, testData, testTarget = hardcoded.loadData(dataType)
    # scale the data
    std_scale = preprocessing.StandardScaler().fit(trainingData)
    trainingData = std_scale.transform(trainingData)
    testData = std_scale.transform(testData)

    train(trainingData, trainingTarget)
    predictions = predict(testData)
    percent1 = hardcoded.test(testTarget, predictions)
    print("The prediction accuracy of my algorithm was %i%%" % percent1)

    # sklearn implementation
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(trainingData, trainingTarget)
    predictions = classifier.predict(testData)
    percent = hardcoded.test(testTarget, predictions)
    print("The prediction accuracy of the sklearn algorithm was %i%%" % percent)

    return percent1, percent


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
    runData(menu.iris)
    print("\nTesting Car...")
    #runData(menu.car)
    print("\nTesting Breast Cancer...")
    runData(menu.cancer)


if __name__ == "__main__":
    main(sys.argv)
