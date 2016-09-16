from sklearn import datasets
from sklearn.utils import shuffle
import random

# import iris data from sklearn for testing
iris = datasets.load_iris()
# shuffle the data using a random number
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=int(random.random() * 100))

trainingData = iris.data[:125]
trainingTarget = iris.target[:125]
testData = iris.data[125:]
testTarget = iris.target[125:]


# Train the machine
def train(data, target):
    print('Training the machine...')
    return


# Predict the outcome of the test data
def predict(data):
    print('Making predictions...')
    prediction = []

    # This is hard coded for now will need to predict latter
    for x in range(len(data)):
        # This will be where the prediction function will get called
        prediction.append(1)

    return prediction


# Check to see what percentage was correct
def test(target, prediction):
    print('Calculating the proficiency of the prediction made...')
    right = 0
    for x in range(len(target)):
        if target[x] == prediction[x]:
            right += 1

    percent = right / 25.0 * 100

    return percent


train(trainingData, trainingTarget)
prediction = predict(testData)
percent = test(testTarget, prediction)

print ("The prediction accuracy of this test was %i%%" % percent)
