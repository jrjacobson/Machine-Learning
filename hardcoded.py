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

    percent = right / float(len(target)) * 100

    return percent


# Prompts to add a flower then predicts what type of flower it was
def add_flower():
    promptForFlower = input('Would you like me to guess your iris? ([Y]es/[N]o: ')
    print(promptForFlower)
    if promptForFlower == 'y' or promptForFlower == 'Y':
        seapalLength = input('What is your flowers sepal length: ')
        seapalWidth = input('What is your flowers sepal width: ')
        petalLength = input('What is your flowers petal length: ')
        petalWidth = input('What is your flowers petal width: ')
        flowerData = {seapalLength, seapalWidth, petalLength, petalWidth}

        guess = predict(flowerData)
        target = []
        target.append(int(input('Okay I have my guess ready what type of flower do you have?'
                                ' Enter 1 for a Setosa, 2 for a Versicolor, or 3 for a Virginica: ')))
        if target == 1 or 2 or 3:
            result = test(target, guess)
            if result == 100:
                print('I got it')
            else:
                print('I was wrong')
        else:
            print('Invalid input')

        return
    elif promptForFlower == 'n' or promptForFlower == 'N':
        return
    else:
        print('You must enter a Y for yes or a N for no.')
        add_flower()
        return


train(trainingData, trainingTarget)
prediction = predict(testData)
percent = test(testTarget, prediction)

print("The prediction accuracy of this test was %i%%" % percent)
add_flower()
