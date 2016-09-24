import menu
import sys


def runData(data, target):
    trainingData, trainingTarget, testData, testTarget = menu.loadData(data, target)
    train(trainingData, trainingTarget)
    prediction = predict(testData)
    percent = menu.test(testTarget, prediction)
    print("The prediction accuracy of this test was %i%%" % percent)


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
            result = menu.test(target, guess)
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


def main(argv):
    # setting up running iris data
    runData(menu.iris.data, menu.iris.target)
    # let user give an iris to guess
    add_flower()


if __name__ == "__main__":
    main(sys.argv)
