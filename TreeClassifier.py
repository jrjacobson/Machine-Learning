import menu
import sys
import numpy as np


class TreeClassifier:
    def __init__(self):
        """ Constructor """

    def calc_entropy(self, p):
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def calc_info_gain(self, data, classes, feature):
        gain = 0
        nData = len(data)

        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0

        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1
            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
            valueIndex += 1
        return gain

    def make_tree(self, data, target, column):
        """Builds a tree given the data set targets and column names"""
        global newColumns, feature
        amtOfData = len(data)
        numOfFeatures = len(data[0])

        try:
            self.column
        except:
            self.column = column

        # List the possible classes
        newTargets = []
        for aclass in target:
            if newTargets.count(aclass) == 0:
                newTargets.append(aclass)

        # Compute the default class (and total entropy)
        frequency = np.zeros(len(newTargets))

        totalEntropy = 0
        index = 0
        for aclass in newTargets:
            frequency[index] = target.count(aclass)
            totalEntropy += self.calc_entropy(float(frequency[index]) / amtOfData)
            index += 1
        default = target[np.argmax(frequency)]
        if amtOfData == 0 or numOfFeatures == 0:
            # empty branch
            return default
        elif target.count(target[0]) == amtOfData:
            # only 1 class left
            return target[0]
        else:
            # choose best feature
            gain = np.zeros(numOfFeatures)
            for feature in range(numOfFeatures):
                g = self.calc_info_gain(data, target, feature)
                gain[feature] = totalEntropy - g
            bestFeature = np.argmax(gain)
            tree = {column[bestFeature]: {}}
            values = []

            for dp in data:
                if dp[feature] not in values:
                    values.append(dp[bestFeature])

            for value in values:
                newData = []
                newTargets = []
                index = 0
                for dp in data:
                    if dp[bestFeature] == value:
                        if bestFeature == 0:
                            newdatapoint = dp[1:]
                            newColumns = column[1:]
                        elif bestFeature == numOfFeatures:
                            newdatapoint = dp[:-1]
                            newColumns = column[:-1]
                        else:
                            newdatapoint = dp[:bestFeature]
                            newdatapoint.extend(dp[bestFeature + 1:])
                            newColumns = column[:bestFeature]
                            newColumns.extend(column[bestFeature + 1:])
                        newData.append(newdatapoint)
                        newTargets.append(target[index])
                    index += 1

                # Now recurse to the next level
                subtree = self.make_tree(newData, newTargets, newColumns)

                # And on returning, add the subtree on to the tree
                tree[column[bestFeature]][value] = subtree
            return tree

    def printTree(self, tree, space):
        """Takes a tree and space to print a tree"""
        if type(tree) == dict:
            print(space, list(tree.keys())[0])
            for item in list(tree.values())[0].keys():
                print(space, item)
                self.printTree(list(tree.values())[0][item], space + "\t")
        else:
            print(space, "\t->\t", tree)

    def predictOneRow(self, data, tree, cols):
        """This will predict the target of a row"""
        col = 0
        if type(tree) == dict:
            root = list(tree.keys())[0]
            for col in range(len(cols)):
                if cols[col] == root:
                    break
            if data[col] in tree[root]:
                subTree = tree[root][data[col]] #sometimes the key is not there and I get a KeyError
                return self.predictOneRow(data, subTree, cols)
            else:
                subTree = tree[root]
                return self.predictOneRow(data, subTree, cols)
        else:
            return tree

    def makePredictions(self, data, tree, cols):
        """This takes a test data and passes it by row to the predict row accumulating a list of predictions"""
        predictions = []
        for row in range(len(data)):
            # send a row down the tree append the prediction to the prediction list
            predictions.append(self.predictOneRow(data[row], tree, cols))
        return predictions


def preProVotes(dataset):
    data = []
    target = []
    for items in range(len(dataset)):
        data.append(dataset[items][1:])
        target.append(dataset[items][0])
    trainingData, trainingTarget, testData, testTarget = menu.loadData(data, target)
    cols = []
    for i in range(len(trainingData[0])):
        cols.append(i)
    return trainingData, trainingTarget, testData, testTarget, cols


def preProLenses(dataset):
    data = []
    target = []
    for row in range(len(dataset)):
        dataset[row] = dataset[row][0].split()
        data.append(dataset[row][1:5])
        if dataset[row][5] == '1':
            dataset[row][5] = 'one'
        elif dataset[row][5] == '2':
            dataset[row][5] = 'two'
        else:
            dataset[row][5] = 'three'
        target.append(dataset[row][5])
    trainingData, trainingTarget, testData, testTarget = menu.loadData(data, target)
    cols = []
    for i in range(len(trainingData[0])):
        cols.append(i)
    return trainingData, trainingTarget, testData, testTarget, cols


def runTreeClassifier(trainingData, trainingTarget, testData, testTarget, cols):
    """driver program for treeClassifier"""
    tree = TreeClassifier()
    this_tree = tree.make_tree(trainingData, trainingTarget, cols)
    tree.printTree(this_tree, ' ')
    vote_predictions = tree.makePredictions(testData, this_tree, cols)
    percent = menu.test(testTarget, vote_predictions)
    print('You got %i%%' % percent)


def main(argsv):
    """preprocessors data and pass it to driver"""
    print('\nProcessing Votes...')
    vote = menu.votes
    trainingData, trainingTarget, testData, testTarget, cols = preProVotes(vote)
    runTreeClassifier(trainingData, trainingTarget, testData, testTarget, cols)
    print('\nProcessing Lenses...')
    lenses = menu.lenses
    trainingData, trainingTarget, testData, testTarget, cols = preProLenses(lenses)
    runTreeClassifier(trainingData, trainingTarget, testData, testTarget, cols)


if __name__ == '__main__':
    main(sys.argv)
