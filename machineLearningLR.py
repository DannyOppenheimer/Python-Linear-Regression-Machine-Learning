# Python Linear Regression Machine Learning using the Pandas and Sklearn Libraries
# Paulo Cortez, University of Minho, GuimarÃ£es, Portugal, http://www3.dsi.uminho.pt/pcortez
# UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/student+performance]
# required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import style
import pickle

# initial options for user
trainAI = False
wantTrain = input("Would you like to train the AI? [y/n]: ")
if wantTrain.lower() == "y" or wantTrain.lower() == "yes" or wantTrain.lower() == "yeah":
    trainAI = True
    timesTrain = input("How many training rounds? (don't break computer pl0x)? [num]: ")
else:
    trainAI = False

# read .csv file
studentData = pd.read_csv("Test Scores/student-dat.csv", sep=";")
studentData = studentData[["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrelquality", "freetime", "goout", "dayalcohol", "weekendalcohol", "health", "absences", "G1", "G2", "G3"]]

# we want to predict the end grade, so the prediction goal is G3
goalPrediction = "G3"

# X is the set without the final grade (G3) to find all the values we want to train off of
# Y is the final grad (G3) that we want to predict
X = np.array(studentData.drop(goalPrediction, 1))
Y = np.array(studentData[goalPrediction])

# x_train and y_train is training data the computer can read and learns off of
# x_test and y_test is the prediction data the computer is given to calculate new final grade
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# best current data set, saved in bestLearn.txt
listBest = open("Test Scores/bestLearn.txt", "r").readlines(1)
winningDatSet = float(listBest[0].translate({ord('b'): None}))

# has the training rounds yielded any better data sets?
foundMoreAccDat = False

# training section
if trainAI:
    for x in range(int(timesTrain)):
        # refresh the test pool with every run through
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

        # fit a linear regression like to the data collected
        linearRegress = linear_model.LinearRegression()
        linearRegress.fit(x_train, y_train)

        # accuracy of best fit line
        linearAcc = linearRegress.score(x_test, y_test)
        print("Training round ", x + 1, " had an accuracy of ", f, "%", sep="")

        # did any of the training rounds surpass the pickled data file's accuracy?
        if linearAcc > winningDatSet:
            foundMoreAccDat = True
            winningDatSet = linearAcc

            # save that value to a txt file
            with open("Test Scores/bestLearn.txt", "w") as h:
                h.write(str(winningDatSet))
            print("Training round ", x + 1, " has the running best accuracy of ", "%.01f" % (winningDatSet * 100), "%", sep="")

            # create a new pickle file if there isn't one, or write to an existing one
            with open("Test Scores/studentModel.pickle", "wb") as f:
                # dump linear regression into above pickle file
                pickle.dump(linearRegress, f)

    if foundMoreAccDat:
        print("Final accuracy: ", "%.01f" % (winningDatSet * 100), "%", sep="")
    else:
        print("This training run didn't get a higher accuracy than a saved one :( current set has an accuracy of: ", "%.01f" % (winningDatSet * 100), "%", sep="")


# open pickle file to access other information
pickleIn = open("Test Scores/studentModel.pickle", "rb")
linearRegressData = pickle.load(pickleIn)
print("\n***********************************************")
print("* Coefficients: ", linearRegressData.coef_)
print("* Intercept: ", linearRegressData.intercept_)
print("***********************************************")

# print out computer's predictions for the x_test data
print("\nHow to read: <guessed score> [students background] <actual score>")
predictions = linearRegressData.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# graph plotting to see the relationships for individual pieces of data
def ask():
    graphReq = input("\nWhich graph would you like to display?")
    try:
        style.use("ggplot")
        pyplot.scatter(studentData[graphReq], studentData["G3"])
        pyplot.xlabel(graphReq)
        pyplot.ylabel("Final Grade")
        pyplot.show()
    except KeyError:
        print("Oops! That's not an option. These are though!")
        print("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrelquality", "freetime", "goout", "dayalcohol", "weekendalcohol", "health", "absences", "G1", "G2", "G3", sep=", ")
        ask()


# prompt graph
exitRequest = input("Would you like to see a graph? [y/n]: ")

if exitRequest.lower() == "y" or exitRequest.lower() == "yes" or exitRequest.lower() == "yeah":
    ask()

# User indicates whether or not they would like to save the most accurate data set
overwriteLearn = input("Would you like to overwrite the most accurate data set and set it to 0? [y/n]: ")
if overwriteLearn.lower() == "y" or overwriteLearn.lower() == "yes" or overwriteLearn.lower() == "yeah":
    with open("Test Scores/bestLearn.txt", "w") as h:
        h.write(str("0.0"))